# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LocalRF implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Tuple, Type

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.datamanagers.sequential_datamanager import SequentialDatamanager
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.encodings import TensorCPEncoding, TensorVMEncoding, TriplaneEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.tensorf_field import TensoRFField
from nerfstudio.model_components.losses import MSELoss, nerfstudio_distortion_loss, tv_loss
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, misc

TensorEncoding = TensorCPEncoding | TensorVMEncoding | TriplaneEncoding


@dataclass
class LocalRFModelConfig(ModelConfig):
    """LocalRF model config"""

    _target: Type = field(default_factory=lambda: LocalRFModel)

    # Field params
    num_density_components = 16
    """number of components in density field tensor decomposition"""
    num_color_components = 48
    """number of components in color field tensor decomposition"""
    num_dir_sh_levels = 3
    """number of levels in spherical harmonics enconding for view direction"""
    head_mlp_layer_width = 128
    """width of mlp head layer"""
    field_aabb_size: float = 1.0
    """size of a bounding box in which field is uncontracted"""

    # Sampling params
    num_uniform_samples: int = 200
    """number of rough samples"""
    num_samples: int = 128
    """number of fine samples"""

    # Training params
    num_initial_cameras: int = 5
    """number of cameras to start optimization from"""
    use_prior_transforms: bool = False
    """if True, camera positions from transforms.json are used"""
    min_camera_distance: float = 0.1
    """minimal distance between sequential cameras in video,
    if camera is further than this treshold, camera considered as outlier and prior info not used"""
    num_steps_per_field: int = 10000
    """number of steps to train a single radiance field"""
    init_resolution = 64
    """initial tensorf field resolution"""
    final_resolution = 300
    """maximal tensorf field resolution"""
    upsampling_every_iters: int = 1500
    """frequency of field resolution upsampling"""
    num_upsampling_steps: int = 5
    """total number of field upsampling steps, before reaching final resolution"""
    add_camera_every_iters: int = 100  # 500
    """frequency of adding a new camera to the optimization process"""
    scene_contraction_norm: Literal["l2", "l-inf"] = "l-inf"
    """scene contraction norm order, if l2 then scene is contracted into a ball of radius 2,
    if l-inf then scene is contracted into aabb [-2, 2]"""
    regularization: Literal["none", "l1", "tv"] = "l1"
    """regularization method used in tensorf paper"""
    use_distortion_loss: bool = True
    """if true distortion loss from mipNerf360 is used for regularization"""

    loss_coefficients: Dict[str, float] = to_immutable_dict(
        {
            "rgb_loss": 1.0,
            "tv_reg_density": 1e-3,
            "tv_reg_color": 1e-4,
            "l1_reg": 5e-4,
            "distortion_loss": 2e-3,
        }
    )
    """parameters to instantiate density field with"""


class LocalRFModel(Model):
    """LocalRF Model

    Args:
        config (LocalRFModelConfig): LocalRF model config to instantiate model
    """

    config: LocalRFModelConfig

    def __init__(self, config: LocalRFModelConfig, **kwargs):
        self.config = config

        self.contraction_order = torch.inf if self.config.scene_contraction_norm == "l-inf" else 2
        self.active_cameras = []

        self.upsampling_resolutions = (
            torch.linspace(
                np.log(config.init_resolution),
                np.log(config.final_resolution),
                config.num_upsampling_steps + 1,
            )
            .exp()
            .round()
            .long()
            .tolist()
        )

        self.next_upsampling_step = 0
        self.refine = False

        super().__init__(config=config, **kwargs)

    def before_training_callback(self, training_callback_attributes: TrainingCallbackAttributes, step: int) -> None:
        pipeline = training_callback_attributes.pipeline
        assert pipeline is not None

        datamanager = pipeline.datamanager
        assert isinstance(datamanager, SequentialDatamanager)

        initial_cameras = list(range(self.config.num_initial_cameras))
        datamanager.enable_cameras(initial_cameras)
        self.active_cameras.extend(initial_cameras)

    def upsample_current_field_callback(
        self, training_callback_attributes: TrainingCallbackAttributes, step: int
    ) -> None:
        # if current field have final resolution, then do nothing
        if self.next_upsampling_step >= self.config.num_upsampling_steps:
            return

        # skip first resolution setup as it is initialized during field creation
        if self.next_upsampling_step == 0:
            self.next_upsampling_step += 1
            return

        optimizers = training_callback_attributes.optimizers
        pipeline = training_callback_attributes.pipeline

        assert optimizers is not None
        assert pipeline is not None

        # fetch next level resolution
        index = self.next_upsampling_step
        self.next_upsampling_step += 1
        resolution = self.upsampling_resolutions[index]

        assert isinstance(self.current_field.density_encoding, TensorEncoding)
        assert isinstance(self.current_field.color_encoding, TensorEncoding)

        # upsample the position and direction grids
        self.current_field.density_encoding.upsample_grid(resolution)
        self.current_field.color_encoding.upsample_grid(resolution)

        # reinitialize the encodings optimizer
        optimizers_config = optimizers.config
        enc = pipeline.get_param_groups()["encodings"]
        lr_init = optimizers_config["encodings"]["optimizer"].lr

        optimizers.optimizers["encodings"] = optimizers_config["encodings"]["optimizer"].setup(params=enc)
        if optimizers_config["encodings"]["scheduler"]:
            optimizers.schedulers["encodings"] = (
                optimizers_config["encodings"]["scheduler"]
                .setup()
                .get_scheduler(optimizer=optimizers.optimizers["encodings"], lr_init=lr_init)
            )

    @staticmethod
    def _get_adjusted_camera_translation(
        camera_idx: int, cameras: Cameras, pose_optimizer: CameraOptimizer
    ) -> Float[Tensor, "3"]:
        camera_c2w = cameras.camera_to_worlds[camera_idx]

        # R, t <- [R | t]
        c2w_rotation, c2w_translation = camera_c2w.split([3, 1], dim=1)
        # t, w <- [t, w], w is axis-angle adjustment rotation representation, not needed here
        translation_adjustment, _ = pose_optimizer.pose_adjustment[camera_idx].detach().split([3, 3])

        # Calculate adjusted translation (in world coords)
        c2w_translation = c2w_rotation @ translation_adjustment + c2w_translation

        return c2w_translation

    @staticmethod
    def _check_point_in_aabb(
        point: Float[Tensor, "3"],
        aabb: Float[Tensor, "2 3"],
    ):
        return (aabb[0] < point).all() and (point < aabb[1]).all()

    def add_new_camera_callback(self, training_callback_attributes: TrainingCallbackAttributes, step: int) -> None:
        pipeline = training_callback_attributes.pipeline
        assert pipeline is not None

        datamanager = pipeline.datamanager
        assert isinstance(datamanager, SequentialDatamanager)

        ray_generator: RayGenerator = datamanager.train_ray_generator
        cameras: Cameras = ray_generator.cameras
        pose_optimizer: CameraOptimizer = ray_generator.pose_optimizer

        # Fetch last added camera's translation
        last_added_camera = self.active_cameras[-1]
        last_added_camera_translation = self._get_adjusted_camera_translation(
            last_added_camera, cameras, pose_optimizer
        )

        # If camera is outside the circle (or aabb) of radius 1 (uncontracted space)
        # then stop adding new cameras and start field refining
        current_aabb_center = self.current_field.aabb.mean(dim=0)
        if torch.norm(last_added_camera_translation - current_aabb_center, self.contraction_order) > 1.0:
            self.refine = True
            return

        # Turn on the new camera (add it to the sampling and optimizing processes)
        next_camera = last_added_camera + 1
        datamanager.enable_cameras([next_camera])
        self.active_cameras.append(next_camera)

    def add_new_field_callback(self, training_callback_attributes: TrainingCallbackAttributes, step: int) -> None:
        if step == 0:
            return

        pipeline = training_callback_attributes.pipeline
        optimizers = training_callback_attributes.optimizers
        assert pipeline is not None
        assert optimizers is not None

        datamanager = pipeline.datamanager
        assert isinstance(datamanager, SequentialDatamanager)

        cameras: Cameras = datamanager.train_ray_generator.cameras
        pose_optimizer: CameraOptimizer = datamanager.train_ray_generator.pose_optimizer

        last_added_camera = self.active_cameras[-1]
        last_added_camera_translation = self._get_adjusted_camera_translation(
            last_added_camera, cameras, pose_optimizer
        )

        old_field = self.current_field

        # create new field with the last camera in the center of it's aabb
        new_field = self.create_new_field(last_added_camera_translation, self.device)

        # disable cameras outside the new and old fields
        # freeze cameras outside the new field
        cameras_to_freeze = []
        cameras_to_disable = []
        new_active_cameras = []

        for camera in self.active_cameras:
            camera_translation = self._get_adjusted_camera_translation(camera, cameras, pose_optimizer)

            in_old_field = self._check_point_in_aabb(camera_translation, old_field.aabb)
            in_new_field = self._check_point_in_aabb(camera_translation, new_field.aabb)

            if in_old_field and not in_new_field:
                cameras_to_freeze.append(camera)

            if not in_old_field and not in_new_field:
                cameras_to_disable.append(camera)
            else:
                new_active_cameras.append(camera)

        datamanager.disable_cameras(cameras_to_disable)
        datamanager.freeze_cameras(cameras_to_freeze)

        # move current field to cpu (to reduce vram usage)
        self.fields[self.current_field_idx] = self.current_field.cpu()
        self.current_field_idx += 1

        # move new field onto device
        self.current_field = new_field.to(self.device)

        self.fields.append(new_field)

        # reinitialize the encodings optimizer with new field's weights
        optimizers_config = optimizers.config
        param_groups = pipeline.get_param_groups()

        for group in ["encodings", "fields"]:
            params = param_groups[group]
            lr_init = optimizers_config[group]["optimizer"].lr

            optimizers.optimizers[group] = optimizers_config[group]["optimizer"].setup(params=params)
            if optimizers_config[group]["scheduler"]:
                optimizers.schedulers[group] = (
                    optimizers_config[group]["scheduler"]
                    .setup()
                    .get_scheduler(optimizer=optimizers.optimizers[group], lr_init=lr_init)
                )

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                func=self.before_training_callback,
                iters=(0,),
                args=[training_callback_attributes],
            ),
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                func=self.upsample_current_field_callback,
                update_every_num_iters=self.config.upsampling_every_iters,
                args=[training_callback_attributes],
            ),
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                func=self.add_new_camera_callback,
                update_every_num_iters=self.config.add_camera_every_iters,
                args=[training_callback_attributes],
            ),
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                func=self.add_new_field_callback,
                update_every_num_iters=self.config.num_steps_per_field,
                args=[training_callback_attributes],
            ),
        ]

        return callbacks

    def populate_modules(self) -> None:
        super().populate_modules()

        self.spatial_distortion = SceneContraction(order=self.contraction_order)

        self.fields = []

        init_center = torch.zeros(3)
        field = self.create_new_field(init_center)

        self.current_field_idx = 0
        self.current_field = field

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_uniform_samples, single_jitter=True)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_samples, single_jitter=True, include_original=False)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def create_new_field(self, center: Float[Tensor, "3"], device=None) -> TensoRFField:
        aabb = (
            torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=torch.float, device=device)
            * self.config.field_aabb_size
            + center
        )

        density_encoding = TriplaneEncoding(
            resolution=self.config.init_resolution, num_components=self.config.num_density_components
        )
        color_encoding = TriplaneEncoding(
            resolution=self.config.init_resolution, num_components=self.config.num_color_components
        )

        field = TensoRFField(
            aabb,
            density_encoding=density_encoding,
            color_encoding=color_encoding,
            head_mlp_layer_width=self.config.head_mlp_layer_width,
            use_sh=True,
            sh_levels=3,
            spatial_distortion=self.spatial_distortion,
        )

        return field

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        fields_group = []
        fields_group.extend(self.current_field.mlp_head.parameters())
        fields_group.extend(self.current_field.B.parameters())
        fields_group.extend(self.current_field.field_output_rgb.parameters())

        encodings_group = []
        encodings_group.extend(self.current_field.density_encoding.parameters())
        encodings_group.extend(self.current_field.color_encoding.parameters())

        param_groups = {
            "fields": fields_group,
            "encodings": encodings_group,
        }

        return param_groups

    def _select_closest_field(self, camera_origin: Float[Tensor, "3"]) -> None:
        if self._check_point_in_aabb(camera_origin, self.current_field.aabb):
            return

        closest_field_idx = None
        closest_field_distance = torch.inf

        for field_idx, tensorf_field in enumerate(self.fields):
            field_distance = torch.norm(camera_origin - tensorf_field.aabb.mean(dim=0))

            if self._check_point_in_aabb(camera_origin, tensorf_field.aabb) and field_distance < closest_field_distance:
                closest_field_distance = field_distance
                closest_field_idx = field_idx

        assert closest_field_idx is not None

        self.current_field.cpu()
        self.current_field = self.fields[closest_field_idx].to(self.device)

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, Tensor | List]:
        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        densities = self.current_field.get_density(ray_samples_uniform)
        weights = ray_samples_uniform.get_weights(densities)
        coarse_accumulation = self.renderer_accumulation(weights)
        acc_mask = (coarse_accumulation > 0.0001).reshape(-1)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights)

        # fine field:
        field_outputs_fine = self.current_field.forward(
            ray_samples_pdf, mask=acc_mask, bg_color=colors.WHITE.to(weights.device)
        )

        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])

        accumulation = self.renderer_accumulation(weights_fine)
        depth = self.renderer_depth(weights_fine, ray_samples_pdf)

        rgb = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )

        rgb = torch.where(accumulation < 0, colors.WHITE.to(rgb.device), rgb)
        accumulation = torch.clamp(accumulation, min=0)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        if self.training and self.config.use_distortion_loss:
            outputs.update({"weights": weights_fine, "ray_samples": ray_samples_pdf})

        return outputs

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)

        if self.training and self.config.use_distortion_loss:
            metrics_dict["distortion_loss"] = nerfstudio_distortion_loss(
                ray_samples=outputs["ray_samples"],
                weights=outputs["weights"],
            ).mean()

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, Tensor]:
        # Scaling metrics by coefficients to create the losses.
        image = batch["image"].to(self.device)

        rgb_loss = self.rgb_loss(image, outputs["rgb"])

        loss_dict = {"rgb_loss": rgb_loss}

        if metrics_dict is not None and "distortion_loss" in metrics_dict:
            loss_dict["distortion_loss"] = metrics_dict["distortion_loss"]

        if self.config.regularization == "l1":
            l1_parameters = []
            for parameter in self.current_field.density_encoding.parameters():
                l1_parameters.append(parameter.view(-1))
            loss_dict["l1_reg"] = torch.abs(torch.cat(l1_parameters)).mean()

        elif self.config.regularization == "tv":
            density_plane_coef = self.current_field.density_encoding.plane_coef
            color_plane_coef = self.current_field.color_encoding.plane_coef
            assert isinstance(color_plane_coef, torch.Tensor) and isinstance(
                density_plane_coef, torch.Tensor
            ), "TV reg only supported for TensoRF encoding types with plane_coef attribute"
            loss_dict["tv_reg_density"] = tv_loss(density_plane_coef)
            loss_dict["tv_reg_color"] = tv_loss(color_plane_coef)

        elif self.config.regularization == "none":
            pass
        else:
            raise ValueError(f"Regularization {self.config.regularization} not supported")

        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """

        # check if all the rays are from the same camera and select closest field for it
        camera_origin = camera_ray_bundle.origins[0]
        assert torch.norm(camera_ray_bundle.origins - camera_origin) < 1e-6
        self._select_closest_field(camera_origin)

        return super().get_outputs_for_camera_ray_bundle(camera_ray_bundle)

    def get_image_metrics_and_images(
        self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, Tensor]]:
        image = batch["image"].to(outputs["rgb"].device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        assert self.config.collider_params is not None
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        assert isinstance(ssim, Tensor)

        metrics_dict = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim.item()),
            "lpips": float(lpips.item()),
        }
        images_dict = {"img": combined_rgb, "accumulation": acc, "depth": depth}
        return metrics_dict, images_dict

    def load_model(self, loaded_state: Dict[str, Any]) -> None:
        return super().load_model(loaded_state)

    def update_to_step(self, step: int) -> None:
        return super().update_to_step(step)
