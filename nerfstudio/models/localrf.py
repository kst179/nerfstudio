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

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import Parameter

from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.sequential_datamanager import SequentialDatamanager
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.encodings import TriplaneEncoding
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.tensorf_field import TensoRFField
from nerfstudio.models.base_model import Model, ModelConfig


@dataclass
class LocalRFModelConfig(ModelConfig):
    """LocalRF model config"""

    init_resolution = 64
    final_resolution = 300
    num_density_components = 16
    num_color_components = 48
    num_dir_sh_levels = 3
    head_mlp_layer_width = 128
    """Width of mlp head layer"""

    rf_aabb_size: float = 1.0

    num_initial_steps = 10000
    num_steps_per_field = 10000
    upsampling_every_iters: int = 1500
    add_camera_every_iters: int = 500

    disable_scene_contraction: bool = False


class LocalRFModel(Model):
    """LocalRF Model

    Args:
        config (LocalRFModelConfig): LocalRF model config to instantiate model
    """

    config: LocalRFModelConfig

    def __init__(self, config: LocalRFModelConfig, **kwargs):
        self.config = config
        super().__init__(config=config, **kwargs)

        self.added_cameras = []

    def upsample_current_field_callback(self, training_callback_attributes: TrainingCallbackAttributes) -> None:
        pass

    def add_new_camera(self, training_callback_attributes: TrainingCallbackAttributes) -> None:
        pipeline = training_callback_attributes.pipeline
        assert pipeline is not None

        datamanager = pipeline.datamanager
        assert isinstance(datamanager, SequentialDatamanager)

        train_cameras_optimizer: CameraOptimizer = datamanager.train_camera_optimizer

        last_added_camera = self.added_cameras[-1]
        last_added_camera_position = train_cameras_optimizer.pose_adjustment[last_added_camera, :3].detach()

        if last_added_camera:
            pass

        datamanager.enable_images([])

    def add_new_field(self, training_callback_attributes: TrainingCallbackAttributes) -> None:
        pass

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns a list of callbacks that run functions at the specified training iterations."""

        callbacks = [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                func=self.upsample_current_field_callback,
                update_every_num_iters=self.config.upsampling_every_iters,
            ),
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                func=self.add_new_camera,
                update_every_num_iters=self.config.add_camera_every_iters,
            ),
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                func=self.upsample_current_field_callback,
                update_every_num_iters=self.config.num_steps_per_field,
            ),
        ]

        return callbacks

    def populate_modules(self) -> None:
        """Set the necessary modules to get the network working."""

        super().populate_modules()

        if self.config.disable_scene_contraction:
            self.spatial_distortion = None
        else:
            self.spatial_distortion = SceneContraction()

        self.radiance_fields = []

        init_aabb = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]) * self.config.rf_aabb_size

        field = self.create_new_field(init_aabb)

        self.current_field_idx = 0
        self.current_field = field

    def create_new_field(self, aabb: Float[Tensor, "2 3"]) -> TensoRFField:
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
        return super().get_param_groups()

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, Tensor | List]:
        return super().get_outputs(ray_bundle)

    def forward(self, ray_bundle: RayBundle) -> Dict[str, Tensor | List]:
        return super().forward(ray_bundle)

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        return super().get_metrics_dict(outputs, batch)

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, Tensor]:
        return super().get_loss_dict(outputs, batch, metrics_dict)

    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, Tensor]:
        return super().get_outputs_for_camera_ray_bundle(camera_ray_bundle)

    def get_image_metrics_and_images(
        self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, Tensor]]:
        return super().get_image_metrics_and_images(outputs, batch)

    def load_model(self, loaded_state: Dict[str, Any]) -> None:
        return super().load_model(loaded_state)

    def update_to_step(self, step: int) -> None:
        return super().update_to_step(step)
