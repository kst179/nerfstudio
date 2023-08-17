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

from dataclasses import dataclass, field
from typing import List, Literal, Type, Union

import torch

import nerfstudio.utils.poses as poses_utils
from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.data.utils.dataloaders import ToggleCacheDataloader
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class SequentialDatamagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: SequentialDatamanager)

    # num_initial_cameras: int = 5


class SequentialDatamanager(VanillaDataManager):
    def __init__(
        self,
        config: SequentialDatamagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.config: SequentialDatamagerConfig
        super().__init__(config, device, test_mode, world_size, local_rank, **kwargs)

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = ToggleCacheDataloader(
            self.train_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
            # initial_image_idxs=list(range(self.config.num_initial_cameras)),
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras.to(self.device),
            self.train_camera_optimizer,
        )

    def enable_cameras(self, indices: List[int]) -> None:
        """Enables image to sample rays from it

        Args:
            indices (List[int]): indices of cameras to enable
        """
        self.train_image_dataloader.enable_images(indices)

    def disable_cameras(self, indices: List[int]) -> None:
        """Disables image to stop sampling rays from it

        Args:
            indices (List[int]): indices of cameras to disable
        """
        self.train_image_dataloader.disable_images(indices)

    def freeze_cameras(self, indices: List[int], apply_pose_adjustment: bool = True) -> None:
        """Stops cameras' position optimization

        Args:
            indices (List[int]): list of camera indices to freeze (disable adjustment optimization)
            apply_pose_adjustment (bool): if True, the learned poses are applied to initial cameras' position
        """
        pose_optimizer: CameraOptimizer = self.train_ray_generator.pose_optimizer
        cameras: Cameras = self.train_ray_generator.cameras

        indices_tensor = torch.tensor(indices, device=pose_optimizer.device)

        # Remove already freezed indices
        if pose_optimizer.non_trainable_camera_indices is not None:
            already_freezed_indices = torch.isin(indices_tensor, pose_optimizer.non_trainable_camera_indices)
            indices_tensor = indices_tensor[~already_freezed_indices]

        # Apply adjustments to static camera poses
        if apply_pose_adjustment:
            cameras.camera_to_worlds[indices_tensor] = poses_utils.multiply(
                cameras.camera_to_worlds[indices], pose_optimizer.pose_adjustment[indices]
            )

        # Concat new freezed camera indices with the old one
        if pose_optimizer.non_trainable_camera_indices is None:
            pose_optimizer.non_trainable_camera_indices = indices_tensor
            return

        pose_optimizer.non_trainable_camera_indices = torch.cat(
            (pose_optimizer.non_trainable_camera_indices, indices_tensor),
            dim=0,
        )
