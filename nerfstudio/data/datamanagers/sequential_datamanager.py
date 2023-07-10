from dataclasses import field
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Type, Union

import torch
from future import __annotations__
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManagerConfig, VanillaDataManager, VanillaDataManagerConfig)
from nerfstudio.data.utils.dataloaders import ToggleCacheDataloader
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.misc import IterableWrapper
from nerfstudio.utils.rich_utils import CONSOLE


class SequentialDatamagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: SequentialDatamanager)

    num_initial_cameras: int = 5


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
            initial_image_idxs=list(range(self.config.num_initial_cameras)),
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras.to(self.device), self.train_camera_optimizer,
        )

    def enable_images(self, indices: List[int]) -> None:
        """Enables image to sample rays from it

        Args:
            indices (List[int]): _description_
        """
        self.train_image_dataloader.enable_images(indices)

    def disable_images(self, indices: List[int]) -> None:
        """Disables image to stop sampling rays from it

        Args:
            indices (List[int]): _description_
        """
        self.train_image_dataloader.disable_images(indices)
