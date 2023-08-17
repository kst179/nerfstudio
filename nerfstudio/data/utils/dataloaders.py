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
Code for sampling images from a dataset of images.
"""

# for multithreading
import concurrent.futures
import multiprocessing
import random
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sized, Tuple, Union

import torch
from rich.progress import track
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.utils.misc import get_dict_to_torch
from nerfstudio.utils.rich_utils import CONSOLE


class CacheDataloader(DataLoader):
    """Collated image dataset that implements caching of default-pytorch-collatable data.
    Creates batches of the InputDataset return type.

    Args:
        dataset: Dataset to sample from.
        num_samples_to_collate: How many images to sample rays for each batch. -1 for all images.
        num_times_to_repeat_images: How often to collate new images. -1 to never pick new images.
        device: Device to perform computation.
        collate_fn: The function we will use to collate our training data
    """

    def __init__(
        self,
        dataset: Dataset,
        num_images_to_sample_from: int = -1,
        num_times_to_repeat_images: int = -1,
        device: Union[torch.device, str] = "cpu",
        collate_fn: Callable[[Any], Any] = nerfstudio_collate,
        exclude_batch_keys_from_device: Optional[List[str]] = None,
        **kwargs,
    ):
        if exclude_batch_keys_from_device is None:
            exclude_batch_keys_from_device = ["image"]
        self.dataset = dataset
        assert isinstance(self.dataset, Sized)

        super().__init__(dataset=dataset, **kwargs)  # This will set self.dataset
        self.num_times_to_repeat_images = num_times_to_repeat_images
        self.cache_all_images = (num_images_to_sample_from == -1) or (num_images_to_sample_from >= len(self.dataset))
        self.num_images_to_sample_from = len(self.dataset) if self.cache_all_images else num_images_to_sample_from
        self.device = device
        self.collate_fn = collate_fn
        self.num_workers = kwargs.get("num_workers", 0)
        self.exclude_batch_keys_from_device = exclude_batch_keys_from_device

        self.num_repeated = self.num_times_to_repeat_images  # starting value
        self.first_time = True

        self.cached_collated_batch = None
        if self.cache_all_images:
            CONSOLE.print(f"Caching all {len(self.dataset)} images.")
            if len(self.dataset) > 500:
                CONSOLE.print(
                    "[bold yellow]Warning: If you run out of memory, try reducing the number of images to sample from."
                )
            self.cached_collated_batch = self._get_collated_batch()
        elif self.num_times_to_repeat_images == -1:
            CONSOLE.print(
                f"Caching {self.num_images_to_sample_from} out of {len(self.dataset)} images, without resampling."
            )
        else:
            CONSOLE.print(
                f"Caching {self.num_images_to_sample_from} out of {len(self.dataset)} images, "
                f"resampling every {self.num_times_to_repeat_images} iters."
            )

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def _get_batch_list(self):
        """Returns a list of batches from the dataset attribute."""

        assert isinstance(self.dataset, Sized)
        indices = random.sample(range(len(self.dataset)), k=self.num_images_to_sample_from)
        batch_list = []
        results = []

        num_threads = int(self.num_workers) * 4
        num_threads = min(num_threads, multiprocessing.cpu_count() - 1)
        num_threads = max(num_threads, 1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for idx in indices:
                res = executor.submit(self.dataset.__getitem__, idx)
                results.append(res)

            for res in track(results, description="Loading data batch", transient=True):
                batch_list.append(res.result())

        return batch_list

    def _get_collated_batch(self):
        """Returns a collated batch."""
        batch_list = self._get_batch_list()
        collated_batch = self.collate_fn(batch_list)
        collated_batch = get_dict_to_torch(
            collated_batch, device=self.device, exclude=self.exclude_batch_keys_from_device
        )
        return collated_batch

    def __iter__(self):
        while True:
            if self.cache_all_images:
                collated_batch = self.cached_collated_batch
            elif self.first_time or (
                self.num_times_to_repeat_images != -1 and self.num_repeated >= self.num_times_to_repeat_images
            ):
                # trigger a reset
                self.num_repeated = 0
                collated_batch = self._get_collated_batch()
                # possibly save a cached item
                self.cached_collated_batch = collated_batch if self.num_times_to_repeat_images != 0 else None
                self.first_time = False
            else:
                collated_batch = self.cached_collated_batch
                self.num_repeated += 1
            yield collated_batch


class ToggleCacheDataloader(DataLoader):
    """Collated image data loader that loads and caches images by request and can remove them from cache.
    While not recieving requests, it yields the same cached batch of images.

    Args:
        dataset (InputDataset): _description_
        device (Union[torch.device, str], optional): _description_. Defaults to "cpu".
        collate_fn (Callable[[Any], Any], optional): _description_. Defaults to nerfstudio_collate.
        exclude_batch_keys_from_device (Optional[List[str]], optional): _description_. Defaults to None.
        initial_image_idxs (List, optional): _description_. Defaults to [].
    """

    def __init__(
        self,
        dataset: InputDataset,
        device: Union[torch.device, str] = "cpu",
        collate_fn: Callable[[Any], Any] = nerfstudio_collate,
        exclude_batch_keys_from_device: Optional[List[str]] = None,
        # initial_image_idxs: List = [],
        **kwargs,
    ):
        super().__init__(dataset=dataset, **kwargs)

        self.device = device
        self.collate_fn = collate_fn
        self.num_workers = kwargs.get("num_workers", 0)
        self.exclude_batch_keys_from_device = exclude_batch_keys_from_device

        # self.first_time = True
        self.cached_collated_batch = None

        self.selected_image_idxs = set()
        # self.initial_image_idxs = initial_image_idxs

    def enable_images(self, indices: List[int]):
        # Select only images which are not loaded yet
        indices = [idx for idx in indices if idx not in self.selected_image_idxs]

        if len(indices) == 0:
            return

        batch_list = self._get_selection_as_batch_list(indices)
        batch_list = get_dict_to_torch(batch_list, device=self.device, exclude=self.exclude_batch_keys_from_device)
        collated_batch = self.collate_fn(batch_list)

        # Merge collated batches
        if self.cached_collated_batch is None:
            self.cached_collated_batch = collated_batch
        else:
            for key, value in collated_batch.items():
                cached_value = self.cached_collated_batch[key]

                if isinstance(cached_value, list):
                    cached_value.extend(value)
                elif isinstance(cached_value, torch.Tensor):
                    cached_value = torch.cat([cached_value, value], dim=0)

                self.cached_collated_batch[key] = cached_value

        self.selected_image_idxs.update(indices)

    def disable_images(self, indices: List[int]):
        assert self.cached_collated_batch is not None

        self.selected_image_idxs.difference_update(indices)

        collated_batch = self.cached_collated_batch
        mask = [idx in self.selected_image_idxs for idx in collated_batch["image_idx"]]
        mask = torch.tensor(mask)
        for key, value in collated_batch.items():
            if isinstance(value, torch.Tensor):
                collated_batch[key] = value[mask.to(value.device)]
            else:
                collated_batch[key] = [item for i, item in enumerate(value) if mask[i]]

    def _get_selection_as_batch_list(self, indices: List[int]):
        """Returns a list of batches from the dataset attribute."""

        assert isinstance(self.dataset, Sized)
        batch_list = []
        results = []

        num_threads = int(self.num_workers) * 4
        num_threads = min(num_threads, len(indices))
        num_threads = min(num_threads, multiprocessing.cpu_count() - 1)
        num_threads = max(num_threads, 1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for idx in indices:
                res = executor.submit(self.dataset.__getitem__, idx)
                results.append(res)

            for res in track(results, description="Loading data batch", transient=True):
                batch_list.append(res.result())

        return batch_list

    def __iter__(self):
        while True:
            # if self.first_time:
            #     self.enable_images(self.initial_image_idxs)
            #     self.first_time = False

            yield self.cached_collated_batch


class EvalDataloader(DataLoader):
    """Evaluation dataloader base class

    Args:
        input_dataset: InputDataset to load data from
        device: Device to load data to
    """

    def __init__(
        self,
        input_dataset: InputDataset,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        self.input_dataset = input_dataset
        self.cameras = input_dataset.cameras.to(device)
        self.device = device
        self.kwargs = kwargs
        super().__init__(dataset=input_dataset)

    @abstractmethod
    def __iter__(self):
        """Iterates over the dataset"""
        return self

    @abstractmethod
    def __next__(self) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data"""

    def get_camera(self, image_idx: int = 0) -> Cameras:
        """Get camera for the given image index

        Args:
            image_idx: Camera image index
        """
        return self.cameras[image_idx]

    def get_data_from_image_idx(self, image_idx: int) -> Tuple[RayBundle, Dict]:
        """Returns the data for a specific image index.

        Args:
            image_idx: Camera image index
        """
        ray_bundle = self.cameras.generate_rays(camera_indices=image_idx, keep_shape=True)
        batch = self.input_dataset[image_idx]
        batch = get_dict_to_torch(batch, device=self.device, exclude=["image"])
        assert isinstance(batch, dict)
        return ray_bundle, batch


class FixedIndicesEvalDataloader(EvalDataloader):
    """Dataloader that returns a fixed set of indices.

    Args:
        input_dataset: InputDataset to load data from
        image_indices: List of image indices to load data from. If None, then use all images.
        device: Device to load data to
    """

    def __init__(
        self,
        input_dataset: InputDataset,
        image_indices: Optional[Tuple[int]] = None,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        super().__init__(input_dataset, device, **kwargs)
        if image_indices is None:
            self.image_indices = list(range(len(input_dataset)))
        else:
            self.image_indices = image_indices
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < len(self.image_indices):
            image_idx = self.image_indices[self.count]
            ray_bundle, batch = self.get_data_from_image_idx(image_idx)
            self.count += 1
            return ray_bundle, batch
        raise StopIteration


class RandIndicesEvalDataloader(EvalDataloader):
    """Dataloader that returns random images.
    Args:
        input_dataset: InputDataset to load data from
        device: Device to load data to
    """

    def __iter__(self):
        return self

    def __next__(self):
        # choose a random image index
        image_idx = random.randint(0, len(self.cameras) - 1)
        ray_bundle, batch = self.get_data_from_image_idx(image_idx)
        return ray_bundle, batch
