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
Nerfacto augmented with dense monocular depth supervision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Type

import torch
from torch import nn

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig


@dataclass
class MonodepthNerfactoModelConfig(NerfactoModelConfig):
    """Additional parameters for depth supervision."""

    _target: Type = field(default_factory=lambda: MonodepthNerfactoModel)
    monodepth_loss_mult: float = 1e-2
    """Lambda of the depth loss."""
    depth_render_method: Literal["median", "expected"] = "expected"
    """Either render termination depth with median point or weighted depth average"""


class MonodepthNerfactoModel(NerfactoModel):
    """Depth loss augmented nerfacto model.

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: MonodepthNerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        num_images = self.num_train_data

        self.scale = nn.Parameter(torch.ones(num_images))
        self.shift = nn.Parameter(torch.zeros(num_images))

    def get_outputs(self, ray_bundle: RayBundle):
        outputs = super().get_outputs(ray_bundle)
        if ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata:
            outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)

        if self.training:
            gt_inverted_depth = batch["monodepth_image"].to(device=self.device, dtype=torch.float)
            image_indices = batch["indices"][:, 0]
            scale = self.scale[image_indices].unsqueeze(1)
            shift = self.shift[image_indices].unsqueeze(1)

            # Convert euclidean depth to z-depth
            depth = outputs["depth"] / outputs["directions_norm"]

            inverted_depth = 1 / (depth + 1e-8)
            inverted_depth = scale * inverted_depth + shift

            metrics_dict["monodepth_loss"] = torch.functional.F.mse_loss(inverted_depth, gt_inverted_depth)

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.training:
            assert metrics_dict is not None and "monodepth_loss" in metrics_dict
            loss_dict["monodepth_loss"] = self.config.monodepth_loss_mult * metrics_dict["monodepth_loss"]

        return loss_dict

    # def get_image_metrics_and_images(
    #     self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    # ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    #     """Appends ground truth depth to the depth image."""
    #     metrics, images = super().get_image_metrics_and_images(outputs, batch)
    #     ground_truth_depth = batch["depth_image"].to(self.device)
    #     if not self.config.is_euclidean_depth:
    #         ground_truth_depth = ground_truth_depth * outputs["directions_norm"]

    #     ground_truth_depth_colormap = colormaps.apply_depth_colormap(ground_truth_depth)
    #     predicted_depth_colormap = colormaps.apply_depth_colormap(
    #         outputs["depth"],
    #         accumulation=outputs["accumulation"],
    #         near_plane=float(torch.min(ground_truth_depth).cpu()),
    #         far_plane=float(torch.max(ground_truth_depth).cpu()),
    #     )
    #     images["depth"] = torch.cat([ground_truth_depth_colormap, predicted_depth_colormap], dim=1)
    #     depth_mask = ground_truth_depth > 0
    #     metrics["depth_mse"] = float(
    #         torch.nn.functional.mse_loss(outputs["depth"][depth_mask], ground_truth_depth[depth_mask]).cpu()
    #     )
    #     return metrics, images
