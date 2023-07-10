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

"""Processes an image sequence to a sequence of optical flow between consecutive images."""

import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import cv2
import numpy as np
import torch
from raft import RAFT
from raft.utils import InputPadder, flow_viz

from nerfstudio.process_data.base_converter_to_nerfstudio_dataset import BaseConverterToNerfstudioDataset
from nerfstudio.process_data.download_utils import download_file
from nerfstudio.process_data.process_data_utils import downscale_images
from nerfstudio.utils.rich_utils import CONSOLE, get_progress

MODELS_URL = "https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip?dl=1"


@dataclass
class ImagesToOpticalFlow:
    """Transforms a sequence of images to optical flow"""

    data: Path
    """Path the dataset in nerfstudio format"""

    verbose: bool = False

    model_cache_path: Path = Path("cache/raft")
    """Path to cache models weights"""

    skip_if_exists: bool = True
    """If True, skips flow calculation, for already processed frames"""

    flow_visualization: bool = False
    """If True, saves flow visualizations in human-readable format"""

    scale: float = 0.5
    """Images scale before flow calculation"""

    num_downscales: int = 3

    device: Literal["cpu", "cuda"] = "cuda"
    """Device to run RAFT model on"""

    device_ids: Tuple[int] = (0,)
    """If device is set to 'cuda', sets gpu ids to run model on"""

    raft_model: Literal["chairs", "kitti", "sintel", "small", "things"] = "things"
    """Raft model checkpoint"""

    small: bool = False
    """If True, smaller model of raft is used"""

    mixed_precision: bool = False
    """If True enables mixed precition in inference"""

    alternate_corr: bool = False
    """If True, alternative correlation implementation is used (more efficient, but needs jit compilation)"""

    @torch.no_grad()
    def main(self) -> None:
        model_path = self.model_cache_path / f"raft-{self.raft_model}.pth"

        if not model_path.exists():
            self._download_weights()

        # Initialize optical flow model
        model = torch.nn.DataParallel(
            RAFT(small=self.small, alternate_corr=self.alternate_corr, mixed_precision=self.mixed_precision),
            device_ids=self.device_ids,
        )
        model.load_state_dict(torch.load(model_path))

        if self.device == "cuda":
            model.module.to(f"cuda:{self.device_ids[0]}")

        model.eval()

        # Read and preprocess the video
        images_dir = self.data / "images"
        input_files = sorted(images_dir.iterdir())

        flow_dir = self.data / "optical_flow"
        flow_dir.mkdir(parents=True, exist_ok=True)

        flow_visualisation_dir = self.data / "flow_visualisation"
        if self.flow_visualization:
            flow_visualisation_dir.mkdir(parents=True, exist_ok=True)

        transforms_path = self.data / "transforms.json"
        transforms_json = None
        img_path_to_frame = None

        if transforms_path.exists():
            with open(transforms_path, "r", encoding="utf-8") as json_file:
                transforms_json = json.load(json_file)

            img_path_to_frame = {frame["file_path"]: frame for frame in transforms_json["frames"]}

        prev_frame_torch = None

        progress = get_progress("[bold yellow]Calculating optical flow...")
        with progress:
            for filepath in progress.track(input_files):
                fbase = filepath.stem
                flow_fwd_path = flow_dir / f"fwd_{fbase}.png"
                flow_bwd_path = flow_dir / f"bwd_{fbase}.png"

                flow_vis_fwd_path = flow_visualisation_dir / f"fwd_{fbase}.png"
                flow_vis_bwd_path = flow_visualisation_dir / f"bwd_{fbase}.png"

                # Read and rescale frame
                frame = cv2.imread(filepath.as_posix())
                height, width, _ = frame.shape

                ds_frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
                ds_frame = cv2.cvtColor(ds_frame, cv2.COLOR_BGR2RGB)

                # Convert to torch bhwc format and move to device
                frame_torch = torch.from_numpy(ds_frame).float()

                if self.device == "cuda":
                    frame_torch = frame_torch.to(f"cuda:{self.device_ids[0]}")

                frame_torch = torch.einsum("hwc->chw", frame_torch).unsqueeze(0)

                relative_img_path = filepath.relative_to(self.data).as_posix()
                relative_fwd_flow_path = flow_fwd_path.relative_to(self.data).as_posix()
                relative_bwd_flow_path = flow_bwd_path.relative_to(self.data).as_posix()

                if img_path_to_frame is not None and relative_img_path in img_path_to_frame:
                    frame_metadata = img_path_to_frame[relative_img_path]
                    frame_metadata["fwd_flow_file"] = relative_fwd_flow_path
                    frame_metadata["bwd_flow_file"] = relative_bwd_flow_path

                if (
                    self.skip_if_exists
                    and flow_fwd_path.exists()
                    and flow_bwd_path.exists()
                    and (flow_vis_fwd_path.exists() or not self.flow_visualization)
                    and (flow_vis_bwd_path.exists() or not self.flow_visualization)
                ):
                    prev_frame_torch = frame_torch
                    continue

                # Get optical flow
                if prev_frame_torch is not None:
                    image1 = torch.cat([prev_frame_torch, frame_torch], dim=0)
                    image2 = torch.cat([frame_torch, prev_frame_torch], dim=0)
                    padder = InputPadder(image1.shape)
                    image1, image2 = padder.pad(image1, image2)
                    _, flow_up = model(image1, image2, iters=30, test_mode=True)

                    fwd_flow = padder.unpad(flow_up[0]).cpu()
                    bwd_flow = padder.unpad(flow_up[1]).cpu()

                    fwd_flow = torch.functional.F.interpolate(
                        fwd_flow.unsqueeze(0), (height, width), mode="bilinear", align_corners=True
                    ).squeeze(0)

                    bwd_flow = torch.functional.F.interpolate(
                        bwd_flow.unsqueeze(0), (height, width), mode="bilinear", align_corners=True
                    ).squeeze(0)

                    fwd_flow = torch.einsum("chw->hwc", fwd_flow).numpy()
                    bwd_flow = torch.einsum("chw->hwc", bwd_flow).numpy()

                    mask_fwd = self._compute_fwdbwd_mask(fwd_flow, bwd_flow)
                    mask_bwd = self._compute_fwdbwd_mask(bwd_flow, fwd_flow)
                else:
                    fwd_flow = np.zeros((height, width, 2), dtype=float)
                    bwd_flow = np.zeros((height, width, 2), dtype=float)
                    mask_fwd = np.zeros((height, width), dtype=bool)
                    mask_bwd = np.zeros((height, width), dtype=bool)

                # Save flow
                cv2.imwrite(flow_fwd_path.as_posix(), self._encode_flow(fwd_flow, mask_fwd))
                cv2.imwrite(flow_bwd_path.as_posix(), self._encode_flow(bwd_flow, mask_bwd))

                if self.flow_visualization:
                    cv2.imwrite(flow_vis_fwd_path.as_posix(), flow_viz.flow_to_image(fwd_flow))
                    cv2.imwrite(flow_vis_bwd_path.as_posix(), flow_viz.flow_to_image(bwd_flow))

                prev_frame_torch = frame_torch

        downscale_status = downscale_images(
            flow_dir, self.num_downscales, folder_name="monodepths", verbose=self.verbose
        )

        if transforms_json is not None:
            with open(transforms_path, "w", encoding="utf-8") as json_file:
                json.dump(transforms_json, json_file, ensure_ascii=False, indent=4)

        CONSOLE.log("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

    def _download_weights(self):
        self.model_cache_path.mkdir(exist_ok=True, parents=True)
        model_zip_path = self.model_cache_path / "models.zip"

        downloaded = download_file(
            MODELS_URL,
            model_zip_path,
            description="[bold yellow]:inbox_tray: :inbox_tray: :inbox_tray: "
            "Downloading RAFT weights "
            ":inbox_tray: :inbox_tray: :inbox_tray:",
        )

        if not downloaded:
            raise RuntimeError("Cannot download weights for RAFT")

        with zipfile.ZipFile(model_zip_path, "r") as zip:
            zip.extractall(self.model_cache_path)

        models_dir = self.model_cache_path / "models"

        for filename in models_dir.iterdir():
            shutil.move(filename.as_posix(), self.model_cache_path)

        models_dir.rmdir()

        model_zip_path.unlink()

    def _compute_fwdbwd_mask(self, fwd_flow, bwd_flow, alpha=0.05, beta=0.5):
        bwd2fwd_flow = self._warp_flow(bwd_flow, fwd_flow)

        fwd_flow_norm = np.linalg.norm(fwd_flow, axis=-1)
        bwd2fwd_flow_norm = np.linalg.norm(bwd2fwd_flow, axis=-1)
        fwd_lr_error = np.linalg.norm(fwd_flow + bwd2fwd_flow, axis=-1)

        mask = fwd_lr_error < alpha * (fwd_flow_norm + bwd2fwd_flow_norm) + beta

        return mask

    @staticmethod
    def _warp_flow(img, flow):
        h, w, _ = flow.shape
        grid = np.mgrid[0:h, 0:w].transpose(1, 2, 0).astype(np.float32)

        res = cv2.remap(img, flow + grid, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return res

    @staticmethod
    def _encode_flow(flow, mask):
        flow = 1 ** 15 + flow * (2 ** 8)
        mask &= np.max(flow, axis=-1) < (2 ** 16 - 1)
        mask &= 0 < np.min(flow, axis=-1)
        return np.concatenate([flow.astype(np.uint16), mask[..., None].astype(np.uint16) * (2 ** 16 - 1)], axis=-1)

