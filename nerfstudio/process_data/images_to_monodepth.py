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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
import util.io
from dpt.models import DPTDepthModel
from dpt.transforms import NormalizeImage, PrepareForNet, Resize
from torchvision.transforms import Compose

from nerfstudio.process_data.base_converter_to_nerfstudio_dataset import \
    BaseConverterToNerfstudioDataset
from nerfstudio.process_data.download_utils import download_file
from nerfstudio.process_data.process_data_utils import downscale_images
from nerfstudio.utils.rich_utils import CONSOLE, get_progress

MODEL_CONFIGS = {
    "dpt_large": {
        "net_h": 384,
        "net_w": 384,
        "scale": 1.0,
        "shift": 0.0,
        "backbone": "vitl16_384",
        "non_negative": True,
        "enable_attention_hooks": False,
        "weights": "https://github.com/isl-org/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
    },
    "dpt_hybrid": {
        "net_h": 384,
        "net_w": 384,
        "scale": 1.0,
        "shift": 0.0,
        "backbone": "vitb_rn50_384",
        "non_negative": True,
        "enable_attention_hooks": False,
        "weights": "https://github.com/isl-org/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt",
    },
    "dpt_hybrid_kitti": {
        "net_h": 352,
        "net_w": 1216,
        "scale": 0.00006016,
        "shift": 0.00579,
        "backbone": "vitb_rn50_384",
        "non_negative": True,
        "enable_attention_hooks": False,
        "weights": "https://github.com/isl-org/DPT/releases/download/1_0/dpt_hybrid_kitti-cb926ef4.pt",
    },
    "dpt_hybrid_nyu": {
        "net_h": 480,
        "net_w": 640,
        "scale": 0.000305,
        "shift": 0.1378,
        "backbone": "vitb_rn50_384",
        "non_negative": True,
        "enable_attention_hooks": False,
        "weights": "https://github.com/isl-org/DPT/releases/download/1_0/dpt_hybrid_nyu-2ce69ec7.pt",
    },
}


@dataclass
class ImagesToMonodepth(BaseConverterToNerfstudioDataset):
    model_cache_path: Path = Path("cache/dpt")
    """Path to cache models weights"""

    device: Literal["cpu", "cuda"] = "cuda"
    """Device to run model on"""

    dpt_model: Literal["dpt_large", "dpt_hybrid", "dpt_hybrid_kitti", "dpt_hybrid_nyu"] = "dpt_large"
    """DPT model which would be used for monocular depth estimation"""

    optimize: bool = True
    """If True, does calculations in half precision, in channels-last memory order (faster)"""

    absolute_depth: bool = False
    """If True"""

    num_downscales: int = 3
    """Number of times to downscale the depth images. Downscales by 2 each time."""

    @torch.no_grad()
    def main(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        config = MODEL_CONFIGS[self.dpt_model]
        model_path = self.model_cache_path / f"{self.dpt_model}.pt"

        # Download weights if not found
        if not model_path.exists():
            downloaded = download_file(
                config["weights"],
                model_path,
                description=f":inbox_tray: :inbox_tray: :inbox_tray: "
                f"Downloading weights for {self.dpt_model} "
                f":inbox_tray: :inbox_tray: :inbox_tray:",
            )
            if not downloaded:
                return

        net_h = config["net_h"]
        net_w = config["net_w"]

        model = DPTDepthModel(
            path=model_path,
            scale=config["scale"],
            shift=config["shift"],
            backbone=config["backbone"],
            non_negative=config["non_negative"],
            enable_attention_hooks=config["enable_attention_hooks"],
        )

        transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                PrepareForNet(),
            ]
        )

        model.eval()

        # Hardware convolution optimization
        if self.optimize and self.device.startswith("cuda"):
            model = model.to(memory_format=torch.channels_last)
            model = model.half()

        model.to(self.device)

        # Load images
        paths_to_images = sorted(self.data.iterdir())

        # Create output dir
        monodepth_dir = self.output_dir / "monodepths"
        monodepth_dir.mkdir(parents=True, exist_ok=True)

        transforms_path = self.output_dir / "transforms.json"

        transforms_json = None
        img_path_to_frame = None

        if transforms_path.exists():
            with open(transforms_path, "r", encoding="utf-8") as json_file:
                transforms_json = json.load(json_file)

            img_path_to_frame = {frame["file_path"]: frame for frame in transforms_json["frames"]}

        progress = get_progress(description="[bold yellow]Calculating monocular depth...")
        with progress:
            for image_path in progress.track(paths_to_images):
                img = util.io.read_image(image_path.as_posix())

                img_input = transform({"image": img})["image"]

                sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)

                if self.optimize is True and self.device.startswith("cuda"):
                    sample = sample.to(memory_format=torch.channels_last)
                    sample = sample.half()

                prediction = model.forward(sample)
                prediction = (
                    torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=img.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    )
                    .squeeze()
                    .cpu()
                    .numpy()
                )

                prediction = prediction - prediction.min()
                prediction = prediction / prediction.max()

                prediction = (prediction * ((1 << 16) - 1)).round().astype(np.uint16)

                # Replace original frame extension with png to save 16-bit depth images
                depth_path = monodepth_dir / f"{image_path.stem}.png"
                cv2.imwrite(depth_path.as_posix(), prediction)

                relative_img_path = image_path.relative_to(self.output_dir).as_posix()
                relative_depth_path = depth_path.relative_to(self.output_dir).as_posix()

                if img_path_to_frame and relative_img_path in img_path_to_frame:
                    img_path_to_frame[relative_img_path]["monodepth_file_path"] = relative_depth_path

        downscale_status = downscale_images(
            monodepth_dir, self.num_downscales, folder_name="monodepths", verbose=self.verbose
        )
        if self.verbose:
            CONSOLE.log(downscale_status)

        if transforms_json:
            with open(transforms_path, "w", encoding="utf-8") as json_file:
                json.dump(transforms_json, json_file, indent=4)

        CONSOLE.log("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")
