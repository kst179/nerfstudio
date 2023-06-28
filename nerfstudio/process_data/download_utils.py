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

from pathlib import Path

import requests

from nerfstudio.utils.rich_utils import CONSOLE, get_progress


def download_file(url: str, path_to_save: Path, description="") -> bool:
    """Downloads file from given direct link with progress

    Args:
        url (str): URL to file, should be a direct link
        path_to_save (Path): path to save file on disk
        description (str, optional): message shown near the progressbar. Defaults to "".

    Raises:
        requests.RequestException: _description_

    Returns:
        bool: _description_
    """
    path_to_save.parent.mkdir(exist_ok=True, parents=True)

    try:
        progress = get_progress(description=description)

        with progress:
            response = requests.get(url, stream=True)

            if not response.ok:
                raise requests.RequestException(
                    f"Request returned code {response.status_code} with reason {response.reason}"
                )

            with open(path_to_save, "wb") as file:
                total_length = response.headers.get("content-length")

                if total_length is None:
                    file.write(response.content)
                else:
                    total_length = int(total_length)
                    task = progress.add_task("", total=total_length)
                    for data in response.iter_content(chunk_size=4096):
                        file.write(data)
                        progress.update(task, advance=len(data))

    except requests.RequestException as exc:
        CONSOLE.log(f"[bold red]Error while downloading weights: {exc.args[0]}")
        return False

    return True
