# halodiffuse

---

[![Radeon Graphics](https://img.shields.io/badge/Radeon-Graphics-ED1C24.svg?logo=amd&logoSize=auto)](https://www.amd.com/)
[![Rust: Nightly](https://img.shields.io/badge/Rust-Nightly-orange.svg?logo=rust&logoSize=auto)](https://www.rust-lang.org/)
[![Vulkan: 1.3](https://img.shields.io/badge/Vulkan-1.3-AC162C.svg?logo=vulkan&logoSize=auto)](https://www.vulkan.org/)
[![License: AGPL v3.0](https://img.shields.io/badge/License-AGPL_v3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Bare-metal Rust + Vulkan 1.3 FP16 text-to-image inference engine built around hand-written GLSL 460 cooperative-matrix compute shaders.

---

### [stabilityai/stable-diffusion-3.5-large-turbo](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo)

![sd35 demo](assets/sd35.png)

### [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)

![zimage demo](assets/zimage.png)

---

## Benchmarks

| Hardware                                 | Stack                        | SD 3.5 Large Turbo (4 steps) | Z-Image-Turbo (8 steps) | Download |
| :--------------------------------------- | :--------------------------- | :--------------------------: | :---------------------: | :------- |
| **AMD Ryzen AI Max+ 395 (Radeon 8060S)** | **halodiffuse (Vulkan 1.3)** |            `4.6s`            |         `18.4s`         |  `34MB`  |
| **AMD Ryzen AI Max+ 395 (Radeon 8060S)** | **ComfyUI ROCm nightlies 2** |            `5.6s`            |         `24.2s`         |  `12GB`   |

---

## Installation

**Requires AI MAX+ 395 and Linux**

**Note:** `stabilityai/stable-diffusion-3.5-large-turbo` is a gated repository. Make sure you have accepted its license on [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo) and provide an [HF_TOKEN](https://huggingface.co/settings/tokens/new?preset=read-only).

```shell
curl -LO https://github.com/Legendofzaday/halodiffuse/releases/latest/download/halodiffuse
chmod +x halodiffuse
sudo mv halodiffuse /usr/local/bin/
HF_TOKEN=<token> halodiffuse install sd35
halodiffuse install zimage
# --raw-dir <DIR> to use local weights instead of downloading
# --force to overwrite or resume corrupted packs
```

---

## Usage

`halodiffuse` runs a lightweight background daemon per model over a Unix domain socket so weights stay resident in GPU memory between prompts.

```shell
halodiffuse --help
# halodiffuse 1.2.0
#
#
# Usage:
#  halodiffuse install <sd35|zimage> [--force] [--raw-dir <DIR>]
#  halodiffuse start <sd35|zimage>
#  halodiffuse stop <sd35|zimage>
#  halodiffuse status [sd35|zimage]
#  halodiffuse <sd35|zimage> --prompt <TEXT> --path <FILE> [--seed <U64>]
#
# Options:
#  -p, --prompt <TEXT>  Text prompt
#  -o, --path <FILE>    Output PNG path
#  -s, --seed <U64>     RNG seed
#  -h, --help           Show help
#  -V, --version        Show version

halodiffuse start sd35
halodiffuse start zimage
halodiffuse sd35 --prompt "A cozy, quiet Japanese village nestled in green rolling hills at sunset, warm golden hour light casting long shadows, wooden traditional houses with glowing amber windows, lush cherry blossom trees in full bloom, soft mist in the valley, Studio Ghibli anime style, peaceful and nostalgic mood, wide-shot, 4k resolution." --path assets/sd35.png --seed 0
halodiffuse zimage --prompt "A cozy, quiet Japanese village nestled in green rolling hills at sunset, warm golden hour light casting long shadows, wooden traditional houses with glowing amber windows, lush cherry blossom trees in full bloom, soft mist in the valley, Studio Ghibli anime style, peaceful and nostalgic mood, wide-shot, 4k resolution." --path assets/zimage.png --seed 0
halodiffuse status
halodiffuse stop sd35
halodiffuse stop zimage
```

---

## Code License

halodiffuse  
Copyright (C) 2026 Thopuri Omkar Eeswar

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see https://www.gnu.org/licenses/.

## Model Licenses

The Stable Diffusion 3.5 Large Turbo model weights are governed by the [Stability AI Community License Agreement](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo/blob/main/LICENSE.md). `Powered by Stability AI`

The Z-Image-Turbo model weights are governed by the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0). `Powered by Tongyi-MAI`
