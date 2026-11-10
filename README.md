# halodiffuse

[![Radeon Graphics](https://img.shields.io/badge/Ryzen-AI_Max+-F26522.svg?logo=amd&logoSize=auto)](https://www.amd.com/)
[![Radeon Graphics](https://img.shields.io/badge/Radeon-Graphics-ED1C24.svg?logo=amd&logoSize=auto)](https://www.amd.com/)
[![Rust: Nightly](https://img.shields.io/badge/Rust-Nightly-orange.svg?logo=rust&logoSize=auto)](https://www.rust-lang.org/)
[![Vulkan: 1.3](https://img.shields.io/badge/Vulkan-1.3-AC162C.svg?logo=vulkan&logoSize=auto)](https://www.vulkan.org/)
[![License: AGPL v3.0](https://img.shields.io/badge/License-AGPL_v3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Model: SD 3.5 Large Turbo](https://img.shields.io/badge/Model-SD_3.5_Large_Turbo-FFD21E.svg?logo=huggingface)](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo)

## Stable Diffusion 3.5 Large Turbo for Strix Halo

### Usage

```bash
# Save to a specific file path
halodiffuse "orange cat" ~/tmp/cat.png
# Or pass a directory (automatically saves to ~/Pictures/output.png)
halodiffuse "orange cat" ~/Pictures
```

## Installation

**Required: AMD Ryzen AI Max+ 395 64GB+ Linux and agreeing to stability ai**

```bash
chmod +x halodiffuse
sudo mv halodiffuse /usr/local/bin/
pip install torch safetensors huggingface_hub
hf auth login
python install.py # Optional parameters see help
```

## Benchmarks

Stable Diffusion 3.5 Large Turbo FP16 1024x1024 Batch Size 1

| Engine                    | Backend      | Precision   | End-to-End     | Runtime                   |
| :------------------------ | :----------- | :---------- | :------------- | :------------------------ |
| **`halodiffuse`**         | **`Vulkan`** | **`FP16`**  | **`5.3 s`**    | **`~15.5 MB binary`**     |
| **ComfyUI**               | ROCm         | BF16 / FP16 | ~11.8 – 14.5 s | ~18 GB Python + ROCm venv |
| **stable-diffusion.cpp**  | Vulkan       | FP16        | ~13.5 – 16.2 s | ~50 MB binary             |
| **HuggingFace Diffusers** | ROCm         | BF16 / FP16 | ~18.4 s+       | ~18 GB Python + ROCm venv |

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
along with this program. If not, see <http://www.gnu.org/licenses/>

## Model License

The Stable Diffusion 3.5 Large Turbo model weights are governed by the [Stability AI Community License Agreement](https://www.google.com/url?sa=E&q=https%3A%2F%2Fhuggingface.co%2Fstabilityai%2Fstable-diffusion-3.5-large-turbo%2Fblob%2Fmain%2FLICENSE.md) `Powered by Stability AI`
