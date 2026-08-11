# Third-Party Notices

LumiTokens builds on open-source research software. The repository-level
license does not replace copyright, license, or attribution notices attached
to individual files.

## LVSM

- Project: **LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias**
- Source: <https://github.com/haian-jin/LVSM>
- License: CC BY-NC-SA 4.0
- Copyright: Copyright (c) 2025 Haian Jin

The LumiTokens dataset interface, scene encoder/decoder components,
transformer components, loss implementation, and supporting utilities adapt
LVSM code. Modified files retain the upstream copyright header and identify
the LumiTokens modifications.

## Nerfstudio

- Project: **Nerfstudio**
- Source: <https://github.com/nerfstudio-project/nerfstudio>
- License: Apache License 2.0
- Copyright: Copyright 2022 the Regents of the University of California,
  Nerfstudio Team and contributors

`lumitokens/utils/camera.py` contains adapted camera transformation utilities
and retains the upstream Apache 2.0 notice.

## Crowdsampling the Plenoptic Function

- Project: **Crowdsampling the Plenoptic Function**
- Source: <https://github.com/zhengqili/Crowdsampling-the-Plenoptic-Function>
- License: MIT
- Referenced revision: `f5216f312cf82d77f8d20454b5eeb3930324630a`

Parts of the perceptual-loss implementation in `lumitokens/models/loss.py`
were adapted from this project. The source reference is retained in that file.

## Long-LRM

- Project: **Long-LRM** (self-reimplemented version)
- Source: <https://github.com/arthurhero/Long-LRM>
- License: Apache License 2.0

Parts of `lumitokens/models/loss.py` are based on Long-LRM and retain the source
reference in the file.

## PyTorch Benchmark / TorchBench

- Project: **PyTorch Benchmark**
- Source: <https://github.com/pytorch/benchmark>
- License: BSD 3-Clause
- Copyright: Copyright (c) 2019, PyTorch contributors

The `RMSNorm` implementation in `lumitokens/models/transformer.py` cites the
corresponding TorchBench implementation.

## Data-Generation Assets

The repository includes project-authored rendering and preprocessing utilities
but does not redistribute Objaverse or Poly Haven models, textures, or HDRIs.
The bundled JSON files contain only asset identifiers used by the renderer.
Users are responsible for obtaining the source assets and complying with their
respective terms.

## Model Weights and Datasets

Model weights and datasets are distributed separately from this source
repository and may be governed by additional terms stated by their respective
providers.
