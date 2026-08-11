# LumiTokens Checkpoints

Pretrained LumiTokens checkpoints are distributed separately from the source
code through the [LumiTokens Google Drive folder](https://drive.google.com/drive/folders/1hbCQBwag8FuhrM3abqpcvNT5s8EPJJL7?usp=sharing).

## Available Models

| Checkpoint path | Decoder head | Resolution | Config | Size | SHA-256 |
| --- | --- | ---: | --- | ---: | --- |
| `mlp/lumitokens_relight_256_mlp.pt` | MLP | 256 x 256 | `configs/relight_256_mlp.yaml` | 3.43 GiB | `1e95a5023cecf8c5147dfe72391f5740f5d72167064453dbde6d117683810cfe` |
| `mlp/lumitokens_relight_512_mlp.pt` | MLP | 512 x 512 | `configs/relight_512_mlp.yaml` | 3.43 GiB | `b866fdd66a6974ecfc258881d236daa54d9c07216cecbb3617025f816d34966e` |
| `dpt/lumitokens_relight_512_dpt.pt` | DPT | 512 x 512 | `configs/relight_512_dpt.yaml` | 3.61 GiB | `fca5a4216a9c14b71a83d17afaf4c7750555b6011d0019402b76aa04bd08d349` |

Download a checkpoint and place it at the path shown in the first column,
relative to this directory. MLP and DPT checkpoints are not interchangeable;
always use the matching configuration.

## Verify a Download

From the repository root, verify all downloaded checkpoints with:

```bash
sha256sum --check checkpoints/SHA256SUMS
```

If only one checkpoint is present, verify it directly and compare the result
with the table above:

```bash
sha256sum checkpoints/mlp/lumitokens_relight_256_mlp.pt
```
