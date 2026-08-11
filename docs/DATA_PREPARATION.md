# Data Preparation

LumiTokens uses a two-stage data pipeline. Blender first renders posed images
under multiple lighting conditions, then the preprocessor converts those raw
renders into the metadata, image, environment-map, and light-ray layout read by
the model.

```text
3D assets -> object- or scene-level renderer -> raw renders -> preprocessor -> LumiTokens dataset
```

The release contains the migrated implementations under
`tools/data_generation/` and portable launchers under
`scripts/data_generation/`. The launchers do not require Slurm and contain no
machine-specific dataset paths.

## Requirements

Create the main LumiTokens environment first, then install the rendering
dependencies:

```bash
python -m pip install -r requirements-data-generation.txt
```

The development renderer used Linux, Python 3.11, and Blender Python
(`bpy`) 4.4.0. Set `LUMITOKENS_RENDER_PYTHON` if Blender dependencies are in a
different Python environment:

```bash
export LUMITOKENS_RENDER_PYTHON=/path/to/blender-python
```

The preprocessing stage uses PyTorch, Pillow, OpenCV, and `pyexr`. EXR input
requires a working `pyexr` installation.

## Obtain the source assets

Acquire the following assets from their official distributions and observe
their respective licenses:

- Objaverse GLB files, stored by shard as
  `GLBS_ROOT/<shard>/<uid>.glb`.
- Poly Haven HDRIs. The renderers expect `<name>_8k.exr` and
  `white_env_8k.exr` in one directory.
- For scene-level generation, Poly Haven low-quality models and textures.

The repository includes only the JSON name lists used to sample Poly Haven
assets; it does not redistribute the assets themselves.

Create a headerless CSV describing the Objaverse objects to render. Each row
contains the GLB shard and object UID:

```csv
000-000,000074a334c541878360457c672b6c2e
000-001,0001a1520c26444c9201e1f053e91772
```

`--group_start` is inclusive and `--group_end` is exclusive. Both are row
indices into this CSV.

## Generate object-centric data

`render_3dmodels_dense.sh` launches the enhanced object renderer
`render_3dmodels_dense_enhance.py`:

```bash
bash scripts/data_generation/render_3dmodels_dense.sh \
  --models_root /path/to/objaverse/glbs \
  --csv_path /path/to/object_ids.csv \
  --env_map_dir_path /path/to/hdris \
  --white_env_map_dir_path /path/to/hdris \
  --output_dir_override /path/to/raw_object_renders \
  --group_start 0 \
  --group_end 100
```

The default object recipe renders 200 training views and 100 test views at
512 x 512, with one white environment and five sampled environment lights.
Renderer options can change the view count, lighting mix, resolution, Cycles
samples, camera profile, and random seed; run the launcher with `--help` for
the complete list.

## Generate scene-level data

`render_3dscenes_dense.py` composes the selected Objaverse GLB with additional
Objaverse and Poly Haven objects, adds a ground plane, and renders the same raw
lighting layout. The default resolution is 512 x 512:

```bash
bash scripts/data_generation/render_3dscenes_dense.sh \
  --glbs_root_path /path/to/objaverse/glbs \
  --glb_list_path /path/to/object_ids.csv \
  --model_lq_dir /path/to/polyhaven_models \
  --texture_dir /path/to/polyhaven_textures \
  --env_map_dir_path /path/to/hdris \
  --white_env_map_dir_path /path/to/hdris \
  --output_dir /path/to/raw_scene_renders \
  --group_start 0 \
  --group_end 100 \
  --scene_seed 0
```

The bundled low-quality model list defaults to the training split. Pass
`--lq_list_path tools/data_generation/assets/object_ids/polyhaven_models_test.json`
when preparing held-out data.

## Raw renderer layout

Both renderers produce one object directory containing train and test camera
sets. Lighting folders contain `gt_<view>.png` files and the JSON describing
that light. The exact folders present depend on the enabled lighting options.

```text
raw_renders/<uid>/
├── normalize.json
├── train/
│   ├── cameras.json
│   ├── albedo/albedo_cam_<view>.png
│   ├── white_env_0/{gt_<view>.png,white_env.json}
│   ├── env_0/{gt_<view>.png,env.json}
│   ├── white_pl_0/{gt_<view>.png,white_pl.json}
│   ├── rgb_pl_0/{gt_<view>.png,rgb_pl.json}
│   ├── multi_pl_0/{gt_<view>.png,multi_pl.json}
│   ├── area_0/{gt_<view>.png,area.json}
│   └── combined_0/{gt_<view>.png,combined.json}
└── test/
    └── ...
```

Object renders additionally contain a deterministic scene manifest,
validation report, and `done.txt` or `invalid.txt` marker.

## Convert renders to the LumiTokens format

The release launchers correspond to the development scripts
`preprocess_train_full.sh` and `preprocess_test_full_pointLights.sh`. They
preserve the raw renders and write an uncompressed dataset by default.

```bash
bash scripts/data_generation/preprocess_train_full.sh \
  --input /path/to/raw_renders \
  --output /path/to/processed_data \
  --hdri-dir /path/to/hdris

bash scripts/data_generation/preprocess_test_full_pointLights.sh \
  --input /path/to/raw_renders \
  --output /path/to/processed_data \
  --hdri-dir /path/to/hdris
```

Use `--max-objects 5` for a small conversion smoke test. To create a second,
tar-backed copy, invoke `tools/data_generation/preprocess_objaverse.py`
directly with `--output-tar`; the two release launchers intentionally pass
`--no-output-tar`.

The resulting split is directly usable as `--data-root`:

```text
processed_data/test/
├── full_list.txt
├── metadata/<scene_name>.json
├── images/<scene_name>/<view>.png
├── envmaps/<scene_name>/<view>_{hdr,ldr}.png
├── point_light_rays/<scene_name>.npy
└── albedos/<object_uid>/<view>.png
```

Metadata stores OpenCV-convention world-to-camera matrices and pixel-space
intrinsics `[fx, fy, cx, cy]`. Environment-lit scenes receive a camera-aligned
HDR and LDR lighting image for each view. Local-light scenes receive a NumPy
array with rows `[intensity, r, g, b, ray_o_x, ray_o_y, ray_o_z, ray_d_x,
ray_d_y, ray_d_z]`.

The generated metadata and `full_list.txt` currently contain absolute paths.
Prepare the dataset at its final location, or set `training.og_dataset_base`
and `training.local_dataset_base` in the config when relocating it.

## Provenance

The renderers and `bpy_helper` utilities were migrated from the project data
renderer at revision `59795e5`. The preprocessor was migrated from the
LumiTokens development code at revision `1d6d9ee`. Release adaptations replace
private defaults with CLI paths, remove Slurm assumptions, and make
preprocessing non-destructive.
