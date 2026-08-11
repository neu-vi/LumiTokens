"""Single-GPU inference entry point for the LumiTokens release."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relight a scene from the packaged PolyHaven/LVSM example dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/relight_256_mlp.yaml",
        help="Inference YAML (default: configs/relight_256_mlp.yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint path. Defaults to the checkpoint declared by the config.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "examples/polyhaven_lvsm_test",
        help="Extracted example dataset root.",
    )
    parser.add_argument("--scene-index", type=int, default=0, help="Filtered dataset index.")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/quickstart")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--all-frames", action="store_true", help="Render every source camera pose.")
    parser.add_argument(
        "--video",
        action="store_true",
        help="Encode prediction and available target-lighting frames as MP4.",
    )
    parser.add_argument(
        "--num-input-views",
        type=int,
        default=None,
        help="Override the number of context views declared by the config.",
    )
    parser.add_argument("--view-chunk-size", type=int, default=1)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config, checkpoint architecture, and one dataset sample without CUDA inference.",
    )
    return parser.parse_args()


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_config(args: argparse.Namespace):
    config_path = resolve_repo_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    config = OmegaConf.load(config_path)

    data_root = args.data_root.expanduser().resolve()
    manifest = data_root / "full_list.txt"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Dataset manifest does not exist: {manifest}. Extract the example archive first."
        )
    first_entry = next((line.strip() for line in manifest.read_text().splitlines() if line.strip()), None)
    if first_entry is None:
        raise ValueError(f"Dataset manifest is empty: {manifest}")
    # An archived manifest may retain its original dataset prefix. Infer that
    # prefix from `<dataset-root>/metadata/<scene>.json` and remap it to --data-root.
    config.training.og_dataset_base = str(Path(first_entry).parent.parent)
    config.training.dataset_path = str(manifest)
    config.training.local_dataset_base = str(data_root)
    config.inference.if_inference = True
    config.inference.random_chunk_seed = args.seed
    if args.num_input_views is not None:
        if args.num_input_views < 1:
            raise ValueError("--num-input-views must be at least 1.")
        config.training.num_input_views = args.num_input_views
    view_idx_path = config.inference.get("view_idx_file_path", None)
    if view_idx_path:
        config.inference.view_idx_file_path = str(resolve_repo_path(view_idx_path))

    if args.video:
        args.all_frames = True
    config.training.load_all_frames = args.all_frames
    if args.all_frames:
        config.inference.random_chunk_sampling = False
    config.inference.render_all_views = False

    checkpoint_value = args.checkpoint or config.get("checkpoint")
    if checkpoint_value is None:
        raise ValueError("No checkpoint was supplied and the config has no 'checkpoint' field.")
    checkpoint_path = resolve_repo_path(checkpoint_value)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    return config, config_path, checkpoint_path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_checkpoint(checkpoint_path: Path, config) -> dict:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Expected a training checkpoint containing a 'model' state dictionary.")
    state = checkpoint["model"]
    expected_head = str(config.model.decoder_head.get("type", "mlp")).lower()
    detected_head = (
        "dpt" if any(key.startswith("image_token_decoder.layer_projs.") for key in state) else "mlp"
    )
    if detected_head != expected_head:
        raise ValueError(
            f"Checkpoint/config mismatch: checkpoint uses {detected_head.upper()}, "
            f"config requests {expected_head.upper()}."
        )

    # Constructing on the meta device validates every runtime parameter name and
    # shape without allocating the roughly 4 GB model on the CPU or GPU.
    from lumitokens.models.editor import LatentSceneEditor

    with torch.device("meta"):
        expected_state = LatentSceneEditor(config).state_dict()
    missing = sorted(key for key in expected_state if key not in state)
    unexpected = sorted(
        key for key in state if key not in expected_state and not key.startswith("loss_computer.")
    )
    shape_mismatches = sorted(
        key
        for key in expected_state.keys() & state.keys()
        if tuple(expected_state[key].shape) != tuple(state[key].shape)
    )
    if missing or unexpected or shape_mismatches:
        raise ValueError(
            "Checkpoint state does not match the release model: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}, "
            f"shape_mismatches={shape_mismatches[:10]}"
        )
    return {
        "decoder_head": detected_head,
        "parameter_tensors": len(state),
        "training_step": int(
            checkpoint.get("fwdbwd_pass_step", checkpoint.get("param_update_step", -1))
        ),
    }


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "scene"


def first_text(value) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0])
    return str(value)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = (
        tensor.detach()
        .float()
        .cpu()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .mul(255)
        .round()
        .byte()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def save_views(images: torch.Tensor | None, indices: torch.Tensor, directory: Path) -> list[Path]:
    if images is None:
        return []
    directory.mkdir(parents=True, exist_ok=True)
    frame_ids = indices[0, :, 0].detach().cpu().tolist()
    paths = []
    for image, frame_id in zip(images[0], frame_ids):
        path = directory / f"{int(frame_id):05d}.png"
        tensor_to_image(image).save(path)
        paths.append(path)
    return paths


def save_video(frame_paths: list[Path], output_path: Path, fps: int) -> Path | None:
    if not frame_paths:
        return None
    import imageio.v2 as imageio

    frames = [np.asarray(Image.open(path).convert("RGB")) for path in frame_paths]
    imageio.mimsave(output_path, frames, fps=fps, quality=8)
    return output_path


def save_result(result, output: Path, config_path: Path, checkpoint_path: Path, args) -> None:
    source_scene = first_text(result.input.scene_name)
    lighting_scene = first_text(getattr(result.input, "relit_scene_name", "lighting"))
    source_name = safe_name(source_scene)
    lighting_name = safe_name(lighting_scene)
    run_dir = output / f"{source_name}__to__{lighting_name}"

    context_paths = save_views(result.input.image, result.input.index, run_dir / "context")
    source_target_paths = save_views(
        getattr(result.target, "image", None), result.target.index, run_dir / "source_target"
    )
    target_images = getattr(result.target, "relit_images", None)
    target_paths = save_views(target_images, result.target.index, run_dir / "target")
    predicted_paths = save_views(result.render, result.target.index, run_dir / "predicted")

    if source_target_paths and target_paths and predicted_paths:
        rows = [result.target.image[0], target_images[0], result.render[0]]
        row_grids = [torch.cat(list(row), dim=2) for row in rows]
        tensor_to_image(torch.cat(row_grids, dim=1)).save(run_dir / "comparison.png")

    context_lighting_paths = []
    target_lighting_paths = []

    def prepare_lighting(lighting: torch.Tensor) -> torch.Tensor:
        if tuple(lighting.shape[-2:]) != (256, 512):
            import torch.nn.functional as functional

            batch_size, num_views = lighting.shape[:2]
            lighting = functional.interpolate(
                lighting.flatten(0, 1),
                size=(256, 512),
                mode="bilinear",
                align_corners=False,
            ).reshape(batch_size, num_views, 3, 256, 512)
        return lighting

    if hasattr(result.input, "env_ldr") and result.input.env_ldr is not None:
        context_lighting_paths = save_views(
            prepare_lighting(result.input.env_ldr),
            result.input.index,
            run_dir / "lighting" / "context",
        )

    if hasattr(result.target, "env_ldr") and result.target.env_ldr is not None:
        target_lighting_paths = save_views(
            prepare_lighting(result.target.env_ldr),
            result.target.index,
            run_dir / "lighting" / "target",
        )

    video_path = None
    lighting_video_path = None
    if args.video:
        video_path = save_video(predicted_paths, run_dir / "predicted.mp4", args.fps)
        lighting_video_path = save_video(
            target_lighting_paths,
            run_dir / "lighting.mp4",
            args.fps,
        )

    metadata = {
        "source_scene": source_scene,
        "lighting_scene": lighting_scene,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "seed": args.seed,
        "all_frames": args.all_frames,
        "num_input_views": len(context_paths),
        "context_frames": len(context_paths),
        "source_target_frames": len(source_target_paths),
        "target_frames": len(target_paths),
        "predicted_frames": len(predicted_paths),
        "context_lighting_frames": len(context_lighting_paths),
        "target_lighting_frames": len(target_lighting_paths),
        "context_view_indices": result.input.index[0, :, 0].detach().cpu().tolist(),
        "target_view_indices": result.target.index[0, :, 0].detach().cpu().tolist(),
        "video": str(video_path) if video_path else None,
        "lighting_video": str(lighting_video_path) if lighting_video_path else None,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved results to {run_dir}")


def main() -> None:
    args = parse_args()
    if args.scene_index < 0:
        raise ValueError("--scene-index must be non-negative.")
    if args.view_chunk_size < 1:
        raise ValueError("--view-chunk-size must be at least 1.")

    config, config_path, checkpoint_path = load_config(args)
    seed_everything(args.seed)

    from lumitokens.data import Dataset

    dataset = Dataset(config)
    if args.scene_index >= len(dataset):
        raise IndexError(f"--scene-index {args.scene_index} is outside dataset size {len(dataset)}.")
    sample = dataset[args.scene_index]
    batch = {
        key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
        for key, value in sample.items()
    }
    # Dataset sampling and model-side target selection use independent RNGs in
    # the original worker-based pipeline. Restore the main-process seed here.
    seed_everything(args.seed)

    expected_head = str(config.model.decoder_head.get("type", "mlp")).lower()
    checkpoint_info = validate_checkpoint(checkpoint_path, config)
    print(
        f"Validated scene '{first_text(batch['scene_name'])}', {expected_head.upper()} checkpoint "
        f"at step {checkpoint_info['training_step']}."
    )
    if args.validate_only:
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run inference inside a GPU allocation, or use "
            "--validate-only for the CPU-side release check."
        )

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.backends.cuda.matmul.allow_tf32 = bool(config.training.get("use_tf32", True))
    torch.backends.cudnn.allow_tf32 = bool(config.training.get("use_tf32", True))
    device = torch.device("cuda")

    from lumitokens.models.editor import LatentSceneEditor

    model = LatentSceneEditor(config).to(device)
    if model.load_ckpt(str(checkpoint_path)) is None:
        raise RuntimeError(f"Failed to load checkpoint: {checkpoint_path}")
    model.eval()
    batch = move_to_device(batch, device)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = amp_dtype[str(config.training.get("amp_dtype", "bf16"))]
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=dtype,
        enabled=bool(config.training.get("use_amp", True)),
    ):
        if args.all_frames:
            result = model.render_full_target_for_vis(batch, view_chunk_size=args.view_chunk_size)
        else:
            result = model(batch, has_target_image=True, compute_loss=False)

    save_result(result, args.output.expanduser().resolve(), config_path, checkpoint_path, args)


if __name__ == "__main__":
    main()
