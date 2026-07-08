import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
from rknn.api import RKNN


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(root_dir):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    images = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if not images:
        raise RuntimeError(f"no images found under: {root}")
    return sorted(images)


def build_quant_dataset(dataset_root, output_dir, height, width, num_samples, seed):
    rng = random.Random(seed)
    images = collect_images(dataset_root)
    picked = images if len(images) <= num_samples else rng.sample(images, num_samples)

    quant_dir = Path(output_dir) / "quant_rgb_npy"
    quant_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(output_dir) / "quant_dataset_rgb_npy.txt"
    source_manifest = Path(output_dir) / "quant_dataset_rgb_npy_sources.txt"

    manifest_lines = []
    source_lines = []
    for idx, image_path in enumerate(sorted(picked)):
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        image_bgr = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        npy_path = quant_dir / f"quant_{idx:04d}.npy"
        image_rgb_nchw = np.expand_dims(np.transpose(image_rgb, (2, 0, 1)), axis=0)
        np.save(npy_path, image_rgb_nchw.astype(np.uint8))
        manifest_lines.append(str(npy_path.resolve()))
        source_lines.append(str(image_path.resolve()))

    if not manifest_lines:
        raise RuntimeError("failed to build quantization dataset: no valid images were processed")

    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    source_manifest.write_text("\n".join(source_lines) + "\n", encoding="utf-8")
    return str(manifest), str(source_manifest), len(manifest_lines)


def build_one(onnx_path, dataset_path, target_platform, output_path, do_quantization):
    rknn = RKNN(verbose=False)
    result = {
        "target_platform": target_platform,
        "onnx_path": onnx_path,
        "output_path": output_path,
        "do_quantization": do_quantization,
        "load_ret": None,
        "build_ret": None,
        "export_ret": None,
    }
    try:
        rknn.config(
            target_platform=target_platform,
            mean_values=[[123.675, 116.28, 103.53]],
            std_values=[[58.395, 57.12, 57.375]],
            quant_img_RGB2BGR=False,
        )
        result["load_ret"] = rknn.load_onnx(model=onnx_path)
        if result["load_ret"] != 0:
            return result

        result["build_ret"] = rknn.build(
            do_quantization=do_quantization,
            dataset=dataset_path if do_quantization else None,
        )
        if result["build_ret"] != 0:
            return result

        result["export_ret"] = rknn.export_rknn(output_path)
        return result
    except Exception as exc:
        result["exception"] = repr(exc)
        return result
    finally:
        rknn.release()


def main():
    parser = argparse.ArgumentParser(description="Build full SeaFormer RKNN with rknn-toolkit2.")
    parser.add_argument("--onnx", default="deploy/rknn/output/hybrid/seaformer_logits.onnx")
    parser.add_argument("--dataset", default=None)
    parser.add_argument(
        "--dataset-root",
        default="/home/shiro/AFFormer/data/cityscapes/leftImg8bit/train",
    )
    parser.add_argument("--output-dir", default="deploy/rknn/output/hybrid")
    parser.add_argument("--target-platform", default="rv1126b")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fp_path = os.path.join(args.output_dir, "seaformer_logits.rv1126b.fp.rknn")
    int8_path = os.path.join(args.output_dir, "seaformer_logits.rv1126b.int8.rknn")

    if args.dataset:
        dataset_path = args.dataset
        source_manifest = None
        quant_count = None
    else:
        dataset_path, source_manifest, quant_count = build_quant_dataset(
            args.dataset_root,
            args.output_dir,
            args.height,
            args.width,
            args.num_samples,
            args.seed,
        )

    fp_result = build_one(args.onnx, dataset_path, args.target_platform, fp_path, False)
    int8_result = build_one(args.onnx, dataset_path, args.target_platform, int8_path, True)

    summary = {
        "target_platform": args.target_platform,
        "onnx": args.onnx,
        "dataset": dataset_path,
        "dataset_root": args.dataset_root if not args.dataset else None,
        "dataset_source_manifest": source_manifest,
        "dataset_count": quant_count,
        "runtime_input_color": "RGB",
        "runtime_input_dtype": "uint8",
        "fp": fp_result,
        "int8": int8_result,
    }

    out_path = os.path.join(args.output_dir, "build_full_toolkit2_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
