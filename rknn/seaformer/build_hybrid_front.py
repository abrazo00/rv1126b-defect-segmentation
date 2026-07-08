import argparse
import json
import os
from pathlib import Path

from rknn.api import RKNN


def collect_dataset(image_dir, dataset_path, limit):
    image_dir = Path(image_dir)
    image_paths = sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
    )
    image_paths = image_paths[:limit]
    with open(dataset_path, "w", encoding="utf-8") as f:
        for path in image_paths:
            f.write(str(path.resolve()) + "\n")
    return [str(path.resolve()) for path in image_paths]


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
            reorder_channel="2 1 0",
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
    parser = argparse.ArgumentParser(description="Build SeaFormer hybrid front RKNN model.")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", default="deploy/rknn/output/hybrid")
    parser.add_argument("--target-platform", default="rv1126")
    parser.add_argument("--dataset-size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_path = os.path.join(args.output_dir, "quant_dataset.txt")
    dataset_files = collect_dataset(args.image_dir, dataset_path, args.dataset_size)

    fp_path = os.path.join(args.output_dir, "seaformer_front_smb1_smb4.rv1126.fp.rknn")
    int8_path = os.path.join(args.output_dir, "seaformer_front_smb1_smb4.rv1126.int8.rknn")

    fp_result = build_one(args.onnx, dataset_path, args.target_platform, fp_path, False)
    int8_result = build_one(args.onnx, dataset_path, args.target_platform, int8_path, True)

    summary = {
        "dataset_path": dataset_path,
        "dataset_count": len(dataset_files),
        "target_platform": args.target_platform,
        "fp": fp_result,
        "int8": int8_result,
    }

    summary_path = os.path.join(args.output_dir, "build_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
