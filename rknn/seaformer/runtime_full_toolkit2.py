import argparse
import json
import os

import cv2
import numpy as np

try:
    from rknnlite.api import RKNNLite
except ImportError as exc:
    raise SystemExit(f"rknn-toolkit-lite2 is required on board: {exc}")


def overlay_mask(image_bgr, mask):
    color_mask = np.zeros_like(image_bgr)
    color_mask[mask == 1] = (0, 255, 0)
    return cv2.addWeighted(image_bgr, 0.6, color_mask, 0.4, 0.0)


def main():
    parser = argparse.ArgumentParser(description="Run full SeaFormer RKNN model with rknn-toolkit-lite2.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"failed to read image: {args.image}")
    resized_bgr = cv2.resize(image_bgr, (args.width, args.height), interpolation=cv2.INTER_LINEAR)

    # Training uses OpenCV BGR input plus to_rgb=True in Normalize, so the deployed model expects
    # RGB semantic order at its tensor input. We therefore convert board-side BGR to RGB here and
    # build INT8 with RGB .npy calibration samples to keep runtime and quantization consistent.
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    model_input = np.expand_dims(resized_rgb, axis=0)

    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(args.model)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")

    try:
        ret = rknn_lite.init_runtime()
    except TypeError:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")

    outputs = rknn_lite.inference(inputs=[model_input])
    rknn_lite.release()
    logits = outputs[0]

    if logits.ndim == 4 and logits.shape[1] == 2:
        logits_nchw = logits
    elif logits.ndim == 4 and logits.shape[-1] == 2:
        logits_nchw = np.transpose(logits, (0, 3, 1, 2))
    else:
        raise ValueError(f"unexpected logits shape: {logits.shape}")

    mask = np.argmax(logits_nchw, axis=1).astype(np.uint8)[0]
    mask_resized = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = overlay_mask(image_bgr, mask_resized)

    mask_path = os.path.join(args.output_dir, "mask.png")
    overlay_path = os.path.join(args.output_dir, "overlay.png")
    logits_path = os.path.join(args.output_dir, "logits.npy")
    info_path = os.path.join(args.output_dir, "runtime_info.json")

    cv2.imwrite(mask_path, mask_resized)
    cv2.imwrite(overlay_path, overlay)
    np.save(logits_path, logits_nchw)

    info = {
        "mask_path": mask_path,
        "overlay_path": overlay_path,
        "logits_path": logits_path,
        "runtime_input_color": "RGB",
        "runtime_input_dtype": "uint8",
        "mask_unique_values": np.unique(mask_resized).tolist(),
        "logits_shape": list(logits_nchw.shape),
        "logits_dtype": str(logits_nchw.dtype),
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
