import argparse
import json
import os

import cv2
import numpy as np
import onnxruntime as ort

try:
    from rknnlite.api import RKNNLite
except ImportError as exc:
    raise SystemExit(f"rknnlite is required on the board side: {exc}")


def load_metadata(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_bgr(image_path, input_size):
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")
    resized = cv2.resize(image_bgr, (input_size[1], input_size[0]), interpolation=cv2.INTER_LINEAR)
    return image_bgr, resized


def ensure_nchw(output_array, expected_channels):
    if output_array.ndim != 4:
        raise ValueError(f"unexpected RKNN output rank: {output_array.shape}")
    if output_array.shape[1] == expected_channels:
        return output_array
    if output_array.shape[-1] == expected_channels:
        return np.transpose(output_array, (0, 3, 1, 2))
    raise ValueError(f"cannot align output shape {output_array.shape} to channels {expected_channels}")


def overlay_mask(image_bgr, mask):
    color_mask = np.zeros_like(image_bgr)
    color_mask[mask == 1] = (0, 255, 0)
    return cv2.addWeighted(image_bgr, 0.6, color_mask, 0.4, 0.0)


def main():
    parser = argparse.ArgumentParser(description="Run SeaFormer hybrid inference on RV1126B.")
    parser.add_argument("--front-rknn", required=True)
    parser.add_argument("--tail-onnx", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    metadata = load_metadata(args.metadata)
    input_size = tuple(metadata["input_size"])
    detail_channels = metadata["detail_feat_shape"][1]
    mid_channels = metadata["mid_feat_shape"][1]

    original_bgr, resized_bgr = preprocess_bgr(args.image, input_size)
    front_input = np.expand_dims(resized_bgr, axis=0).astype(np.uint8)

    rknn = RKNNLite()
    ret = rknn.load_rknn(args.front_rknn)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")

    try:
        ret = rknn.init_runtime()
    except TypeError:
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")

    front_outputs = rknn.inference(inputs=[front_input])
    rknn.release()

    detail_feat = ensure_nchw(front_outputs[0], detail_channels).astype(np.float32)
    mid_feat = ensure_nchw(front_outputs[1], mid_channels).astype(np.float32)

    ort_session = ort.InferenceSession(args.tail_onnx)
    logits = ort_session.run(
        None,
        {
            "detail_feat": detail_feat,
            "mid_feat": mid_feat,
        },
    )[0]
    mask = np.argmax(logits, axis=1).astype(np.uint8)[0]
    mask_resized = cv2.resize(mask, (original_bgr.shape[1], original_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = overlay_mask(original_bgr, mask_resized)

    mask_path = os.path.join(args.output_dir, "mask.png")
    overlay_path = os.path.join(args.output_dir, "overlay.png")
    npy_path = os.path.join(args.output_dir, "logits.npy")

    cv2.imwrite(mask_path, mask_resized)
    cv2.imwrite(overlay_path, overlay)
    np.save(npy_path, logits)

    print(json.dumps(
        {
            "mask_path": mask_path,
            "overlay_path": overlay_path,
            "logits_path": npy_path,
            "mask_unique_values": np.unique(mask_resized).tolist(),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
