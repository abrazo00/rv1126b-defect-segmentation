import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from rknnlite.api import RKNNLite
except ImportError:
    RKNNLite = None

try:
    from rknn.api import RKNN
except ImportError:
    RKNN = None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run original AFFormer full RKNN model on device.')
    parser.add_argument('--model', required=True, help='Path to .rknn model.')
    parser.add_argument('--metadata', default=None, help='Optional metadata json.')
    parser.add_argument('--image', required=True, help='Input image path.')
    parser.add_argument(
        '--reference-dir',
        default=None,
        help='Optional reference dir. Numeric diff is only valid for FP models.')
    parser.add_argument('--target', default='rv1126b', help='RKNN target for full runtime.')
    parser.add_argument('--device-id', default=None, help='Optional device id.')
    parser.add_argument(
        '--output-dir',
        default='deploy/rknn/runtime_output_original',
        help='Directory to save outputs.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose RKNN log.')
    return parser.parse_args()


def default_metadata():
    return {
        'input_shape_nchw': [1, 3, 512, 512],
        'num_classes': 2,
        'palette': [[0, 0, 0], [0, 0, 255]],
    }


def load_metadata(path: Optional[str]) -> dict:
    metadata = default_metadata()
    if path:
        metadata.update(json.loads(Path(path).read_text(encoding='utf-8')))
    return metadata


def load_runtime(model_path: Path, target: str, device_id: Optional[str], verbose: bool):
    if RKNNLite is not None:
        runtime = RKNNLite(verbose=verbose)
        ret = runtime.load_rknn(str(model_path))
        if ret != 0:
            raise RuntimeError(f'RKNNLite.load_rknn failed with code {ret}')
        ret = runtime.init_runtime()
        if ret != 0:
            raise RuntimeError(f'RKNNLite.init_runtime failed with code {ret}')
        return runtime, 'lite2'

    if RKNN is not None:
        runtime = RKNN(verbose=verbose)
        ret = runtime.load_rknn(str(model_path))
        if ret != 0:
            raise RuntimeError(f'RKNN.load_rknn failed with code {ret}')
        ret = runtime.init_runtime(target=target, device_id=device_id)
        if ret != 0:
            raise RuntimeError(f'RKNN.init_runtime failed with code {ret}')
        return runtime, 'full'

    raise ImportError('Neither rknnlite nor rknn python package is available.')


def prepare_runtime_input(image_path: Path, metadata: dict) -> Tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Failed to read image: {image_path}')
    _, _, height, width = metadata['input_shape_nchw']
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return image, resized[None, ...]


def canonicalize_logits(output: np.ndarray, num_classes: int) -> np.ndarray:
    array = np.asarray(output)
    if array.ndim == 4 and array.shape[1] == num_classes:
        return array
    if array.ndim == 4 and array.shape[-1] == num_classes:
        return np.transpose(array, (0, 3, 1, 2))
    if array.ndim == 3 and array.shape[0] == num_classes:
        return array[None, ...]
    if array.ndim == 3 and array.shape[-1] == num_classes:
        return np.transpose(array, (2, 0, 1))[None, ...]
    raise ValueError(f'Unexpected RKNN output shape: {array.shape}')


def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    return palette[mask].astype(np.uint8)


def save_outputs(output_dir: Path, original_bgr: np.ndarray, mask: np.ndarray, metadata: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    palette = np.asarray(metadata['palette'], dtype=np.uint8)
    color_mask = colorize_mask(mask, palette)
    color_mask = cv2.resize(
        color_mask, (original_bgr.shape[1], original_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_resized = cv2.resize(
        mask, (original_bgr.shape[1], original_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = cv2.addWeighted(original_bgr, 0.6, color_mask, 0.4, 0.0)
    cv2.imwrite(str(output_dir / 'mask.png'), mask_resized)
    cv2.imwrite(str(output_dir / 'mask_color.png'), color_mask)
    cv2.imwrite(str(output_dir / 'overlay.png'), overlay)


def compare_with_reference(reference_dir: Path, logits: np.ndarray):
    reference_mask = np.load(reference_dir / 'reference_mask.npy')
    pred_mask = logits.argmax(axis=1).astype(reference_mask.dtype)
    print(f'mask_match={float(np.mean(pred_mask == reference_mask)):.6f}')
    if np.issubdtype(logits.dtype, np.floating):
        reference_logits = np.load(reference_dir / 'reference_logits.npy')
        print(f'max_abs_diff={float(np.max(np.abs(logits - reference_logits))):.8f}')
        print(f'mean_abs_diff={float(np.mean(np.abs(logits - reference_logits))):.8f}')
    else:
        print('reference_logit_diff=skipped_non_float_output')


def main():
    args = parse_args()
    model_path = Path(args.model).resolve()
    metadata = load_metadata(args.metadata)
    output_dir = Path(args.output_dir).resolve()
    original_bgr, runtime_input = prepare_runtime_input(Path(args.image).resolve(), metadata)
    runtime, runtime_type = load_runtime(model_path, args.target, args.device_id, args.verbose)

    outputs = runtime.inference(inputs=[runtime_input])
    if not outputs:
        raise RuntimeError('RKNN inference returned empty outputs.')

    logits = canonicalize_logits(outputs[0], metadata['num_classes'])
    mask = logits.argmax(axis=1)[0].astype(np.uint8)
    save_outputs(output_dir, original_bgr, mask, metadata)

    print(f'runtime={runtime_type}')
    print(f'output_dtype={logits.dtype}')
    print(f'output_shape={logits.shape}')
    print(f'saved_dir={output_dir}')

    if args.reference_dir:
        compare_with_reference(Path(args.reference_dir).resolve(), logits)


if __name__ == '__main__':
    main()
