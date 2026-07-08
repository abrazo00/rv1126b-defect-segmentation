from __future__ import annotations

import gc
import json
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import psutil
from rknnlite.api import RKNNLite


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEAFORMER_ROOT = PROJECT_ROOT.parent / "rknn" / "seaformer"
MODELS = {
    "fp": SEAFORMER_ROOT / "output" / "hybrid" / "seaformer_logits.rv1126b.fp.rknn",
    "int8": SEAFORMER_ROOT / "output" / "hybrid" / "seaformer_logits.rv1126b.int8.rknn",
}
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
TEMP_ROOT = RUNTIME_ROOT / "tmp"
SAVED_ROOT = RUNTIME_ROOT / "saved"
INPUT_SIZE = (512, 512)
NUM_CLASSES = 2
TEMP_RETENTION_HOURS = 24
QUALITY_WARN_THRESHOLD = 55.0


@dataclass(frozen=True)
class PreprocessConfig:
    mode: str
    label: str
    description: str = ""
    clahe: bool = False
    denoise: bool = False
    sharpen: bool = False
    contrast: float = 1.0
    brightness: float = 0.0
    saturation: float = 1.0
    gamma: float = 1.0
    highlight_suppress: bool = False
    defect_boost: bool = False


PREPROCESS_CONFIGS = {
    "standard": PreprocessConfig(mode="standard", label="标准模式", description="保持原始输入，仅执行尺寸和颜色空间适配。"),
    "low_light": PreprocessConfig(mode="low_light", label="低照度增强", description="提升暗部层次，适合弱光画面。", clahe=True, gamma=0.82, brightness=8.0, contrast=1.08),
    "overexposure_control": PreprocessConfig(mode="overexposure_control", label="逆光/过曝抑制", description="压制高光并恢复局部对比，适合反光和逆光场景。", clahe=True, gamma=1.18, highlight_suppress=True),
    "low_contrast": PreprocessConfig(mode="low_contrast", label="低对比增强", description="增强局部对比度和色彩层次，适合灰雾或低对比画面。", clahe=True, contrast=1.22, saturation=1.08),
    "denoise": PreprocessConfig(mode="denoise", label="噪声抑制", description="降低彩色噪声，适合高 ISO 或压缩噪声画面。", denoise=True),
    "motion_sharpen": PreprocessConfig(mode="motion_sharpen", label="运动模糊锐化", description="强化边缘细节，适合轻微拖影或虚焦画面。", sharpen=True, contrast=1.08),
    "edge_enhance": PreprocessConfig(mode="edge_enhance", label="边缘强化", description="增强边缘和纹理细节，适合突出工件轮廓。", clahe=True, sharpen=True),
    "defect_highlight": PreprocessConfig(mode="defect_highlight", label="工业缺陷突出", description="放大局部纹理差异，让细小异常更容易观察。", clahe=True, sharpen=True, defect_boost=True, contrast=1.16),
}


def _slugify(name: str) -> str:
    name = Path(name).name
    stem = Path(name).stem
    suffix = Path(name).suffix.lower() or ".png"
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem).strip("_") or "image"
    return f"{stem}{suffix}"


def _overlay_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color_mask = np.zeros_like(image_bgr)
    color_mask[mask == 1] = (0, 255, 0)
    return cv2.addWeighted(image_bgr, 0.6, color_mask, 0.4, 0.0)


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color_mask[mask == 1] = (0, 255, 0)
    return color_mask


def _resolve_preprocess_config(mode: str) -> PreprocessConfig:
    return PREPROCESS_CONFIGS.get(mode, PREPROCESS_CONFIGS["standard"])


def _apply_gamma(image_bgr: np.ndarray, gamma: float) -> np.ndarray:
    inverse = 1.0 / max(gamma, 1e-6)
    table = np.array([((idx / 255.0) ** inverse) * 255 for idx in range(256)], dtype=np.uint8)
    return cv2.LUT(image_bgr, table)


def _apply_clahe(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def _adjust_contrast_brightness(image_bgr: np.ndarray, contrast: float, brightness: float) -> np.ndarray:
    if abs(contrast - 1.0) < 0.01 and abs(brightness) < 0.01:
        return image_bgr
    return cv2.convertScaleAbs(image_bgr, alpha=contrast, beta=brightness)


def _adjust_saturation(image_bgr: np.ndarray, saturation: float) -> np.ndarray:
    if abs(saturation - 1.0) < 0.01:
        return image_bgr
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _suppress_highlights(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    compressed = np.where(l_channel > 210, 210 + (l_channel - 210) * 0.35, l_channel)
    return cv2.cvtColor(cv2.merge((compressed.astype(np.uint8), a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def _boost_local_defects(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    high_pass = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    high_pass = cv2.convertScaleAbs(high_pass)
    detail = cv2.cvtColor(high_pass, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(image_bgr, 0.82, detail, 0.18, 0)


def _apply_preprocess(image_bgr: np.ndarray, mode: str) -> tuple[np.ndarray, PreprocessConfig]:
    config = _resolve_preprocess_config(mode)
    output = image_bgr.copy()
    if config.clahe:
        output = _apply_clahe(output)
    if config.denoise:
        output = cv2.fastNlMeansDenoisingColored(output, None, 5, 5, 7, 21)
    if config.highlight_suppress:
        output = _suppress_highlights(output)
    if abs(config.gamma - 1.0) > 0.02:
        output = _apply_gamma(output, config.gamma)
    output = _adjust_contrast_brightness(output, config.contrast, config.brightness)
    output = _adjust_saturation(output, config.saturation)
    if config.sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        output = cv2.filter2D(output, -1, kernel)
    if config.defect_boost:
        output = _boost_local_defects(output)
    return output, config


def _measure_image_quality(image_bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    saturation = float(hsv[:, :, 1].mean())

    brightness_score = max(0.0, 100.0 - abs(brightness - 128.0) * 0.78)
    contrast_score = min((contrast / 64.0) * 100.0, 100.0)
    sharpness_score = min((sharpness / 450.0) * 100.0, 100.0)
    saturation_score = min((saturation / 120.0) * 100.0, 100.0)
    quality_score = brightness_score * 0.26 + contrast_score * 0.24 + sharpness_score * 0.4 + saturation_score * 0.1

    if quality_score >= 75.0:
        quality_level = "优秀"
    elif quality_score >= QUALITY_WARN_THRESHOLD:
        quality_level = "可用"
    else:
        quality_level = "风险"

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "saturation": saturation,
        "quality_score": quality_score,
        "quality_level": quality_level,
    }


def _quality_delta(before: dict, after: dict) -> dict:
    return {
        "brightness_delta": after["brightness"] - before["brightness"],
        "contrast_delta": after["contrast"] - before["contrast"],
        "sharpness_delta": after["sharpness"] - before["sharpness"],
        "saturation_delta": after["saturation"] - before["saturation"],
        "quality_score_delta": after["quality_score"] - before["quality_score"],
    }


def cleanup_temp_runs() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    cutoff = datetime.now() - timedelta(hours=TEMP_RETENTION_HOURS)
    for child in TEMP_ROOT.iterdir():
        try:
            if not child.is_dir():
                continue
            mtime = datetime.fromtimestamp(child.stat().st_mtime)
            if mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            continue


@dataclass
class InferenceResult:
    filename: str
    model_type: str
    threshold: int
    preprocess_mode: str
    preprocess_label: str
    verdict: str
    foreground_pixels: int
    foreground_ratio: float
    inference_ms: float
    total_ms: float
    resident_memory_mb: float
    inference_memory_mb: float
    runtime_input_color: str
    runtime_input_dtype: str
    logits_shape: list[int]
    logits_dtype: str
    mask_unique_values: list[int]
    input_brightness: float
    input_contrast: float
    input_sharpness: float
    input_saturation: float
    quality_score: float
    quality_level: str
    saved: bool
    output_dir: str
    original_url: str
    preprocessed_url: str
    mask_url: str
    mask_color_url: str
    overlay_url: str
    metadata_url: str

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "model_type": self.model_type,
            "threshold": self.threshold,
            "preprocess_mode": self.preprocess_mode,
            "preprocess_label": self.preprocess_label,
            "verdict": self.verdict,
            "foreground_pixels": self.foreground_pixels,
            "foreground_ratio": self.foreground_ratio,
            "inference_ms": self.inference_ms,
            "total_ms": self.total_ms,
            "resident_memory_mb": self.resident_memory_mb,
            "inference_memory_mb": self.inference_memory_mb,
            "runtime_input_color": self.runtime_input_color,
            "runtime_input_dtype": self.runtime_input_dtype,
            "logits_shape": self.logits_shape,
            "logits_dtype": self.logits_dtype,
            "mask_unique_values": self.mask_unique_values,
            "input_brightness": self.input_brightness,
            "input_contrast": self.input_contrast,
            "input_sharpness": self.input_sharpness,
            "input_saturation": self.input_saturation,
            "quality_score": self.quality_score,
            "quality_level": self.quality_level,
            "saved": self.saved,
            "output_dir": self.output_dir,
            "original_url": self.original_url,
            "preprocessed_url": self.preprocessed_url,
            "mask_url": self.mask_url,
            "mask_color_url": self.mask_color_url,
            "overlay_url": self.overlay_url,
            "metadata_url": self.metadata_url,
        }


@dataclass
class PreprocessPreviewResult:
    filename: str
    preprocess_mode: str
    preprocess_label: str
    input_brightness: float
    input_contrast: float
    input_sharpness: float
    input_saturation: float
    quality_score: float
    quality_level: str
    output_dir: str
    original_url: str
    preprocessed_url: str
    metadata_url: str
    original_quality_score: float | None = None
    quality_score_delta: float | None = None
    brightness_delta: float | None = None
    contrast_delta: float | None = None
    sharpness_delta: float | None = None
    saturation_delta: float | None = None

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "preprocess_mode": self.preprocess_mode,
            "preprocess_label": self.preprocess_label,
            "input_brightness": self.input_brightness,
            "input_contrast": self.input_contrast,
            "input_sharpness": self.input_sharpness,
            "input_saturation": self.input_saturation,
            "quality_score": self.quality_score,
            "quality_level": self.quality_level,
            "original_quality_score": self.original_quality_score,
            "quality_score_delta": self.quality_score_delta,
            "brightness_delta": self.brightness_delta,
            "contrast_delta": self.contrast_delta,
            "sharpness_delta": self.sharpness_delta,
            "saturation_delta": self.saturation_delta,
            "output_dir": self.output_dir,
            "original_url": self.original_url,
            "preprocessed_url": self.preprocessed_url,
            "metadata_url": self.metadata_url,
        }


class SeaformerService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runtime: RKNNLite | None = None
        self._current_model_type: str | None = None
        self._resident_rss_bytes = 0
        self._process = psutil.Process()
        cleanup_temp_runs()

    def _ensure_runtime(self, model_type: str) -> None:
        if model_type not in MODELS:
            raise ValueError(f"unsupported model type: {model_type}")
        if self._runtime is not None and self._current_model_type == model_type:
            return

        if self._runtime is not None:
            self._runtime.release()
            self._runtime = None
            self._current_model_type = None
            gc.collect()

        runtime = RKNNLite(verbose=False)
        ret = runtime.load_rknn(str(MODELS[model_type]))
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")
        try:
            ret = runtime.init_runtime()
        except TypeError:
            ret = runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")

        warmup_input = np.zeros((1, INPUT_SIZE[0], INPUT_SIZE[1], 3), dtype=np.uint8)
        runtime.inference(inputs=[warmup_input])

        self._runtime = runtime
        self._current_model_type = model_type
        self._resident_rss_bytes = self._process.memory_info().rss

    def _prepare_input(self, image_bytes: bytes, preprocess_mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, PreprocessConfig]:
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("无法解析上传图片")
        preprocessed_bgr, model_input, config = self._prepare_input_from_bgr(image_bgr, preprocess_mode)
        return image_bgr, preprocessed_bgr, model_input, config

    def _prepare_input_from_bgr(
        self, image_bgr: np.ndarray, preprocess_mode: str
    ) -> tuple[np.ndarray, np.ndarray, PreprocessConfig]:
        preprocessed_bgr, config = _apply_preprocess(image_bgr, preprocess_mode)
        resized_bgr = cv2.resize(preprocessed_bgr, (INPUT_SIZE[1], INPUT_SIZE[0]), interpolation=cv2.INTER_LINEAR)
        resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        return preprocessed_bgr, np.expand_dims(resized_rgb, axis=0), config

    def _canonicalize_logits(self, output: np.ndarray) -> np.ndarray:
        if output.ndim == 4 and output.shape[1] == NUM_CLASSES:
            return output
        if output.ndim == 4 and output.shape[-1] == NUM_CLASSES:
            return np.transpose(output, (0, 3, 1, 2))
        raise ValueError(f"unexpected logits shape: {output.shape}")

    def _create_output_dir(self, persist: bool, filename: str) -> tuple[Path, str]:
        root = SAVED_ROOT if persist else TEMP_ROOT
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = _slugify(filename)
        directory = root / f"{stamp}-{uuid.uuid4().hex[:8]}-{Path(slug).stem}"
        directory.mkdir(parents=True, exist_ok=True)
        kind = "saved" if persist else "tmp"
        return directory, kind

    def _save_artifacts(
        self,
        output_dir: Path,
        storage_kind: str,
        filename: str,
        original_bgr: np.ndarray,
        preprocessed_bgr: np.ndarray,
        mask_resized: np.ndarray,
        overlay: np.ndarray,
        color_mask: np.ndarray,
        metadata: dict,
        logits: np.ndarray,
        persist: bool,
    ) -> dict:
        safe_name = _slugify(filename)
        original_path = output_dir / safe_name
        preprocessed_path = output_dir / "preprocessed.png"
        mask_path = output_dir / "mask.png"
        mask_color_path = output_dir / "mask_color.png"
        overlay_path = output_dir / "overlay.png"
        metadata_path = output_dir / "result.json"
        logits_path = output_dir / "logits.npy"

        cv2.imwrite(str(original_path), original_bgr)
        cv2.imwrite(str(preprocessed_path), preprocessed_bgr)
        cv2.imwrite(str(mask_path), mask_resized)
        cv2.imwrite(str(mask_color_path), color_mask)
        cv2.imwrite(str(overlay_path), overlay)
        if persist:
            np.save(logits_path, logits)

        metadata.update(
            {
                "original_file": original_path.name,
                "preprocessed_file": preprocessed_path.name,
                "mask_file": mask_path.name,
                "mask_color_file": mask_color_path.name,
                "overlay_file": overlay_path.name,
            }
        )
        if persist:
            metadata["logits_file"] = logits_path.name

        with open(metadata_path, "w", encoding="utf-8") as file_obj:
            json.dump(metadata, file_obj, ensure_ascii=False, indent=2)

        base_url = f"/runtime/{storage_kind}/{output_dir.name}"
        return {
            "output_dir": str(output_dir),
            "original_url": f"{base_url}/{original_path.name}",
            "preprocessed_url": f"{base_url}/{preprocessed_path.name}",
            "mask_url": f"{base_url}/{mask_path.name}",
            "mask_color_url": f"{base_url}/{mask_color_path.name}",
            "overlay_url": f"{base_url}/{overlay_path.name}",
            "metadata_url": f"{base_url}/{metadata_path.name}",
        }

    def _save_preprocess_preview(
        self,
        *,
        output_dir: Path,
        storage_kind: str,
        filename: str,
        original_bgr: np.ndarray,
        preprocessed_bgr: np.ndarray,
        metadata: dict,
    ) -> dict:
        safe_name = _slugify(filename)
        original_path = output_dir / safe_name
        preprocessed_path = output_dir / "preprocessed.png"
        metadata_path = output_dir / "preprocess_result.json"

        cv2.imwrite(str(original_path), original_bgr)
        cv2.imwrite(str(preprocessed_path), preprocessed_bgr)

        metadata.update(
            {
                "original_file": original_path.name,
                "preprocessed_file": preprocessed_path.name,
            }
        )

        with open(metadata_path, "w", encoding="utf-8") as file_obj:
            json.dump(metadata, file_obj, ensure_ascii=False, indent=2)

        base_url = f"/runtime/{storage_kind}/{output_dir.name}"
        return {
            "output_dir": str(output_dir),
            "original_url": f"{base_url}/{original_path.name}",
            "preprocessed_url": f"{base_url}/{preprocessed_path.name}",
            "metadata_url": f"{base_url}/{metadata_path.name}",
        }

    def _infer_loaded_runtime(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        model_type: str,
        threshold: int,
        preprocess_mode: str,
        persist: bool,
    ) -> InferenceResult:
        assert self._runtime is not None
        total_t0 = time.perf_counter()
        original_bgr, preprocessed_bgr, model_input, config = self._prepare_input(image_bytes, preprocess_mode)
        quality = _measure_image_quality(preprocessed_bgr)

        rss_before = self._process.memory_info().rss
        infer_t0 = time.perf_counter()
        outputs = self._runtime.inference(inputs=[model_input])
        inference_ms = (time.perf_counter() - infer_t0) * 1000
        rss_after = self._process.memory_info().rss

        logits = self._canonicalize_logits(outputs[0])
        mask = np.argmax(logits, axis=1).astype(np.uint8)[0]
        mask_resized = cv2.resize(mask, (original_bgr.shape[1], original_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        color_mask = _colorize_mask(mask_resized)
        overlay = _overlay_mask(original_bgr, mask_resized)
        foreground_pixels = int(np.count_nonzero(mask_resized == 1))
        foreground_ratio = float(foreground_pixels / mask_resized.size)
        verdict = "NG" if foreground_pixels > threshold else "OK"

        output_dir, storage_kind = self._create_output_dir(persist=persist, filename=filename)
        metadata = {
            "filename": filename,
            "model_type": model_type,
            "threshold": threshold,
            "preprocess_mode": config.mode,
            "preprocess_label": config.label,
            "verdict": verdict,
            "foreground_pixels": foreground_pixels,
            "foreground_ratio": foreground_ratio,
            "inference_ms": inference_ms,
            "total_ms": (time.perf_counter() - total_t0) * 1000,
            "resident_memory_mb": self._resident_rss_bytes / (1024 * 1024),
            "inference_memory_mb": max(rss_after - rss_before, 0) / (1024 * 1024),
            "runtime_input_color": "RGB",
            "runtime_input_dtype": "uint8",
            "mask_unique_values": np.unique(mask_resized).tolist(),
            "logits_shape": list(logits.shape),
            "logits_dtype": str(logits.dtype),
            "input_brightness": quality["brightness"],
            "input_contrast": quality["contrast"],
            "input_sharpness": quality["sharpness"],
            "input_saturation": quality["saturation"],
            "quality_score": quality["quality_score"],
            "quality_level": quality["quality_level"],
            "saved": persist,
        }
        artifact_info = self._save_artifacts(
            output_dir=output_dir,
            storage_kind=storage_kind,
            filename=filename,
            original_bgr=original_bgr,
            preprocessed_bgr=preprocessed_bgr,
            mask_resized=mask_resized,
            overlay=overlay,
            color_mask=color_mask,
            metadata=metadata,
            logits=logits,
            persist=persist,
        )

        return InferenceResult(
            filename=filename,
            model_type=model_type,
            threshold=threshold,
            preprocess_mode=config.mode,
            preprocess_label=config.label,
            verdict=verdict,
            foreground_pixels=foreground_pixels,
            foreground_ratio=foreground_ratio,
            inference_ms=metadata["inference_ms"],
            total_ms=metadata["total_ms"],
            resident_memory_mb=metadata["resident_memory_mb"],
            inference_memory_mb=metadata["inference_memory_mb"],
            runtime_input_color="RGB",
            runtime_input_dtype="uint8",
            logits_shape=metadata["logits_shape"],
            logits_dtype=metadata["logits_dtype"],
            mask_unique_values=metadata["mask_unique_values"],
            input_brightness=metadata["input_brightness"],
            input_contrast=metadata["input_contrast"],
            input_sharpness=metadata["input_sharpness"],
            input_saturation=metadata["input_saturation"],
            quality_score=metadata["quality_score"],
            quality_level=metadata["quality_level"],
            saved=persist,
            output_dir=artifact_info["output_dir"],
            original_url=artifact_info["original_url"],
            preprocessed_url=artifact_info["preprocessed_url"],
            mask_url=artifact_info["mask_url"],
            mask_color_url=artifact_info["mask_color_url"],
            overlay_url=artifact_info["overlay_url"],
            metadata_url=artifact_info["metadata_url"],
        )

    def infer_one(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        model_type: str,
        threshold: int,
        preprocess_mode: str,
        persist: bool,
    ) -> InferenceResult:
        with self._lock:
            self._ensure_runtime(model_type)
            return self._infer_loaded_runtime(
                image_bytes=image_bytes,
                filename=filename,
                model_type=model_type,
                threshold=threshold,
                preprocess_mode=preprocess_mode,
                persist=persist,
            )

    def infer_batch(
        self,
        *,
        items: Iterable[tuple[str, bytes]],
        model_type: str,
        threshold: int,
        preprocess_mode: str,
        persist: bool,
    ) -> list[InferenceResult]:
        results: list[InferenceResult] = []
        with self._lock:
            self._ensure_runtime(model_type)
            for filename, payload in items:
                results.append(
                    self._infer_loaded_runtime(
                        image_bytes=payload,
                        filename=filename,
                        model_type=model_type,
                        threshold=threshold,
                        preprocess_mode=preprocess_mode,
                        persist=persist,
                    )
                )
        return results

    def preview_preprocess(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        preprocess_mode: str,
    ) -> PreprocessPreviewResult:
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        original_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if original_bgr is None:
            raise ValueError("无法解析上传图片")
        preprocessed_bgr, config = _apply_preprocess(original_bgr, preprocess_mode)
        original_quality = _measure_image_quality(original_bgr)
        quality = _measure_image_quality(preprocessed_bgr)
        delta = _quality_delta(original_quality, quality)
        output_dir, storage_kind = self._create_output_dir(persist=False, filename=filename)
        metadata = {
            "filename": filename,
            "preprocess_mode": config.mode,
            "preprocess_label": config.label,
            "preprocess_description": config.description,
            "input_brightness": quality["brightness"],
            "input_contrast": quality["contrast"],
            "input_sharpness": quality["sharpness"],
            "input_saturation": quality["saturation"],
            "quality_score": quality["quality_score"],
            "quality_level": quality["quality_level"],
            "original_quality_score": original_quality["quality_score"],
            **delta,
        }
        artifact_info = self._save_preprocess_preview(
            output_dir=output_dir,
            storage_kind=storage_kind,
            filename=filename,
            original_bgr=original_bgr,
            preprocessed_bgr=preprocessed_bgr,
            metadata=metadata,
        )
        return PreprocessPreviewResult(
            filename=filename,
            preprocess_mode=config.mode,
            preprocess_label=config.label,
            input_brightness=metadata["input_brightness"],
            input_contrast=metadata["input_contrast"],
            input_sharpness=metadata["input_sharpness"],
            input_saturation=metadata["input_saturation"],
            quality_score=metadata["quality_score"],
            quality_level=metadata["quality_level"],
            original_quality_score=metadata["original_quality_score"],
            quality_score_delta=metadata["quality_score_delta"],
            brightness_delta=metadata["brightness_delta"],
            contrast_delta=metadata["contrast_delta"],
            sharpness_delta=metadata["sharpness_delta"],
            saturation_delta=metadata["saturation_delta"],
            output_dir=artifact_info["output_dir"],
            original_url=artifact_info["original_url"],
            preprocessed_url=artifact_info["preprocessed_url"],
            metadata_url=artifact_info["metadata_url"],
        )

    def infer_frame_preview(
        self,
        *,
        frame_bgr: np.ndarray,
        model_type: str,
        threshold: int,
        preprocess_mode: str,
    ) -> dict:
        with self._lock:
            self._ensure_runtime(model_type)
            assert self._runtime is not None
            preprocessed_bgr, model_input, config = self._prepare_input_from_bgr(frame_bgr, preprocess_mode)
            quality = _measure_image_quality(preprocessed_bgr)
            rss_before = self._process.memory_info().rss
            infer_t0 = time.perf_counter()
            outputs = self._runtime.inference(inputs=[model_input])
            inference_ms = (time.perf_counter() - infer_t0) * 1000
            rss_after = self._process.memory_info().rss

            logits = self._canonicalize_logits(outputs[0])
            mask = np.argmax(logits, axis=1).astype(np.uint8)[0]
            mask_resized = cv2.resize(mask, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            overlay = _overlay_mask(frame_bgr, mask_resized)
            foreground_pixels = int(np.count_nonzero(mask_resized == 1))
            foreground_ratio = float(foreground_pixels / mask_resized.size)
            verdict = "NG" if foreground_pixels > threshold else "OK"
            fps_estimate = 1000.0 / inference_ms if inference_ms > 0 else 0.0
            inference_memory_mb = max(rss_after - rss_before, 0) / (1024 * 1024)
            return {
                "frame_bgr": overlay,
                "verdict": verdict,
                "foreground_pixels": foreground_pixels,
                "foreground_ratio": foreground_ratio,
                "inference_ms": inference_ms,
                "inference_memory_mb": inference_memory_mb,
                "fps_estimate": fps_estimate,
                "quality_score": quality["quality_score"],
                "quality_level": quality["quality_level"],
                "preprocess_mode": config.mode,
                "preprocess_label": config.label,
            }

    def current_model(self) -> str | None:
        return self._current_model_type

    def compare_preprocess_modes(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        modes: Iterable[str] | None = None,
    ) -> list[PreprocessPreviewResult]:
        selected_modes = list(modes or PREPROCESS_CONFIGS.keys())
        results = []
        for mode in selected_modes:
            results.append(
                self.preview_preprocess(
                    image_bytes=image_bytes,
                    filename=f"{Path(filename).stem}-{mode}{Path(filename).suffix or '.png'}",
                    preprocess_mode=mode,
                )
            )
        return results
