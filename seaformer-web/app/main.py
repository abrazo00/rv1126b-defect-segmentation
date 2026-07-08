from __future__ import annotations

import base64
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .inference import PREPROCESS_CONFIGS, PROJECT_ROOT, RUNTIME_ROOT, SeaformerService


SERVICE_STARTED_AT = datetime.utcnow()
PREVIEW_LIMIT = 1
EVENT_LIMIT = 60
EVENT_PREVIEW_LIMIT = 1
VIDEO_LIBRARY_ROOT = Path("/home/elf/segment/videos_from_jpegs")
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv", ".m4v"}

app = FastAPI(title="SeaFormer Web", version="0.2.0")
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "app" / "templates"))
service = SeaformerService()
video_status_lock = threading.Lock()
recent_tasks_lock = threading.Lock()
recent_events_lock = threading.Lock()
stats_lock = threading.Lock()
recent_tasks = deque(maxlen=20)
recent_events = deque(maxlen=EVENT_LIMIT)
latest_video_frame_lock = threading.Lock()
latest_video_frame_jpeg: bytes | None = None
latest_video_status = {
    "active": False,
    "source": "",
    "model_type": "",
    "threshold": 0,
    "preprocess_mode": "standard",
    "preprocess_label": PREPROCESS_CONFIGS["standard"].label,
    "verdict": None,
    "foreground_pixels": None,
    "foreground_ratio": None,
    "inference_ms": None,
    "inference_memory_mb": None,
    "fps_estimate": None,
    "quality_score": None,
    "quality_level": None,
    "updated_at": None,
    "message": "视频检测尚未启动。",
}
stability_state = {
    "active": False,
    "started_at": None,
    "stopped_at": None,
    "baseline_frames": 0,
    "baseline_requests": 0,
    "baseline_events": 0,
}
system_stats = {
    "requests_total": 0,
    "single_requests": 0,
    "batch_requests": 0,
    "images_total": 0,
    "ok_count": 0,
    "ng_count": 0,
    "quality_warn_count": 0,
    "video_sessions": 0,
    "video_frames": 0,
    "cumulative_inference_ms": 0.0,
    "cumulative_total_ms": 0.0,
    "by_model": {
        "fp": {"images": 0, "cumulative_inference_ms": 0.0},
        "int8": {"images": 0, "cumulative_inference_ms": 0.0},
    },
}

app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "app" / "static")), name="static")
app.mount("/runtime", StaticFiles(directory=str(RUNTIME_ROOT)), name="runtime")


def _utcnow() -> str:
    return datetime.utcnow().isoformat()


def _uptime_seconds() -> float:
    return (datetime.utcnow() - SERVICE_STARTED_AT).total_seconds()


def _stats_snapshot() -> dict:
    with stats_lock:
        requests_total = system_stats["requests_total"]
        images_total = system_stats["images_total"]
        avg_inference_ms = (
            system_stats["cumulative_inference_ms"] / images_total if images_total else 0.0
        )
        avg_total_ms = system_stats["cumulative_total_ms"] / images_total if images_total else 0.0
        by_model = {}
        for model_name, payload in system_stats["by_model"].items():
            count = payload["images"]
            by_model[model_name] = {
                "images": count,
                "avg_inference_ms": payload["cumulative_inference_ms"] / count if count else 0.0,
            }
        return {
            "started_at": SERVICE_STARTED_AT.isoformat(),
            "uptime_seconds": _uptime_seconds(),
            "requests_total": requests_total,
            "single_requests": system_stats["single_requests"],
            "batch_requests": system_stats["batch_requests"],
            "images_total": images_total,
            "ok_count": system_stats["ok_count"],
            "ng_count": system_stats["ng_count"],
            "quality_warn_count": system_stats["quality_warn_count"],
            "video_sessions": system_stats["video_sessions"],
            "video_frames": system_stats["video_frames"],
            "avg_inference_ms": avg_inference_ms,
            "avg_total_ms": avg_total_ms,
            "by_model": by_model,
        }


def _stability_snapshot() -> dict:
    stats = _stats_snapshot()
    with recent_events_lock:
        event_count = len(recent_events)
        recent_error_count = sum(1 for item in recent_events if item.get("severity") == "error")
    active = bool(stability_state["active"])
    started_at = stability_state["started_at"]
    if active and started_at:
        duration_seconds = (datetime.utcnow() - datetime.fromisoformat(started_at)).total_seconds()
    elif started_at and stability_state["stopped_at"]:
        duration_seconds = (
            datetime.fromisoformat(stability_state["stopped_at"]) - datetime.fromisoformat(started_at)
        ).total_seconds()
    else:
        duration_seconds = stats["uptime_seconds"]
    return {
        "active": active,
        "started_at": started_at,
        "stopped_at": stability_state["stopped_at"],
        "duration_seconds": max(duration_seconds, 0.0),
        "service_uptime_seconds": stats["uptime_seconds"],
        "frames_since_start": max(stats["video_frames"] - stability_state["baseline_frames"], 0),
        "requests_since_start": max(stats["requests_total"] - stability_state["baseline_requests"], 0),
        "events_since_start": max(event_count - stability_state["baseline_events"], 0),
        "video_frames_total": stats["video_frames"],
        "requests_total": stats["requests_total"],
        "recent_error_count": recent_error_count,
        "video_sessions": stats["video_sessions"],
        "avg_inference_ms": stats["avg_inference_ms"],
        "avg_total_ms": stats["avg_total_ms"],
    }


def _record_event(event_type: str, title: str, detail: str, severity: str = "info", **extra) -> None:
    item = {
        "id": datetime.utcnow().strftime("%Y%m%d%H%M%S%f"),
        "created_at": _utcnow(),
        "event_type": event_type,
        "title": title,
        "detail": detail,
        "severity": severity,
        **extra,
    }
    with recent_events_lock:
        recent_events.appendleft(item)


def _record_task(task_type: str, model_type: str, threshold: int, preprocess_mode: str, summary: dict) -> None:
    item = {
        "id": datetime.utcnow().strftime("%Y%m%d%H%M%S%f"),
        "created_at": _utcnow(),
        "task_type": task_type,
        "model_type": model_type,
        "threshold": threshold,
        "preprocess_mode": preprocess_mode,
        **summary,
    }
    with recent_tasks_lock:
        recent_tasks.appendleft(item)


def _record_results(task_type: str, model_type: str, results: list) -> None:
    with stats_lock:
        system_stats["requests_total"] += 1
        if task_type == "single":
            system_stats["single_requests"] += 1
        else:
            system_stats["batch_requests"] += 1
        for item in results:
            system_stats["images_total"] += 1
            system_stats["cumulative_inference_ms"] += float(item.inference_ms)
            system_stats["cumulative_total_ms"] += float(item.total_ms)
            system_stats["ok_count"] += 1 if item.verdict == "OK" else 0
            system_stats["ng_count"] += 1 if item.verdict == "NG" else 0
            system_stats["quality_warn_count"] += 1 if item.quality_level == "风险" else 0
            system_stats["by_model"][model_type]["images"] += 1
            system_stats["by_model"][model_type]["cumulative_inference_ms"] += float(item.inference_ms)

    for item in results:
        if item.verdict == "NG":
            _record_event(
                "ng_detected",
                "发现疑似异常工件",
                f"{item.filename} 被判定为 NG，前景像素 {item.foreground_pixels}。",
                severity="warn",
                filename=item.filename,
                model_type=item.model_type,
                preprocess_mode=item.preprocess_mode,
            )
        if item.quality_level == "风险":
            _record_event(
                "quality_risk",
                "输入图像质量偏低",
                f"{item.filename} 画质得分 {item.quality_score:.1f}，建议切换预处理策略或改善光照。",
                severity="warn",
                filename=item.filename,
                model_type=item.model_type,
                preprocess_mode=item.preprocess_mode,
            )


def _set_video_status(**kwargs) -> None:
    with video_status_lock:
        if kwargs.get("active") is False:
            latest_video_status.update(
                {
                    "verdict": None,
                    "foreground_pixels": None,
                    "foreground_ratio": None,
                    "inference_ms": None,
                    "inference_memory_mb": None,
                    "fps_estimate": None,
                    "quality_score": None,
                    "quality_level": None,
                }
            )
        latest_video_status.update(kwargs)


def _set_latest_video_frame(frame_jpeg: bytes) -> None:
    global latest_video_frame_jpeg
    with latest_video_frame_lock:
        latest_video_frame_jpeg = frame_jpeg


def _resolve_video_source(source: str):
    stripped = source.strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped


def _safe_video_name(name: str) -> str:
    fallback = "upload_video.mp4"
    base = Path(name or fallback).name
    stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in Path(base).stem).strip("_") or "video"
    suffix = Path(base).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        suffix = ".mp4"
    return f"{stem}{suffix}"


def _ensure_video_library() -> None:
    VIDEO_LIBRARY_ROOT.mkdir(parents=True, exist_ok=True)


def _list_video_library() -> list[dict]:
    _ensure_video_library()
    items = []
    for path in sorted(VIDEO_LIBRARY_ROOT.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_VIDEO_SUFFIXES:
            continue
        stat = path.stat()
        items.append(
            {
                "filename": path.name,
                "path": str(path),
                "size_bytes": stat.st_size,
                "updated_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        )
    return items


def _input_status_snapshot() -> dict:
    videos = _list_video_library()
    with video_status_lock:
        video_status = dict(latest_video_status)
    return {
        "local_camera": {
            "label": "本地摄像头",
            "source": "0",
            "status": "可配置",
            "description": "支持 OpenCV 摄像头编号，可直接接入实际可用设备。",
        },
        "video_library": {
            "label": "视频文件库",
            "status": "有可用视频" if videos else "暂无视频",
            "count": len(videos),
            "latest": videos[0]["filename"] if videos else None,
        },
        "web_image_upload": {
            "label": "Web 图片上传",
            "status": "可用",
            "description": "单张、批量和预处理对比共用上传入口。",
        },
        "web_video_upload": {
            "label": "Web/手机视频上传",
            "status": "可用",
            "description": "手机浏览器访问页面后可直接上传图片或视频作为输入源。",
        },
        "mobile_camera": {
            "label": "手机摄像头",
            "status": "运行中" if video_status.get("source") == "手机摄像头" and video_status.get("active") else "待授权",
            "description": "浏览器授权后按固定间隔抽帧上传，由板端返回分割叠加结果。",
        },
        "current_video": {
            "label": "当前视频检测",
            "status": "运行中" if video_status.get("active") else "未运行",
            "source": video_status.get("source") or "",
            "message": video_status.get("message") or "",
        },
    }


def _pipeline_snapshot() -> dict:
    stats = _stats_snapshot()
    with video_status_lock:
        video_status = dict(latest_video_status)
    return {
        "nodes": [
            {
                "id": "input",
                "title": "多源输入",
                "status": "ready",
                "detail": "本地摄像头、视频库、Web 上传、手机上传统一进入检测链路。",
            },
            {
                "id": "preprocess",
                "title": "图像增强",
                "status": "ready",
                "detail": video_status.get("preprocess_label") or "标准/弱光/过曝/锐化/去噪等模式可切换。",
            },
            {
                "id": "npu",
                "title": "RKNN NPU 推理",
                "status": "ready" if service.current_model() else "standby",
                "detail": f"当前模型：{(service.current_model() or '未加载').upper()}",
            },
            {
                "id": "postprocess",
                "title": "分割后处理",
                "status": "ready",
                "detail": "输出 mask、彩色 mask、overlay，并根据前景像素阈值给出 OK/NG。",
            },
            {
                "id": "transport",
                "title": "Web 传输与存储",
                "status": "active" if video_status.get("active") else "ready",
                "detail": f"累计视频帧：{stats['video_frames']}，视频会话：{stats['video_sessions']}。",
            },
        ],
        "current": {
            "model": service.current_model(),
            "video_active": bool(video_status.get("active")),
            "video_source": video_status.get("source") or "",
            "preprocess_label": video_status.get("preprocess_label") or PREPROCESS_CONFIGS["standard"].label,
        },
    }


def _performance_summary() -> dict:
    stats = _stats_snapshot()
    fp_avg = stats["by_model"]["fp"]["avg_inference_ms"]
    int8_avg = stats["by_model"]["int8"]["avg_inference_ms"]
    speedup = fp_avg / int8_avg if fp_avg > 0 and int8_avg > 0 else 0.0
    with video_status_lock:
        video_status = dict(latest_video_status)
    return {
        "fp_avg_inference_ms": fp_avg,
        "int8_avg_inference_ms": int8_avg,
        "int8_speedup": speedup,
        "current_model": service.current_model(),
        "current_video_fps": video_status.get("fps_estimate"),
        "current_video_inference_ms": video_status.get("inference_ms"),
        "images_total": stats["images_total"],
        "video_frames": stats["video_frames"],
        "avg_inference_ms": stats["avg_inference_ms"],
    }


def _probe_video_file(path: Path) -> dict | None:
    import cv2

    capture = cv2.VideoCapture(str(path))
    opened = capture.isOpened()
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    ok, frame = capture.read() if opened else (False, None)
    capture.release()
    if not opened or not ok or frame is None:
        return None
    return {"width": width, "height": height, "fps": fps}


def _mjpeg_stream(source: str, model_type: str, threshold: int, preprocess_mode: str):
    import cv2

    capture = cv2.VideoCapture(_resolve_video_source(source))
    if not capture.isOpened():
        _set_video_status(
            active=False,
            source=source,
            model_type=model_type,
            threshold=threshold,
            preprocess_mode=preprocess_mode,
            preprocess_label=PREPROCESS_CONFIGS.get(preprocess_mode, PREPROCESS_CONFIGS["standard"]).label,
            message="无法打开视频流，请确认地址或设备编号。",
            updated_at=_utcnow(),
        )
        _record_event("video_error", "视频流打开失败", f"视频源 {source} 无法打开。", severity="error")
        raise RuntimeError("无法打开视频流，请确认地址或设备编号。")
    with stats_lock:
        system_stats["video_sessions"] += 1
    _set_video_status(
        active=True,
        source=source,
        model_type=model_type,
        threshold=threshold,
        preprocess_mode=preprocess_mode,
        preprocess_label=PREPROCESS_CONFIGS.get(preprocess_mode, PREPROCESS_CONFIGS["standard"]).label,
        message="视频流已连接，等待检测结果。",
        updated_at=_utcnow(),
    )
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                _set_video_status(
                    active=False,
                    source=source,
                    model_type=model_type,
                    threshold=threshold,
                    preprocess_mode=preprocess_mode,
                    preprocess_label=PREPROCESS_CONFIGS.get(preprocess_mode, PREPROCESS_CONFIGS["standard"]).label,
                    message="视频流已断开或没有继续返回帧。",
                    updated_at=_utcnow(),
                )
                _record_event("video_end", "视频流结束", f"视频源 {source} 已断开或无后续帧。", severity="warn")
                break
            preview = service.infer_frame_preview(
                frame_bgr=frame,
                model_type=model_type,
                threshold=threshold,
                preprocess_mode=preprocess_mode,
            )
            with stats_lock:
                system_stats["video_frames"] += 1
            _set_video_status(
                active=True,
                source=source,
                model_type=model_type,
                threshold=threshold,
                preprocess_mode=preview["preprocess_mode"],
                preprocess_label=preview["preprocess_label"],
                verdict=preview["verdict"],
                foreground_pixels=preview["foreground_pixels"],
                foreground_ratio=preview["foreground_ratio"],
                inference_ms=round(preview["inference_ms"], 3),
                inference_memory_mb=round(preview["inference_memory_mb"], 3),
                fps_estimate=round(preview["fps_estimate"], 3),
                quality_score=round(preview["quality_score"], 2),
                quality_level=preview["quality_level"],
                updated_at=_utcnow(),
                message="视频检测进行中。",
            )
            encoded_ok, encoded = cv2.imencode(".jpg", preview["frame_bgr"], [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not encoded_ok:
                continue
            _set_latest_video_frame(encoded.tobytes())
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
    finally:
        capture.release()
        _set_video_status(
            active=False,
            source=source,
            model_type=model_type,
            threshold=threshold,
            preprocess_mode=preprocess_mode,
            preprocess_label=PREPROCESS_CONFIGS.get(preprocess_mode, PREPROCESS_CONFIGS["standard"]).label,
            message="视频检测已结束。",
            updated_at=_utcnow(),
        )


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "default_threshold": 0,
            "default_host": "0.0.0.0",
            "default_port": 8000,
            "preview_limit": PREVIEW_LIMIT,
            "event_preview_limit": EVENT_PREVIEW_LIMIT,
            "preprocess_options": [vars(item) for item in PREPROCESS_CONFIGS.values()],
            "default_preprocess_mode": "standard",
        },
    )


@app.get("/api/health")
def health() -> dict:
    stats = _stats_snapshot()
    return {
        "status": "ok",
        "loaded_model": service.current_model(),
        "uptime_seconds": stats["uptime_seconds"],
        "requests_total": stats["requests_total"],
    }


@app.get("/api/stats")
def get_stats() -> dict:
    return _stats_snapshot()


@app.get("/api/events/recent")
def get_recent_events() -> dict:
    with recent_events_lock:
        return {"events": list(recent_events)}


@app.get("/api/videos/library")
def get_video_library() -> dict:
    return {"videos": _list_video_library()}


@app.get("/api/demo/overview")
def get_demo_overview() -> dict:
    return {
        "title": "RV1126B 嵌入式多源视觉增强与 NPU 分割检测系统",
        "pipeline": _pipeline_snapshot(),
        "inputs": _input_status_snapshot(),
        "performance": _performance_summary(),
        "stability": _stability_snapshot(),
    }


@app.get("/api/hardware/pipeline")
def get_hardware_pipeline() -> dict:
    return _pipeline_snapshot()


@app.get("/api/performance/summary")
def get_performance_summary() -> dict:
    return _performance_summary()


@app.get("/api/inputs/status")
def get_inputs_status() -> dict:
    return _input_status_snapshot()


@app.get("/api/robustness/scenarios")
def get_robustness_scenarios() -> dict:
    return {
        "scenarios": [
            {"id": "low_light", "title": "弱光场景", "suggested_preprocess": "low_light", "description": "展示低照度增强前后变化。"},
            {"id": "overexposure", "title": "逆光/反光场景", "suggested_preprocess": "overexposure_control", "description": "展示高光压制和局部对比恢复。"},
            {"id": "low_contrast", "title": "低对比场景", "suggested_preprocess": "low_contrast", "description": "展示灰雾和低对比画面的层次增强。"},
            {"id": "blur", "title": "模糊/运动场景", "suggested_preprocess": "motion_sharpen", "description": "展示边缘锐化和清晰度提升。"},
            {"id": "noise", "title": "噪声干扰场景", "suggested_preprocess": "denoise", "description": "展示噪声抑制后画面更稳定。"},
            {"id": "occlusion", "title": "遮挡/局部缺失场景", "suggested_preprocess": "defect_highlight", "description": "展示复杂局部纹理下的分割叠加效果。"},
        ]
    }


@app.post("/api/video/upload")
async def upload_video(video: UploadFile = File(...)):
    filename = video.filename or "upload_video.mp4"
    safe_name = _safe_video_name(filename)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        return JSONResponse(status_code=400, content={"error": "不支持的视频格式"})

    _ensure_video_library()
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    target = VIDEO_LIBRARY_ROOT / f"{timestamp}-{safe_name}"
    payload = await video.read()
    target.write_bytes(payload)

    probe = _probe_video_file(target)
    if probe is None:
        try:
            target.unlink()
        except OSError:
            pass
        _record_event("video_upload", "视频上传失败", f"{filename} 不是可用视频文件。", severity="error")
        return JSONResponse(status_code=400, content={"error": "上传文件不是可用视频，无法打开或读取帧"})

    _record_event(
        "video_upload",
        "视频上传成功",
        f"{target.name} 已保存到视频库，分辨率 {probe['width']}x{probe['height']}，FPS {probe['fps']:.2f}。",
        severity="info",
        filename=target.name,
    )
    return {
        "status": "ok",
        "filename": target.name,
        "path": str(target),
        "size_bytes": len(payload),
        **probe,
    }


@app.post("/api/infer")
async def infer_one(
    image: UploadFile = File(...),
    model_type: str = Form(...),
    threshold: int = Form(0),
    preprocess_mode: str = Form("standard"),
    save_results: bool = Form(False),
):
    try:
        payload = await image.read()
        result = service.infer_one(
            image_bytes=payload,
            filename=image.filename or "upload.png",
            model_type=model_type,
            threshold=threshold,
            preprocess_mode=preprocess_mode,
            persist=save_results,
        )
        _record_task(
            task_type="single",
            model_type=model_type,
            threshold=threshold,
            preprocess_mode=preprocess_mode,
            summary={
                "total": 1,
                "ok_count": 1 if result.verdict == "OK" else 0,
                "ng_count": 1 if result.verdict == "NG" else 0,
                "filenames": [result.filename],
                "avg_inference_ms": round(result.inference_ms, 3),
            },
        )
        _record_results("single", model_type, [result])
        return result.to_dict()
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


@app.post("/api/mobile/frame")
async def infer_mobile_frame(
    image: UploadFile = File(...),
    model_type: str = Form(...),
    threshold: int = Form(0),
    preprocess_mode: str = Form("standard"),
):
    try:
        payload = await image.read()
        frame_array = np.frombuffer(payload, dtype=np.uint8)
        frame_bgr = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return JSONResponse(status_code=400, content={"error": "无法解析手机摄像头帧"})

        preview = service.infer_frame_preview(
            frame_bgr=frame_bgr,
            model_type=model_type,
            threshold=threshold,
            preprocess_mode=preprocess_mode,
        )
        encoded_ok, encoded = cv2.imencode(".jpg", preview["frame_bgr"], [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not encoded_ok:
            return JSONResponse(status_code=400, content={"error": "手机摄像头帧编码失败"})

        frame_jpeg = encoded.tobytes()
        with stats_lock:
            system_stats["video_frames"] += 1
        _set_latest_video_frame(frame_jpeg)
        _set_video_status(
            active=True,
            source="手机摄像头",
            model_type=model_type,
            threshold=threshold,
            preprocess_mode=preview["preprocess_mode"],
            preprocess_label=preview["preprocess_label"],
            verdict=preview["verdict"],
            foreground_pixels=preview["foreground_pixels"],
            foreground_ratio=preview["foreground_ratio"],
            inference_ms=round(preview["inference_ms"], 3),
            inference_memory_mb=round(preview["inference_memory_mb"], 3),
            fps_estimate=round(preview["fps_estimate"], 3),
            quality_score=round(preview["quality_score"], 2),
            quality_level=preview["quality_level"],
            updated_at=_utcnow(),
            message="手机摄像头检测进行中。",
        )
        frame_data_url = "data:image/jpeg;base64," + base64.b64encode(frame_jpeg).decode("ascii")
        return {
            "status": "ok",
            "source": "手机摄像头",
            "frame_data_url": frame_data_url,
            "model_type": model_type,
            "threshold": threshold,
            **{key: value for key, value in preview.items() if key != "frame_bgr"},
        }
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


@app.post("/api/mobile/stop")
def stop_mobile_camera_status():
    _set_video_status(
        active=False,
        source="手机摄像头",
        message="手机摄像头检测已停止。",
        updated_at=_utcnow(),
    )
    return {"status": "ok", "video": dict(latest_video_status)}


@app.post("/api/preprocess/preview")
async def preview_preprocess(
    image: UploadFile = File(...),
    preprocess_mode: str = Form("standard"),
):
    try:
        payload = await image.read()
        result = service.preview_preprocess(
            image_bytes=payload,
            filename=image.filename or "upload.png",
            preprocess_mode=preprocess_mode,
        )
        return result.to_dict()
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


@app.post("/api/preprocess/compare")
async def compare_preprocess(
    image: UploadFile = File(...),
):
    try:
        payload = await image.read()
        results = service.compare_preprocess_modes(
            image_bytes=payload,
            filename=image.filename or "upload.png",
        )
        result_dicts = [item.to_dict() for item in results]
        best = max(result_dicts, key=lambda item: item.get("quality_score", 0.0)) if result_dicts else None
        return {
            "filename": image.filename or "upload.png",
            "best_mode": best["preprocess_mode"] if best else None,
            "best_label": best["preprocess_label"] if best else None,
            "results": result_dicts,
        }
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


@app.post("/api/infer/batch")
async def infer_batch(
    images: list[UploadFile] = File(...),
    model_type: str = Form(...),
    threshold: int = Form(0),
    preprocess_mode: str = Form("standard"),
    save_results: bool = Form(False),
):
    try:
        items = []
        for image in images:
            items.append((image.filename or "upload.png", await image.read()))
        results = service.infer_batch(
            items=items,
            model_type=model_type,
            threshold=threshold,
            preprocess_mode=preprocess_mode,
            persist=save_results,
        )
        result_dicts = [item.to_dict() for item in results]
        ok_filenames = [item.filename for item in results if item.verdict == "OK"]
        ng_filenames = [item.filename for item in results if item.verdict == "NG"]
        _record_task(
            task_type="batch",
            model_type=model_type,
            threshold=threshold,
            preprocess_mode=preprocess_mode,
            summary={
                "total": len(results),
                "ok_count": len(ok_filenames),
                "ng_count": len(ng_filenames),
                "filenames": [item.filename for item in results],
                "avg_inference_ms": round(sum(item.inference_ms for item in results) / len(results), 3)
                if results
                else 0.0,
            },
        )
        _record_results("batch", model_type, results)
        return {
            "summary": {
                "total": len(results),
                "ok_count": len(ok_filenames),
                "ng_count": len(ng_filenames),
                "ok_filenames": ok_filenames,
                "ng_filenames": ng_filenames,
                "preview_limit": PREVIEW_LIMIT,
                "avg_inference_ms": round(sum(item.inference_ms for item in results) / len(results), 3)
                if results
                else 0.0,
                "quality_warn_count": sum(1 for item in results if item.quality_level == "风险"),
            },
            "results": result_dicts,
        }
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


@app.get("/api/video/probe")
def probe_video(source: str):
    import cv2

    capture = cv2.VideoCapture(_resolve_video_source(source))
    opened = capture.isOpened()
    fps = capture.get(cv2.CAP_PROP_FPS) if opened else 0.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
    capture.release()
    if not opened:
        _set_video_status(
            active=False,
            source=source,
            message="视频源探测失败。",
            updated_at=_utcnow(),
        )
        _record_event("video_probe", "视频源探测失败", f"视频源 {source} 无法打开。", severity="error")
        return JSONResponse(status_code=400, content={"error": "无法打开视频流"})
    _set_video_status(
        active=False,
        source=source,
        message="视频源探测成功，可以开始视频检测。",
        updated_at=_utcnow(),
    )
    _record_event(
        "video_probe",
        "视频源探测成功",
        f"视频源 {source} 可用，分辨率 {width}x{height}，FPS {fps:.2f}。",
        severity="info",
    )
    return {"status": "ok", "fps": fps, "width": width, "height": height}


@app.get("/api/video/stream")
def stream_video(source: str, model_type: str = "int8", threshold: int = 0, preprocess_mode: str = "standard"):
    try:
        return StreamingResponse(
            _mjpeg_stream(source=source, model_type=model_type, threshold=threshold, preprocess_mode=preprocess_mode),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


@app.get("/api/video/status")
def get_video_status():
    with video_status_lock:
        return dict(latest_video_status)


@app.post("/api/video/stop")
def stop_video_status():
    _set_video_status(
        active=False,
        message="视频检测已手动停止。",
        updated_at=_utcnow(),
    )
    _record_event("video_stop", "视频检测已停止", "用户已手动停止视频检测展示。", severity="info")
    return {"status": "ok", "video": dict(latest_video_status)}


@app.get("/api/video/snapshot")
def get_video_snapshot():
    with latest_video_frame_lock:
        if latest_video_frame_jpeg is None:
            return JSONResponse(status_code=404, content={"error": "当前没有可用的视频帧快照"})
        content = latest_video_frame_jpeg
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return Response(
        content=content,
        media_type="image/jpeg",
        headers={"Content-Disposition": f'attachment; filename="video-snapshot-{timestamp}.jpg"'},
    )


@app.get("/api/tasks/recent")
def get_recent_tasks():
    with recent_tasks_lock:
        return {"tasks": list(recent_tasks)}


@app.post("/api/stability/start")
def start_stability_run() -> dict:
    stats = _stats_snapshot()
    with recent_events_lock:
        event_count = len(recent_events)
    stability_state.update(
        {
            "active": True,
            "started_at": _utcnow(),
            "stopped_at": None,
            "baseline_frames": stats["video_frames"],
            "baseline_requests": stats["requests_total"],
            "baseline_events": event_count,
        }
    )
    _record_event("stability", "稳定性记录开始", "已开始记录本轮运行状态。", severity="info")
    return _stability_snapshot()


@app.post("/api/stability/stop")
def stop_stability_run() -> dict:
    stability_state.update({"active": False, "stopped_at": _utcnow()})
    _record_event("stability", "稳定性记录结束", "已停止本轮运行状态记录。", severity="info")
    return _stability_snapshot()


@app.get("/api/stability/status")
def get_stability_status() -> dict:
    return _stability_snapshot()


@app.get("/api/export/demo-report")
def export_demo_report() -> dict:
    with recent_events_lock:
        events = list(recent_events)
    with recent_tasks_lock:
        tasks = list(recent_tasks)
    return {
        "generated_at": _utcnow(),
        "overview": {
            "pipeline": _pipeline_snapshot(),
            "inputs": _input_status_snapshot(),
            "performance": _performance_summary(),
            "stability": _stability_snapshot(),
            "stats": _stats_snapshot(),
        },
        "recent_events": events,
        "recent_tasks": tasks,
    }
