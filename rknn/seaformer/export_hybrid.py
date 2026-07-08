import argparse
import json
import os

import mmcv
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from mmcv.runner import load_checkpoint
from mmseg.apis.inference import LoadImage
from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor


def convert_sync_batchnorm(module):
    out = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        out = torch.nn.BatchNorm2d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            out.weight.data = module.weight.data.clone().detach()
            out.bias.data = module.bias.data.clone().detach()
            out.weight.requires_grad = module.weight.requires_grad
            out.bias.requires_grad = module.bias.requires_grad
        out.running_mean = module.running_mean
        out.running_var = module.running_var
        out.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        out.add_module(name, convert_sync_batchnorm(child))
    return out


def build_model(config_path, checkpoint_path):
    cfg = mmcv.Config.fromfile(config_path)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))
    model = convert_sync_batchnorm(model)
    load_checkpoint(model, checkpoint_path, map_location="cpu")
    model.eval()
    return cfg, model


def prepare_input(cfg, image_path, input_size):
    pipeline = cfg.data.test.pipeline
    pipeline[1]["img_scale"] = (input_size[1], input_size[0])
    pipeline[1]["transforms"][0]["keep_ratio"] = False
    pipeline = Compose([LoadImage()] + pipeline[1:])
    data = pipeline(dict(img=image_path))
    img_tensor = data["img"][0].unsqueeze(0)

    bgr = mmcv.imread(image_path, channel_order="bgr")
    bgr = mmcv.imresize(bgr, (input_size[1], input_size[0]))
    bgr_nhwc = np.expand_dims(bgr, axis=0).astype(np.uint8)
    return img_tensor, bgr_nhwc


class FullLogitsWrapper(torch.nn.Module):
    def __init__(self, seg_model):
        super().__init__()
        self.seg_model = seg_model

    def forward(self, x):
        return self.seg_model.encode_decode(x, None)


class HybridFrontWrapper(torch.nn.Module):
    def __init__(self, seg_model):
        super().__init__()
        backbone = seg_model.backbone
        self.smb1 = backbone.smb1
        self.smb2 = backbone.smb2
        self.smb3 = backbone.smb3
        self.smb4 = backbone.smb4

    def forward(self, x):
        x = self.smb1(x)
        x = self.smb2(x)
        detail_feat = x
        x = self.smb3(x)
        mid_feat = self.smb4(x)
        return detail_feat, mid_feat


class HybridTailWrapper(torch.nn.Module):
    def __init__(self, seg_model, input_size):
        super().__init__()
        backbone = seg_model.backbone
        self.trans1 = backbone.trans1
        self.smb5 = backbone.smb5
        self.trans2 = backbone.trans2
        self.decode_head = seg_model.decode_head
        self.align_corners = seg_model.align_corners
        self.input_size = input_size

    def forward(self, detail_feat, mid_feat):
        x = self.trans1(mid_feat)
        feat160 = x
        x = self.smb5(x)
        x = self.trans2(x)
        feat192 = x
        logits = self.decode_head([detail_feat, feat160, feat192])
        logits = F.interpolate(
            logits,
            size=self.input_size,
            mode="bilinear",
            align_corners=self.align_corners,
        )
        return logits


def export_onnx(model, inputs, output_path, input_names, output_names, opset):
    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            output_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            do_constant_folding=True,
        )
    onnx.checker.check_model(onnx.load(output_path))


def verify_single_io(onnx_path, input_name, inputs, torch_output):
    session = ort.InferenceSession(onnx_path)
    ort_output = session.run(None, {input_name: inputs})[0]
    return float(np.max(np.abs(torch_output - ort_output))), ort_output


def main():
    parser = argparse.ArgumentParser(description="Export SeaFormer ONNX and hybrid deployment assets.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="deploy/rknn/output/hybrid")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--opset", type=int, default=11)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_size = (args.height, args.width)
    cfg, model = build_model(args.config, args.checkpoint)
    img_tensor, bgr_nhwc = prepare_input(cfg, args.image, input_size)

    full_wrapper = FullLogitsWrapper(model).eval()
    front_wrapper = HybridFrontWrapper(model).eval()
    tail_wrapper = HybridTailWrapper(model, input_size).eval()

    with torch.no_grad():
        full_logits = full_wrapper(img_tensor).numpy()
        detail_feat, mid_feat = front_wrapper(img_tensor)
        hybrid_logits = tail_wrapper(detail_feat, mid_feat).numpy()

    full_onnx = os.path.join(args.output_dir, "seaformer_logits.onnx")
    front_onnx = os.path.join(args.output_dir, "seaformer_front_smb1_smb4.onnx")
    tail_onnx = os.path.join(args.output_dir, "seaformer_tail_cpu.onnx")

    export_onnx(full_wrapper, img_tensor, full_onnx, ["input"], ["logits"], args.opset)
    export_onnx(
        front_wrapper,
        img_tensor,
        front_onnx,
        ["input"],
        ["detail_feat", "mid_feat"],
        args.opset,
    )
    export_onnx(
        tail_wrapper,
        (detail_feat, mid_feat),
        tail_onnx,
        ["detail_feat", "mid_feat"],
        ["logits"],
        args.opset,
    )

    full_diff, full_ort = verify_single_io(full_onnx, "input", img_tensor.numpy(), full_logits)

    front_sess = ort.InferenceSession(front_onnx)
    front_ort = front_sess.run(None, {"input": img_tensor.numpy()})
    tail_sess = ort.InferenceSession(tail_onnx)
    tail_ort = tail_sess.run(
        None,
        {"detail_feat": front_ort[0], "mid_feat": front_ort[1]},
    )[0]

    np.save(os.path.join(args.output_dir, "sample_input_nchw.npy"), img_tensor.numpy())
    np.save(os.path.join(args.output_dir, "sample_input_bgr_uint8_nhwc.npy"), bgr_nhwc)
    np.save(os.path.join(args.output_dir, "sample_full_logits.npy"), full_logits)
    np.save(os.path.join(args.output_dir, "sample_hybrid_logits.npy"), hybrid_logits)
    np.save(os.path.join(args.output_dir, "sample_detail_feat.npy"), detail_feat.numpy())
    np.save(os.path.join(args.output_dir, "sample_mid_feat.npy"), mid_feat.numpy())

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "sample_image": args.image,
        "input_size": [args.height, args.width],
        "mean": [123.675, 116.28, 103.53],
        "std": [58.395, 57.12, 57.375],
        "to_rgb": True,
        "rknn_reorder_channel": "2 1 0",
        "num_classes": int(model.num_classes),
        "full_onnx": os.path.relpath(full_onnx),
        "front_onnx": os.path.relpath(front_onnx),
        "tail_onnx": os.path.relpath(tail_onnx),
        "torch_vs_full_onnx_max_abs_diff": full_diff,
        "torch_full_vs_hybrid_torch_max_abs_diff": float(np.max(np.abs(full_logits - hybrid_logits))),
        "torch_full_vs_hybrid_onnx_max_abs_diff": float(np.max(np.abs(full_logits - tail_ort))),
        "detail_feat_shape": list(detail_feat.shape),
        "mid_feat_shape": list(mid_feat.shape),
        "logits_shape": list(full_logits.shape),
    }

    summary_path = os.path.join(args.output_dir, "export_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
