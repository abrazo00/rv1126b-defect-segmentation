# AFFormer RKNN 目录说明

这个目录现在只保留“原始 AFFormer 整模型导出”这一条线，不再保留早期失败尝试和混合部署文件。

## 当前保留内容

- [ORIGINAL_AFFORMER_RKNN.md](/home/shiro/AFFormer/deploy/rknn/ORIGINAL_AFFORMER_RKNN.md)
  - 原始 AFFormer 整模型导出说明
  - 记录了 ONNX、FP RKNN、INT8 RKNN 的结果和风险

- [runtime_full_original.py](/home/shiro/AFFormer/deploy/rknn/runtime_full_original.py)
  - 板端 `RKNNLite2 / RKNN Runtime` 推理脚本
  - 适用于当前保留下来的整模型 `.rknn`

- [output_original](/home/shiro/AFFormer/deploy/rknn/output_original)
  - 当前有效产物目录

- [WORKLOG.md](/home/shiro/AFFormer/deploy/rknn/WORKLOG.md)
  - 清理日志和最终目录状态

## 当前有效产物

- ONNX:
  - [afformer_tiny_cityscapes_original_logits.onnx](/home/shiro/AFFormer/deploy/rknn/output_original/afformer_tiny_cityscapes_original_logits.onnx)

- FP RKNN:
  - [afformer_tiny_cityscapes_original.rv1126b.fp.rknn](/home/shiro/AFFormer/deploy/rknn/output_original/afformer_tiny_cityscapes_original.rv1126b.fp.rknn)

- INT8 RKNN:
  - [afformer_tiny_cityscapes_original.rv1126b.int8.rknn](/home/shiro/AFFormer/deploy/rknn/output_original/afformer_tiny_cityscapes_original.rv1126b.int8.rknn)

- INT8 构建摘要:
  - [afformer_tiny_cityscapes_original_build_int8_summary.json](/home/shiro/AFFormer/deploy/rknn/output_original/afformer_tiny_cityscapes_original_build_int8_summary.json)

- 导出摘要:
  - [afformer_tiny_cityscapes_original_export_summary.json](/home/shiro/AFFormer/deploy/rknn/output_original/afformer_tiny_cityscapes_original_export_summary.json)

- INT8 量化集:
  - [quant_dataset_clean.txt](/home/shiro/AFFormer/deploy/rknn/output_original/quant_dataset_clean.txt)
  - [quant_images](/home/shiro/AFFormer/deploy/rknn/output_original/quant_images)

## 板端运行

FP:

```bash
python3 deploy/rknn/runtime_full_original.py \
  --model deploy/rknn/output_original/afformer_tiny_cityscapes_original.rv1126b.fp.rknn \
  --image your_test.jpg \
  --output-dir deploy/rknn/runtime_output_original_fp
```

INT8:

```bash
python3 deploy/rknn/runtime_full_original.py \
  --model deploy/rknn/output_original/afformer_tiny_cityscapes_original.rv1126b.int8.rknn \
  --image your_test.jpg \
  --output-dir deploy/rknn/runtime_output_original_int8
```

如果你要核对板端结果，优先看：

- `mask_match`
- `overlay.png`

不要先盯 `float logits` 的逐值误差，因为当前导出链本身包含导出期 shim。
