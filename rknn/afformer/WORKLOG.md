# AFFormer RKNN 清理日志

日期：`2026-04-16`

## 本次做了什么

这次不是继续加新路线，而是把 `deploy/rknn` 收口成一个可交付目录，只保留当前有效的“原始 AFFormer 整模型导出”结果。

保留原则：

- 保留当前真正能用的整模型产物
- 保留板端推理脚本
- 保留原始 AFFormer 的说明文档
- 删除早期失败尝试、混合部署中间件和缓存文件

## 删除了什么

已删除：

- `__pycache__/`
- `build_hybrid_front.py`
- `build_rknn.py`
- `export_hybrid.py`
- `export_onnx.py`
- `runtime_hybrid.py`
- `runtime_infer.py`
- `output/`
- `output_original/quant_dataset.txt`

这些文件删除的原因分别是：

- `__pycache__/` 只是缓存
- `build_hybrid_front.py / export_hybrid.py / runtime_hybrid.py` 属于旧的混合部署路线
- `build_rknn.py / export_onnx.py / runtime_infer.py` 对应更早的一套旧导出/运行路径，当前目录里已经没有与它们配套的产物
- `output/` 是旧产物目录，和当前保留的 `output_original/` 重复且已过期
- `quant_dataset.txt` 是之前失败的临时量化清单，路径解析有问题，已被 `quant_dataset_clean.txt` 替代

## 当前目录结构

当前 `deploy/rknn` 只保留：

- [README.md](/home/shiro/AFFormer/deploy/rknn/README.md)
- [WORKLOG.md](/home/shiro/AFFormer/deploy/rknn/WORKLOG.md)
- [ORIGINAL_AFFORMER_RKNN.md](/home/shiro/AFFormer/deploy/rknn/ORIGINAL_AFFORMER_RKNN.md)
- [runtime_full_original.py](/home/shiro/AFFormer/deploy/rknn/runtime_full_original.py)
- [output_original](/home/shiro/AFFormer/deploy/rknn/output_original)

## 当前有效结果

整模型导出结果：

- ONNX:
  - [afformer_tiny_cityscapes_original_logits.onnx](/home/shiro/AFFormer/deploy/rknn/output_original/afformer_tiny_cityscapes_original_logits.onnx)

- FP RKNN:
  - [afformer_tiny_cityscapes_original.rv1126b.fp.rknn](/home/shiro/AFFormer/deploy/rknn/output_original/afformer_tiny_cityscapes_original.rv1126b.fp.rknn)

- INT8 RKNN:
  - [afformer_tiny_cityscapes_original.rv1126b.int8.rknn](/home/shiro/AFFormer/deploy/rknn/output_original/afformer_tiny_cityscapes_original.rv1126b.int8.rknn)

INT8 转换结果：

- `load_ret = 0`
- `build_ret = 0`
- `export_ret = 0`

详细见：

- [afformer_tiny_cityscapes_original_build_int8_summary.json](/home/shiro/AFFormer/deploy/rknn/output_original/afformer_tiny_cityscapes_original_build_int8_summary.json)

## 当前还没解决的问题

1. 当前整模型导出不是严格数学等价
   - 导出时对 `LowPassModule` 做了仅限导出进程的临时 shim
   - 这一步没有改回源码，但会影响导出数值一致性

2. 还没有完成 RV1126B 板端实测核对
   - `.rknn` 已经生成
   - 但还没用你板端的真实运行结果完成最终精度确认

3. 还没有把“完全无 shim 的原始 AFFormer 导出”做出来
   - 当前状态是“可运行、可上板测试”
   - 不是“完全等价复现原始 PyTorch”
