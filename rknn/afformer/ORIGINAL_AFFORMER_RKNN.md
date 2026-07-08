# 原始 AFFormer 整模型 RKNN 记录

这份记录只对应“最开始的 AFFormer”，不是之前为了硬转 RKNN 改过 attention 和部署结构的那一版。

## 我做了什么

1. 恢复原始 AFFormer 实现
   - [tools/afformer.py](/home/shiro/AFFormer/tools/afformer.py) 已恢复到仓库 `HEAD`
   - [mmseg/models/backbones/afformer.py](/home/shiro/AFFormer/mmseg/models/backbones/afformer.py) 已恢复为原始实现

2. 使用指定权重导出整模型 ONNX
   - checkpoint: [iter_128000.pth](/home/shiro/AFFormer/work_dirs/AFFormer_tiny_cityscapes/50交叉熵/iter_128000.pth)
   - 输入尺寸固定为 `1x3x512x512`
   - 预处理固定为：
     - `mean=[123.675, 116.28, 103.53]`
     - `std=[58.395, 57.12, 57.375]`
     - `to_rgb=True`

3. 使用 `rknn-toolkit2 2.3.2` 重新构建 `rv1126b` 的整模型 `.rknn`
   - FP 成功
   - INT8 成功

## 这次导出的关键点

原始 AFFormer 不能直接导 ONNX，卡点是 `AdaptiveAvgPool2d` 的输出尺寸不是输入尺寸的整数因子。

这次没有再改 AFFormer 源码，而是在导出进程里临时 monkeypatch 了 `LowPassModule.forward`，只为让 ONNX 导出通过。这个 shim 只存在于导出时的内存里，不会写回模型源码。

当前导出摘要见：
- [afformer_tiny_cityscapes_original_export_summary.json](/home/shiro/AFFormer/deploy/rknn/output_original/afformer_tiny_cityscapes_original_export_summary.json)

其中最关键的一项是：
- `max_abs_diff_vs_export_shim_torch = 8.278324127197266`

这说明：
- 现在已经能产出 ONNX 和 RKNN
- 但导出期的 `LowPassModule` 近似对数值有明显影响
- 这条链路更适合先验证“板端是否能稳定跑通”
- 还不能把它当成“与原始 PyTorch 完全等价”的最终精度版

## 产物

整模型 ONNX：
- [afformer_tiny_cityscapes_original_logits.onnx](/home/shiro/AFFormer/deploy/rknn/output_original/afformer_tiny_cityscapes_original_logits.onnx)

FP RKNN：
- [afformer_tiny_cityscapes_original.rv1126b.fp.rknn](/home/shiro/AFFormer/deploy/rknn/output_original/afformer_tiny_cityscapes_original.rv1126b.fp.rknn)

INT8 RKNN：
- [afformer_tiny_cityscapes_original.rv1126b.int8.rknn](/home/shiro/AFFormer/deploy/rknn/output_original/afformer_tiny_cityscapes_original.rv1126b.int8.rknn)

构建摘要：
- FP 结果可直接看终端日志，返回码为 `load_ret=0 / build_ret=0 / export_ret=0`
- INT8 摘要: [afformer_tiny_cityscapes_original_build_int8_summary.json](/home/shiro/AFFormer/deploy/rknn/output_original/afformer_tiny_cityscapes_original_build_int8_summary.json)

量化集：
- 清单: [quant_dataset_clean.txt](/home/shiro/AFFormer/deploy/rknn/output_original/quant_dataset_clean.txt)
- 样本目录: [quant_images](/home/shiro/AFFormer/deploy/rknn/output_original/quant_images)

板端推理脚本：
- [runtime_full_original.py](/home/shiro/AFFormer/deploy/rknn/runtime_full_original.py)

## INT8 转换结果

INT8 整模型转换最终结果：

- `load_ret = 0`
- `build_ret = 0`
- `export_ret = 0`

构建日志里还有两类需要注意的信息：

1. 量化告警
   - 有若干 outlier 权重值，可能影响量化精度
2. 编译告警
   - 日志末尾仍然打印了多条 `No lowering found for: Einsum_*`

但从工具链返回值看，这些没有阻止 `rv1126b` 的 `.rknn` 导出完成。

另外，INT8 构建时工具链明确提示：
- 默认输入 dtype 已从 `float32` 变成 `int8`
- 默认输出 dtype 已从 `float32` 变成 `int8`

所以板端跑 INT8 时要注意：
- 输入仍然送 `uint8` 图像，预处理交给 RKNN 配置
- 输出可能是量化后的 `int8 logits`
- 直接和 PC 侧 float logits 做数值 diff 没意义，优先比 `argmax mask`

## 板端推理示例

FP:

```bash
python3 deploy/rknn/runtime_full_original.py \
  --model deploy/rknn/output_original/afformer_tiny_cityscapes_original.rv1126b.fp.rknn \
  --image your_test.jpg \
  --reference-dir deploy/rknn/output/reference \
  --output-dir deploy/rknn/runtime_output_original_fp
```

INT8:

```bash
python3 deploy/rknn/runtime_full_original.py \
  --model deploy/rknn/output_original/afformer_tiny_cityscapes_original.rv1126b.int8.rknn \
  --image your_test.jpg \
  --reference-dir deploy/rknn/output/reference \
  --output-dir deploy/rknn/runtime_output_original_int8
```

INT8 模式下更应该看：
- `mask_match`
- 视觉叠加图 `overlay.png`

不应该把 `reference_logits.npy` 的浮点差值当成最终判断标准。

## 还没解决的问题

1. 这不是“完全原样”的数学等价导出
   - 原始 `LowPassModule` 在 ONNX 导出时必须用临时近似替掉
   - 当前最大误差偏大

2. 还没有做 RV1126B 实机精度确认
   - `.rknn` 已经生成
   - 但还没完成板端 `mask_match` 和可视化结果核验

3. 还没有得到一个“完全不需要 shim”的原始 AFFormer 导出方案
   - 现在是“可运行”
   - 还不是“严格等价”
