# SeaFormer RKNN Worklog

## 目标

- 阅读并理解 `seaformer-seg0516` 项目目录。
- 将 `SeaFormer` 分割模型导出到 RKNN。
- 给出 INT8 转换结果。
- 提供板端推理脚本。

## 本次使用的模型

- 配置文件: `local_configs/seaformer/seaformer_small_1024x512_160k_1x8city.py`
- 权重文件: `work_dirs_1v50/seaformer_small_1024x512_160k_1x8city/iter_160000.pth`
- 输入尺寸: `512x512`
- 类别数: `2`
- 训练预处理:
  - `mean=[123.675, 116.28, 103.53]`
  - `std=[58.395, 57.12, 57.375]`
  - `to_rgb=True`

对应 RKNN 配置:

- `mean_values=[[123.675, 116.28, 103.53]]`
- `std_values=[[58.395, 57.12, 57.375]]`
- `reorder_channel='2 1 0'`

## 阅读项目后得到的关键结论

- 这个仓库是基于 `mmseg` 定制的 SeaFormer 语义分割工程。
- 主模型 backbone 在 [mmseg/models/backbones/seaformer.py](/home/shiro/seaformer-seg0516/mmseg/models/backbones/seaformer.py)。
- decode head 在 [mmseg/models/decode_heads/light_head.py](/home/shiro/seaformer-seg0516/mmseg/models/decode_heads/light_head.py)。
- 现成的 `tools/pytorch2onnx.py` 能导出 ONNX，但默认导出的最终输出包含 `softmax + argmax`，更接近推理结果，不适合直接做稳定的 RKNN 部署。
- SeaFormer 的 axial attention 子图里有大量 `ReduceMean / Resize / Reshape / Transpose / MatMul / Softmax` 组合，对 `rknn-toolkit 1.7.5 + rv1126` 不够稳定。

## 我做了什么

### 1. 核对目录和可用权重

确认了以下文件可直接用于部署:

- [local_configs/seaformer/seaformer_small_1024x512_160k_1x8city.py](/home/shiro/seaformer-seg0516/local_configs/seaformer/seaformer_small_1024x512_160k_1x8city.py)
- [local_configs/seaformer/seaformer_small.py](/home/shiro/seaformer-seg0516/local_configs/seaformer/seaformer_small.py)
- [work_dirs_1v50/seaformer_small_1024x512_160k_1x8city/iter_160000.pth](/home/shiro/seaformer-seg0516/work_dirs_1v50/seaformer_small_1024x512_160k_1x8city/iter_160000.pth)

### 2. 完整模型导出为 logits ONNX

新增了 [deploy/rknn/export_hybrid.py](/home/shiro/seaformer-seg0516/deploy/rknn/export_hybrid.py)，做了这些事:

- 将 `SyncBN` 替换为普通 `BN`，避免导出问题。
- 不再导出最终 `argmax` 分割图，而是导出固定 `512x512` 的 `logits`。
- 同时导出完整模型 ONNX 和混合部署所需的前后两段 ONNX。
- 生成参考输入输出和 `export_summary.json`。

实际导出结果:

- 完整模型 ONNX: [deploy/rknn/output/hybrid/seaformer_logits.onnx](/home/shiro/seaformer-seg0516/deploy/rknn/output/hybrid/seaformer_logits.onnx)
- 前段 ONNX: [deploy/rknn/output/hybrid/seaformer_front_smb1_smb4.onnx](/home/shiro/seaformer-seg0516/deploy/rknn/output/hybrid/seaformer_front_smb1_smb4.onnx)
- 后段 ONNX: [deploy/rknn/output/hybrid/seaformer_tail_cpu.onnx](/home/shiro/seaformer-seg0516/deploy/rknn/output/hybrid/seaformer_tail_cpu.onnx)

数值校验结果:

- PyTorch vs 完整 ONNX `max_abs_diff = 1.1444091796875e-05`
- 完整 PyTorch vs 混合 PyTorch `max_abs_diff = 0.0`
- 完整 PyTorch vs 混合 ONNX `max_abs_diff = 1.1444091796875e-05`

这说明:

- 模型本身能稳定导出 ONNX。
- 将 SeaFormer 切成 “前段卷积 + 后段 transformer/head” 后，数值上仍然和完整模型一致。

### 3. 尝试完整模型 ONNX -> RKNN

我先尝试直接把完整模型 ONNX 转成 RV1126 用的 `.rknn`。

结果:

- `load_onnx` 失败
- 报错发生在 RKNN 工具链内部 `C2T_Switcher`
- 核心异常: `IndexError: list index out of range`

结论:

- 不是 SeaFormer 不能导出 ONNX。
- 是这版 `rknn-toolkit 1.7.5` 在处理 SeaFormer attention 子图时崩在工具链内部。
- 这不是预处理参数错误，也不是 checkpoint 错误。

### 4. 改为混合部署

由于完整模型在 RKNN 工具链里不稳定，我改成了可落地的混合方案:

- NPU 前段: `smb1 -> smb4`
- CPU 后段: `trans1 -> smb5 -> trans2 -> decode_head -> resize`

这样做的原因:

- `smb1~smb4` 全是卷积路径，RKNN 更容易处理。
- 问题集中在后面的 axial attention 和相关形状变换。
- 对 RV1126B 来说，先让链路稳定跑通，比继续硬转全图更重要。

### 5. 实际做了 INT8 转换

新增了 [deploy/rknn/build_hybrid_front.py](/home/shiro/seaformer-seg0516/deploy/rknn/build_hybrid_front.py)。

它会:

- 从训练集图片目录生成量化数据集列表。
- 尝试构建前段的 FP RKNN。
- 构建前段的 INT8 RKNN。
- 保存 `build_summary.json`。

本次实际构建结果:

- 量化数据集: `32` 张训练图片
- 数据集列表: [deploy/rknn/output/hybrid/quant_dataset.txt](/home/shiro/seaformer-seg0516/deploy/rknn/output/hybrid/quant_dataset.txt)
- FP RKNN:
  - `load_ret = 0`
  - `build_ret = -1`
  - 失败位置: `load_input_meta -> DatabaseMeta.__init__ -> AssertionError`
- INT8 RKNN:
  - `load_ret = 0`
  - `build_ret = 0`
  - `export_ret = 0`

INT8 成功产物:

- [deploy/rknn/output/hybrid/seaformer_front_smb1_smb4.rv1126.int8.rknn](/home/shiro/seaformer-seg0516/deploy/rknn/output/hybrid/seaformer_front_smb1_smb4.rv1126.int8.rknn)

构建摘要:

- [deploy/rknn/output/hybrid/build_summary.json](/home/shiro/seaformer-seg0516/deploy/rknn/output/hybrid/build_summary.json)

## 新增或修改了哪些文件

新增:

- [deploy/rknn/export_hybrid.py](/home/shiro/seaformer-seg0516/deploy/rknn/export_hybrid.py)
- [deploy/rknn/build_hybrid_front.py](/home/shiro/seaformer-seg0516/deploy/rknn/build_hybrid_front.py)
- [deploy/rknn/runtime_hybrid.py](/home/shiro/seaformer-seg0516/deploy/rknn/runtime_hybrid.py)
- [deploy/rknn/WORKLOG.md](/home/shiro/seaformer-seg0516/deploy/rknn/WORKLOG.md)

生成产物:

- [deploy/rknn/output/hybrid/seaformer_logits.onnx](/home/shiro/seaformer-seg0516/deploy/rknn/output/hybrid/seaformer_logits.onnx)
- [deploy/rknn/output/hybrid/seaformer_front_smb1_smb4.onnx](/home/shiro/seaformer-seg0516/deploy/rknn/output/hybrid/seaformer_front_smb1_smb4.onnx)
- [deploy/rknn/output/hybrid/seaformer_tail_cpu.onnx](/home/shiro/seaformer-seg0516/deploy/rknn/output/hybrid/seaformer_tail_cpu.onnx)
- [deploy/rknn/output/hybrid/seaformer_front_smb1_smb4.rv1126.int8.rknn](/home/shiro/seaformer-seg0516/deploy/rknn/output/hybrid/seaformer_front_smb1_smb4.rv1126.int8.rknn)
- [deploy/rknn/output/hybrid/export_summary.json](/home/shiro/seaformer-seg0516/deploy/rknn/output/hybrid/export_summary.json)
- [deploy/rknn/output/hybrid/build_summary.json](/home/shiro/seaformer-seg0516/deploy/rknn/output/hybrid/build_summary.json)

## 板端推理脚本

板端混合推理脚本:

- [deploy/rknn/runtime_hybrid.py](/home/shiro/seaformer-seg0516/deploy/rknn/runtime_hybrid.py)

它的运行方式是:

1. 用 `RKNNLite` 加载前段 INT8 `.rknn`
2. 输入板端原始 `BGR uint8` 图像
3. 前段在 NPU 输出两个特征图
4. 将这两个特征图转成 `NCHW float32`
5. 用 `onnxruntime` 运行 CPU 后段
6. 做 `argmax`
7. 输出 `mask.png`、`overlay.png` 和 `logits.npy`

示例命令:

```bash
python deploy/rknn/runtime_hybrid.py \
  --front-rknn deploy/rknn/output/hybrid/seaformer_front_smb1_smb4.rv1126.int8.rknn \
  --tail-onnx deploy/rknn/output/hybrid/seaformer_tail_cpu.onnx \
  --metadata deploy/rknn/output/hybrid/export_summary.json \
  --image your_test.jpg \
  --output-dir deploy/rknn/runtime_out
```

## 目前还没有解决的问题

### 1. 完整 SeaFormer 还不能直接转成 RV1126 的 `.rknn`

当前阻塞:

- 完整模型 ONNX 在 `rknn-toolkit 1.7.5` 的 `load_onnx` 阶段崩掉
- 错误来自工具链内部的 `mode_switcher`

这意味着:

- SeaFormer 全图不是当前工具链下的稳定目标
- 如果一定要全模型纯 NPU，需要进一步重写 attention 子图，或者换工具链版本继续试

### 2. 前段 FP RKNN 没有构建成功

当前结果是:

- 前段 `INT8` 能成功 build/export
- 前段 `FP` 在 `build()` 阶段内部断言失败

这很不理想，但它是实际结果，不是脚本问题。

### 3. 没有做真实板端数值对比

原因:

- 当前这台 PC 上的 `rknn-toolkit 1.7.5` 已不支持 simulator
- 没有直接接入 RV1126B 开发板 runtime 环境

所以目前能确认的是:

- ONNX 数值对齐正常
- INT8 `.rknn` 已成功导出
- 板端脚本已提供

但还没有在 RV1126B 实机上跑出最终分割结果截图。

## 当前推荐的落地路线

现阶段最实际的方案是:

- 不再硬转 SeaFormer 全模型到单个 `.rknn`
- 直接使用“前段 NPU INT8 + 后段 CPU ONNXRuntime”的混合部署

原因很简单:

- 这条链路已经有真实转换产物
- 前后段数值已经在 PC 上对齐
- 板端只差实机运行验证

如果后续还要继续压到全 NPU，可以沿两个方向做:

- 重写 axial attention，进一步减少 `ReduceMean / Resize / Transpose / MatMul` 复杂组合
- 换更高版本的 RKNN 工具链或不同导出策略再试

## 2026-04-16 更新: 切换到 rknn-toolkit2 2.3.2 后的结果

### 背景

板端实际环境不是旧的 `rknn-toolkit-lite 1.x`，而是:

- `rknn-toolkit-lite2 2.3.2`
- `librknnrt 2.3.2`
- `driver 0.9.8`

这意味着:

- 之前用 `rknn-toolkit 1.7.5` 导出的 `.rknn` 和板端 runtime 不兼容
- 必须改用 `rknn-toolkit2 2.3.2` 在 PC 侧重新生成 `.rknn`

### 我新增或调整的内容

新增:

- [deploy/rknn/build_full_toolkit2.py](/home/shiro/seaformer-seg0516/deploy/rknn/build_full_toolkit2.py)
- [deploy/rknn/runtime_full_toolkit2.py](/home/shiro/seaformer-seg0516/deploy/rknn/runtime_full_toolkit2.py)

主机侧环境调整:

- 下载并解包官方 `airockchip/rknn-toolkit2 v2.3.2`
- 在 `segment` 环境里卸载了 `rknn-toolkit 1.7.5`
- 在 `segment` 环境里安装了 `rknn-toolkit2 2.3.2`

## 2026-04-17 更新: 修正 INT8 预处理与量化集

### 问题定位

之前整模型 INT8 准确率低，主要有两个风险点：

1. 量化集太小
   - 旧量化集只有 `32` 张图
   - 且样本分布过于集中

2. 量化和板端运行的颜色语义不够自洽
   - 训练侧是 `BGR` 读图，再通过 `to_rgb=True` 进入模型
   - 旧脚本里同时出现了 `quant_img_RGB2BGR=True`、`reorder_channel` 记录和板端手工 `BGR->RGB`
   - 这很容易让量化和运行阶段对输入通道语义理解不一致

### 这次调整

我重新组织了整模型导出链：

- 量化数据改用 `/home/shiro/AFFormer/data/cityscapes/leftImg8bit/train`
- 从中随机抽样 `256` 张
- 每张图先按部署尺寸缩放到 `512x512`
- 再转换成 `RGB uint8`
- 保存成 `NCHW` 的 `.npy`
- 使用 `.npy` 量化清单，避免 toolkit2 在图片加载阶段再做隐式通道处理

同时修改了脚本：

- [deploy/rknn/build_full_toolkit2.py](/home/shiro/seaformer-seg0516/deploy/rknn/build_full_toolkit2.py)
  - 默认量化集根目录改为 `/home/shiro/AFFormer/data/cityscapes/leftImg8bit/train`
  - 自动生成随机量化样本
  - 量化样本保存为 `RGB NCHW .npy`
  - `quant_img_RGB2BGR=False`

- [deploy/rknn/runtime_full_toolkit2.py](/home/shiro/seaformer-seg0516/deploy/rknn/runtime_full_toolkit2.py)
  - 明确板端输入语义为 `RGB uint8`
  - 继续由脚本将 OpenCV 读入的 `BGR` 转为 `RGB` 后再推理

### 本次重新导出结果

新的整模型构建结果：

- FP:
  - `load_ret = 0`
  - `build_ret = 0`
  - `export_ret = 0`

- INT8:
  - `load_ret = 0`
  - `build_ret = 0`
  - `export_ret = 0`

新的构建摘要：

- [deploy/rknn/output/hybrid/build_full_toolkit2_summary.json](/home/shiro/seaformer-seg0516/deploy/rknn/output/hybrid/build_full_toolkit2_summary.json)

新的量化清单：

- [deploy/rknn/output/hybrid/quant_dataset_rgb_npy.txt](/home/shiro/seaformer-seg0516/deploy/rknn/output/hybrid/quant_dataset_rgb_npy.txt)
- [deploy/rknn/output/hybrid/quant_dataset_rgb_npy_sources.txt](/home/shiro/seaformer-seg0516/deploy/rknn/output/hybrid/quant_dataset_rgb_npy_sources.txt)

新的量化样本目录：

- `deploy/rknn/output/hybrid/quant_rgb_npy/`

### 当前建议

现在应该先在板端做一组同图对比：

1. 同一张图跑 `FP`
2. 同一张图跑新的 `INT8`
3. 对比 `argmax` 后的 mask，而不是直接对比 float logits

如果这次精度仍然明显偏低，再去看：

- 量化样本数量是否还要继续扩大到 `512` 或 `1024`
- 是否需要做类别均衡抽样
- 是否有某些后处理步骤在板端和 PC 侧不一致

### 整模型转换结果

这次不再走前段混合部署，直接拿完整 `logits ONNX`:

- [deploy/rknn/output/hybrid/seaformer_logits.onnx](/home/shiro/seaformer-seg0516/deploy/rknn/output/hybrid/seaformer_logits.onnx)

用 `rknn-toolkit2 2.3.2` 针对 `rv1126b` 做整模型转换，结果如下:

- 整模型 FP:
  - `load_ret = 0`
  - `build_ret = 0`
  - `export_ret = 0`
- 整模型 INT8:
  - `load_ret = 0`
  - `build_ret = 0`
  - `export_ret = 0`

产物:

- [deploy/rknn/output/hybrid/seaformer_logits.rv1126b.fp.rknn](/home/shiro/seaformer-seg0516/deploy/rknn/output/hybrid/seaformer_logits.rv1126b.fp.rknn)
- [deploy/rknn/output/hybrid/seaformer_logits.rv1126b.int8.rknn](/home/shiro/seaformer-seg0516/deploy/rknn/output/hybrid/seaformer_logits.rv1126b.int8.rknn)

这说明:

- 之前转不动 SeaFormer 全模型，核心不是模型本身绝对不能转
- 真正的问题是 `1.7.5` 工具链和 `RV1126B + runtime 2.3.2` 的能力与格式都不匹配
- 换到 `toolkit2 2.3.2` 之后，整模型已经能正常导出

### toolkit2 和旧脚本的差异

这次踩到的关键接口变化有一个:

- `toolkit1` 用 `reorder_channel`
- `toolkit2` 不再支持 `reorder_channel`
- `toolkit2` 量化图片通道转换要用 `quant_img_RGB2BGR`

所以后续不要再拿旧的 `build_hybrid_front.py` 配置方式去跑 `toolkit2`。

### 当前最推荐的部署方式

现在优先推荐直接使用整模型 `.rknn`:

- 板端 FP 模型:
  - [deploy/rknn/output/hybrid/seaformer_logits.rv1126b.fp.rknn](/home/shiro/seaformer-seg0516/deploy/rknn/output/hybrid/seaformer_logits.rv1126b.fp.rknn)
- 板端 INT8 模型:
  - [deploy/rknn/output/hybrid/seaformer_logits.rv1126b.int8.rknn](/home/shiro/seaformer-seg0516/deploy/rknn/output/hybrid/seaformer_logits.rv1126b.int8.rknn)

板端推理脚本:

- [deploy/rknn/runtime_full_toolkit2.py](/home/shiro/seaformer-seg0516/deploy/rknn/runtime_full_toolkit2.py)

示例:

```bash
python deploy/rknn/runtime_full_toolkit2.py \
  --model deploy/rknn/output/hybrid/seaformer_logits.rv1126b.int8.rknn \
  --image your_test.jpg \
  --output-dir deploy/rknn/runtime_out
```

### 现在还没做的事

还没在你的 RV1126B 板端实际跑出最终 mask 截图。

当前已经完成的是:

- 主机侧整模型 ONNX 导出
- 主机侧整模型 FP `.rknn` 导出
- 主机侧整模型 INT8 `.rknn` 导出
- 板端 `Lite2 2.3.2` 推理脚本准备完成

还差最后一步:

- 把 `.rknn` 和 [runtime_full_toolkit2.py](/home/shiro/seaformer-seg0516/deploy/rknn/runtime_full_toolkit2.py) 拷到板端实跑验证

## 2026-04-16 板端实测补充

### 实机运行状态

这次已经直接在当前 `rv1126b` 本机完成了整模型 `rknn-toolkit-lite2 2.3.2` 推理验证，不再只是“主机侧导出完成”。

板端实际环境:

- `rknn-toolkit-lite2 2.3.2`
- `librknnrt 2.3.2`
- `driver 0.9.8`

实机运行命令:

```bash
python runtime_full_toolkit2.py \
  --model output/hybrid/seaformer_logits.rv1126b.int8.rknn \
  --image 2.jpg \
  --output-dir runtime_out/infer_2
```

可以正常输出:

- `mask.png`
- `overlay.png`
- `logits.npy`

这说明当前目录里保留的 `rv1126b` 整模型 `.rknn` 已经能在板端真正跑通。

### 1.jpg 到 8.jpg 的 INT8 板端结果

用整模型 `INT8` 在板端实测 `1.jpg ~ 8.jpg`，结果如下:

- `1.jpg`: `mask_unique_values = [0]`
- `2.jpg`: `mask_unique_values = [0, 1]`
- `3.jpg`: `mask_unique_values = [0]`
- `4.jpg`: `mask_unique_values = [0, 1]`
- `5.jpg`: `mask_unique_values = [0, 1]`
- `6.jpg`: `mask_unique_values = [0]`
- `7.jpg`: `mask_unique_values = [0, 1]`
- `8.jpg`: `mask_unique_values = [0]`

对应输出目录:

- `runtime_out/batch_1`
- `runtime_out/batch_2`
- `runtime_out/batch_3`
- `runtime_out/batch_4`
- `runtime_out/batch_5`
- `runtime_out/batch_6`
- `runtime_out/batch_7`
- `runtime_out/batch_8`

### FP 和 INT8 的板端结果复测

为了确认“精度偏低”是不是量化导致，这次又对 `1.jpg ~ 8.jpg` 同时跑了 `FP` 和 `INT8`，并直接比较板端 mask。

逐图结果:

- `1.jpg`
  - `FP = [0]`
  - `INT8 = [0]`
  - `mask_match = 1.0`
- `2.jpg`
  - `FP = [0, 1]`
  - `INT8 = [0, 1]`
  - `mask_match = 0.999945794373039`
- `3.jpg`
  - `FP = [0, 1]`
  - `INT8 = [0]`
  - `mask_match = 0.9999822653764728`
- `4.jpg`
  - `FP = [0, 1]`
  - `INT8 = [0, 1]`
  - `mask_match = 0.9999248478756608`
- `5.jpg`
  - `FP = [0, 1]`
  - `INT8 = [0, 1]`
  - `mask_match = 0.9999815031752882`
- `6.jpg`
  - `FP = [0]`
  - `INT8 = [0]`
  - `mask_match = 1.0`
- `7.jpg`
  - `FP = [0, 1]`
  - `INT8 = [0, 1]`
  - `mask_match = 0.9999930886723339`
- `8.jpg`
  - `FP = [0]`
  - `INT8 = [0]`
  - `mask_match = 1.0`

前景像素占比也一起统计了，整体都非常小:

- `2.jpg`
  - `FP 前景占比 = 0.00023767082590612002`
  - `INT8 前景占比 = 0.00020014385339462738`
- `4.jpg`
  - `FP 前景占比 = 0.0002724264507295627`
  - `INT8 前景占比 = 0.000197274326390373`
- `5.jpg`
  - `FP 前景占比 = 4.3159257660768235e-05`
  - `INT8 前景占比 = 2.466243294901042e-05`
- `7.jpg`
  - `FP 前景占比 = 6.220194899440183e-05`
  - `INT8 前景占比 = 5.529062132835718e-05`

这组结果说明:

- `INT8` 和 `FP` 的板端输出几乎一致
- 当前这批图上“前景响应太弱”更像是模型本身表现问题
- 至少从这次实测看，不像是 INT8 量化把精度明显打坏了

复测输出目录:

- `runtime_retest/fp_1 ~ fp_8`
- `runtime_retest/int8_1 ~ int8_8`

### 纯 inference() 时间

为了避免把图片读写、模型加载、`init_runtime` 和结果保存混进去，这次单独测了 `init_runtime` 完成之后的纯 `rknn_lite.inference()` 调用时间。

测试条件:

- 输入: `2.jpg`
- 先 warmup `2` 次
- 再统计 `10` 次

结果:

- `FP`
  - 平均: `145.33 ms`
  - 最小: `143.00 ms`
  - 最大: `149.40 ms`
- `INT8`
  - 平均: `95.16 ms`
  - 最小: `93.62 ms`
  - 最大: `97.70 ms`

### 推理阶段内存占用

这次还单独测了 `init_runtime` 之后的内存基线，并估算了“纯推理阶段额外占用”的峰值。

测试口径:

- 不统计模型加载前的进程启动阶段
- 先 `load_rknn + init_runtime`
- 再统计 `inference()` 阶段的 RSS 变化

结果:

- `FP`
  - `post_init_rss = 96.24 MB`
  - `inference_peak_rss = 127.61 MB`
  - `推理阶段额外峰值占用 = 31.37 MB`
  - `推理阶段平均额外占用 = 18.98 MB`
- `INT8`
  - `post_init_rss = 106.05 MB`
  - `inference_peak_rss = 136.99 MB`
  - `推理阶段额外峰值占用 = 30.94 MB`
  - `推理阶段平均额外占用 = 16.86 MB`

需要注意:

- 这里统计的是进程 RSS，不是只看 NPU 权重本身
- 它包含 `RKNN runtime`、输入输出 buffer、工作区、Python 进程等一起的实际占用
- 因此 `INT8` 的整进程常驻内存不一定比 `FP` 更小
