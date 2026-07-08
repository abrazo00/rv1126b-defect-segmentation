# 基于 RV1126B 的工件缺陷分割检测系统

本仓库为基于 RV1126B 的工件缺陷分割检测系统代码包，包含板端 Web 检测服务、SeaFormer/RKNN 模型转换与运行脚本、FP/INT8 模型文件及相关工程记录。

## 主要功能

- 工件缺陷分割检测
- SeaFormer 语义分割模型推理
- RKNN FP / INT8 模型部署
- RV1126B 板端 Web 检测界面
- 单张图片与批量图片推理
- 前景像素阈值 OK / NG 判定
- 图像增强与预处理策略
- 视频流检测与移动端输入链路

## 目录结构

```text
seaformer-web/   板端 FastAPI Web 检测服务与前端页面
rknn/            RKNN 模型转换、运行脚本与模型文件
```

## Web 服务运行

建议使用现有板端环境 `/userdata/elf-env`：

```bash
cd /home/elf/segment/seaformer-web
/userdata/elf-env/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

启动后可通过：

```text
http://板子IP:8000
```

访问检测界面。

## 说明

仓库中不包含私钥文件、运行缓存、大规模原始数据集和临时检测输出。完整数据集和运行产物建议单独管理。
