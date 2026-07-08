# SeaFormer Web

基于当前 `RV1126B` 板端可运行的 `SeaFormer RKNN` 模型，提供局域网可访问的中文 Web 检测界面。

## 功能

- 单张图片上传推理
- 批量图片上传推理
- `FP / INT8` 模型切换
- 前景像素阈值判定 `OK / NG`
- 展示纯推理时间与推理阶段内存占用
- 可选保存完整结果
- 返回批量 `OK` 文件名列表
- 预留视频流检测入口

## 运行

建议使用现有板端环境 `/userdata/elf-env`：

```bash
cd /home/elf/segment/seaformer-web
/userdata/elf-env/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

启动后可通过 `http://板子IP:8000` 访问。
