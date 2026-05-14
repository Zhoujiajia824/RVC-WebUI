<div align="center">

<h1>RVC-WebUI</h1>
基于 Retrieval-based-Voice-Conversion-WebUI 的声音模型训练与推理项目<br><br>

</div>

## 项目简介

本项目基于 [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)，在原项目基础上修复了新版依赖的兼容性问题，并完成了声音模型的训练实践。

主要特点：
- 基于 VITS 的变声框架，使用 top1 检索杜绝音色泄漏
- 少量数据（10分钟+）即可训练出较好效果
- 修复了 Gradio 6.x / Matplotlib 新版本兼容性问题
- 支持 NVIDIA / AMD / Intel 显卡

## 环境配置

### 基本要求
- Python >= 3.8
- Conda（推荐）
- NVIDIA GPU（推荐 RTX 3060 及以上）

### 1. 创建 Conda 环境

```bash
conda create -n rvc python=3.10
conda activate rvc
```

### 2. 安装 PyTorch

NVIDIA GPU：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

AMD / Intel GPU (DirectML)：
```bash
pip install torch torchvision torchaudio
pip install -r requirements-dml.txt
```

### 3. 安装项目依赖

```bash
pip install -r requirements.txt
```

### 4. 安装额外依赖

训练所需的依赖（原 requirements.txt 中可能未自动安装）：
```bash
pip install tensorboard tensorboardX
```

### 5. 安装 FFmpeg

- **Windows**：下载 [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe) 和 [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe)，放置在项目根目录
- **Ubuntu/Debian**：`sudo apt install ffmpeg`
- **MacOS**：`brew install ffmpeg`

## 预训练模型下载

运行自动下载脚本：
```bash
python tools/download_models.py
```

下载完成后，以下目录应包含对应文件：

| 目录 | 文件 | 说明 |
|---|---|---|
| `assets/hubert/` | `hubert_base.pt` | HuBERT 特征提取模型 |
| `assets/pretrained/` | `f0G40k.pth`, `f0D40k.pth` 等 | v1 预训练权重 |
| `assets/pretrained_v2/` | `f0G40k.pth`, `f0D40k.pth` 等 | v2 预训练权重 |
| `assets/rmvpe/` | `rmvpe.pt` | RMVPE 音高提取模型 |
| `assets/uvr5_weights/` | 多个 `.pth` 文件 | 人声分离模型 |

## 启动 WebUI

```bash
conda activate rvc
python infer-web.py
```

浏览器打开 `http://localhost:7865` 即可使用。

## 声音模型训练流程

### Step 1：准备训练数据

将目标说话人的音频文件放入一个文件夹，要求：

| 项目 | 建议 |
|---|---|
| 音频时长 | 至少 10 分钟（推荐 30 分钟 ~ 2 小时） |
| 格式 | wav / flac / mp3（wav 最佳） |
| 采样率 | 44.1kHz 或 48kHz |
| 质量 | 无背景音乐、无噪声、无混响，仅目标说话人 |

### Step 2：在 WebUI 中执行训练

进入 **"训练"** 标签页：

1. **填写实验名称**（如 `my_voice`），选择模型版本 **v2**
2. **处理数据** — 选择目标采样率（推荐 48k），点击"处理数据"
3. **特征提取** — 音高提取算法选择 `rmvpe`，点击"特征提取"
4. **训练模型** — 设置参数后点击"训练模型"

训练参数建议：

| 参数 | 建议值 | 说明 |
|---|---|---|
| 保存频率 | 10 ~ 15 | 每隔多少 epoch 保存一次 |
| 总训练轮数 | 200 ~ 500 | 数据少则多训，数据多则少训 |
| 批量大小 | 3 ~ 4 | 取决于显存（6GB 显存建议 3-4） |
| 缓存 GPU | 是 | 加速训练 |

5. **训练特征索引** — 训练完成后点击"训练特征索引"

### Step 3：导出模型

在 **"ckpt处理"** 标签页：
1. 选择训练好的检查点（如 `G_200.pth`）
2. 点击"提取小模型"，生成最终的 `.pth` 文件到 `assets/weights/`

### Step 4：推理

在 **"模型推理"** 标签页：
1. 加载 `.pth` 模型文件和 `.index` 索引文件
2. 上传音频进行测试
3. 男生→女生音高偏移 +12，女生→男生 -12

### 训练产出文件路径

| 文件 | 路径 |
|---|---|
| 训练检查点 | `logs/<实验名>/G_*.pth`, `D_*.pth` |
| 特征索引 | `logs/<实验名>/added_IVF*.index` |
| 提取后模型 | `assets/weights/<实验名>.pth` |

## 已修复的兼容性问题

### 1. Gradio 6.x 兼容性

新版 Gradio 移除了 `queue()` 的 `concurrency_count` 参数，启动时会报错：
```
TypeError: Blocks.queue() got an unexpected keyword argument 'concurrency_count'
```

修复：`infer-web.py` 中移除 `concurrency_count` 参数。

### 2. Matplotlib 新版兼容性

新版 Matplotlib 移除了 `FigureCanvasAgg.tostring_rgb()` 方法，训练第一个 epoch 后崩溃：
```
AttributeError: 'FigureCanvasAgg' object has no attribute 'tostring_rgb'
```

修复：`infer/lib/train/utils.py` 中将 `tostring_rgb()` + `np.fromstring()` 替换为 `buffer_rgba()` + `np.frombuffer()`。

## 训练监控

训练过程中可通过 TensorBoard 监控 loss：
```bash
tensorboard --logdir logs/<实验名>/
```

## 常见问题

| 问题 | 解决方案 |
|---|---|
| `ModuleNotFoundError: No module named 'gradio'` | `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'tensorboard'` | `pip install tensorboard tensorboardX` |
| `TypeError: Blocks.queue() got an unexpected keyword argument` | 已修复，见上方兼容性说明 |
| `AttributeError: 'FigureCanvasAgg' object has no attribute 'tostring_rgb'` | 已修复，见上方兼容性说明 |
| 显存不足 | 减小 batch_size，或降低采样率到 32k |
| 训练 loss 不降 | 检查数据质量，避免过拟合 |
| 声音有电音/机械感 | 增加训练数据量，确保数据干净 |
| 链接索引到外部失败 | Windows 下硬链接权限问题，不影响使用 |

## 参考项目

- [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) - 原项目
- [ContentVec](https://github.com/auspicious3000/contentvec/)
- [VITS](https://github.com/jaywalnut310/vits)
- [HIFIGAN](https://github.com/jik876/hifi-gan)
- [RMVPE](https://github.com/Dream-High/RMVPE)
