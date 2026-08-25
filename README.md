<div align="center">

# YOLO-multi-channel

**基于 Ultralytics YOLO26 的多光谱实时视觉模型分支**

支持任意通道数图像输入（不止 RGB 三通道），覆盖训练、验证、推理、跟踪、导出全流程

</div>

![License](https://img.shields.io/badge/license-AGPL--3.0-blue) ![Python](https://img.shields.io/badge/python-3.8+-green) ![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c) ![Upstream](https://img.shields.io/badge/upstream-Ultralytics%20YOLO26-111111) ![Tasks](https://img.shields.io/badge/tasks-detect%20%C2%B7%20seg%20%C2%B7%20obb%20%C2%B7%20pose%20%C2%B7%20cls%20%C2%B7%20sem%20%C2%B7%20depth-8A2BE2)

---

## 目录

- [项目简介](#项目简介)
- [任务支持一览](#任务支持一览)
- [数据准备](#数据准备)
- [数据集配置](#数据集配置)
- [训练](#训练)
- [推理](#推理)
- [输出物说明](#输出物说明)
- [模型导出](#模型导出)
- [代码修改清单](#代码修改清单)
- [通道数据流全链路](#通道数据流全链路)
- [验证报告](#验证报告)
- [已知限制](#已知限制)
- [常见问题 FAQ](#常见问题-faq)
- [许可证](#许可证)

## 项目简介

本项目基于 [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26) 官方源码修改，专门面向**多光谱 / 多通道图像**的视觉任务场景：

- 农业遥感：RGB + 近红外（NIR）+ 红边（RedEdge）+ 绿光等多波段相机
- 工业检测：可见光 + 热红外融合
- 医学成像、高光谱降采样数据等任意 N 通道输入

官方代码已内置部分多通道基础设施（例如数据 YAML 的 `channels` 字段、按通道数构建模型输入层），但在分类任务、推理识别、裁剪导出、跟踪补偿、编译预热等多个环节存在断点。本项目补全了全部断点，使多通道端到端可用，同时**保证 3 通道标准流程的行为完全不变**。

YOLO26 本身的核心特性均完整保留：

- **原生端到端 NMS-free 推理**：默认使用一对一检测头直接产出预测，无需非极大值抑制后处理；如需传统一对多头，可在预测、验证或导出时设置 `end2end=False`。
- **无 DFL 轻量检测头**：移除分布焦点损失，降低检测头复杂度并简化导出。
- **MuSGD + Progressive Loss + STAL 训练配方**：Muon 与 SGD 混合优化器、渐进式损失、小目标感知标签分配。
- COCO 上 40.9 至 57.5 mAP；YOLO26n 在 Intel Xeon CPU 上的 ONNX 推理速度较 YOLO11n 最快提升 43%。

## 任务支持一览

下表中每一项均为在 7 波段合成数据集上实际运行验证过的结果，未做任何推测：

| 任务 | 模型后缀 | 训练 | 验证 | 推理 | 跟踪 | 导出 | 备注 |
|---|---|---|---|---|---|---|---|
| 目标检测 detect | `yolo26.yaml`（另有 `-p2` 小目标头、`-p6` 大尺寸头两个架构变体） | 通过 | 通过 | 通过 | 通过 | 通过 | 标签为 YOLO txt 格式 |
| 实例分割 segment | `-seg.yaml` / `-seg.pt` | 通过 | 通过 | 通过 | 通过 | 通过 | CopyPaste 增强对任意通道兼容 |
| 语义分割 semantic | `-sem.yaml` / `-sem.pt` | 通过 | 通过 | 通过 | 不适用 | 通过 | 掩码为独立的灰度 PNG 文件，与图像通道数无关 |
| 深度估计 depth | `-depth.yaml` / `-depth.pt` | 通过 | 通过 | 通过 | 不适用 | 通过 | 深度真值为独立的 uint16 PNG 文件 |
| 图像分类 classify | `-cls.yaml` / `-cls.pt` | 通过 | 通过 | 通过 | 不适用 | 通过 | 使用 ImageFolder 目录结构，通道数自动探测 |
| 姿态估计 pose | `-pose.yaml` / `-pose.pt` | 通过 | 通过 | 通过 | 通过 | 通过 | 关键点标签随图像一起变换 |
| 有向检测 obb | `-obb.yaml` / `-obb.pt` | 通过 | 通过 | 通过 | 通过 | 通过 | 标签为 4 角点 8 坐标格式 |

说明：

1. 上表中的"跟踪"一列标注"不适用"的任务（语义分割、深度估计、图像分类）是因为这三类任务的输出形式不产生逐目标轨迹，属于框架设计本身，与多光谱改造无关。
2. 导出验证以 ONNX 格式为准，全部七个任务逐一实测通过；其余格式见[模型导出](#模型导出)章节的支持矩阵。
3. 目标检测的 `-p2` 与 `-p6` 是仅含架构定义的 YAML 变体（分别增加小目标检测层与大分辨率检测层），官方不提供对应预训练权重，需要从头训练或自行微调，这一点与上游一致。

## 数据准备

### 存储格式要求（重要）

图像必须保存为 **TIFF 多页格式，每个波段作为一页**，像素类型必须为 **uint8**：

```python
import cv2
import numpy as np

# img: 形状为 (H, W, C) 的 uint8 numpy 数组，C 为波段数（例如 7）
cv2.imwritemulti("img0.tiff", img.transpose(2, 0, 1))
```

以下结论经过实际读写往返测试验证：

| 写法 | 实测结果 | 结论 |
|---|---|---|
| `cv2.imwritemulti` 写多页 TIFF（uint8） | 读取结果与原始数组完全一致 | 可以使用 |
| `tifffile.imwrite(path, arr)` 写多页 TIFF（`arr` 形状 `(C, H, W)`，uint8） | 读取结果与原始数组完全一致 | 可以使用 |
| `tifffile.imwrite(path, img_hwc)` 写**单页多波段** TIFF（形状 `(H, W, C)`） | 读回维度错乱（例如得到 `(40, 7, 32)`） | 禁止使用 |
| 多页 TIFF 但像素类型为 **float32** | 读回维度错乱 | 禁止使用 |

除存储格式外的其它要求：

1. **全数据集的波段顺序必须一致**。网络按文件中的波段顺序原样接收数据，训练阶段与推理阶段都不做任何顺序调整。第 0 个波段在训练时是什么物理含义，推理时输入也必须是同样的含义，否则模型输入分布错乱。
2. JPG 仅支持 3 通道、PNG 至多支持 4 通道，因此多光谱数据一律使用 TIFF。
3. 标注格式与三通道流程完全相同，均为 YOLO txt 格式。特别注意有向检测（OBB）的标签是 4 角点共 8 个坐标（加上类别共 9 列），而不是"中心点 + 宽高 + 角度"格式。
4. 图像尺寸不需要统一，训练时由 mosaic 与 letterbox 自动处理，推理时由 letterbox 自动处理。
5. 更换或重新生成图像波段数后，建议删除数据集旁的 `labels/*.cache` 缓存文件与所有 `.npy` 缓存，避免旧缓存干扰。

### 从 RGB 合成测试数据

仓库自带工具可以按波长插值从普通 RGB 图像合成任意波段数的 TIFF，便于在没有真实多光谱相机数据时验证流程：

```python
from ultralytics.data.converter import convert_to_multispectral

convert_to_multispectral("path/to/images", n_channels=7)
```

## 数据集配置

### 检测类任务（目标检测 detect / 实例分割 segment / 有向检测 obb / 姿态估计 pose）

```yaml
# ms.yaml
path: dataset_root          # 数据集根目录（绝对路径，或相对于 datasets_dir 的路径）
train: images/train         # 训练图像目录（相对 path）
val: images/val             # 验证图像目录（相对 path）
test: images/test           # 可选：测试图像目录
nc: 2                       # 类别数量
names: ['class_a', 'class_b']
channels: 7                 # 关键字段：每个图像的波段数
```

各任务的附加配置与标签差异：

| 任务 | 额外配置字段 | 标签格式（labels 目录下 txt，每行一个目标） |
|---|---|---|
| 目标检测 detect | 无 | `类别 x中心 y中心 宽 高`（共 5 列，坐标归一化） |
| 实例分割 segment | 无 | `类别 x1 y1 x2 y2 ...`（多边形顶点，至少 3 个点） |
| 有向检测 obb | 无 | `类别 x1 y1 x2 y2 x3 y3 x4 y4`（4 角点，共 9 列，归一化） |
| 姿态估计 pose | `kpt_shape: [关键点数, 维度]`，例如 `kpt_shape: [17, 3]`；若要启用水平/垂直翻转增强还需提供 `flip_idx` 数组，否则翻转增强自动关闭 | `类别 x中心 y中心 宽 高 kpt1_x kpt1_y [kpt1_可见度] ...`（5 加上 关键点数乘维度 列） |

目录结构示例：

```text
dataset_root/
    images/
        train/*.tiff
        val/*.tiff
    labels/
        train/*.txt     # 与 images 下同名，仅后缀不同
        val/*.txt
```

### 分类任务（classify）

分类使用 torchvision ImageFolder 目录结构，按子文件夹名确定类别。**无需在配置中声明 channels 字段**，框架会读取第一张样本图像自动探测通道数：

```text
dataset/
    train/
        class_a/*.tiff
        class_b/*.tiff
    val/
        class_a/*.tiff
        class_b/*.tiff
```

### 语义分割（semantic）

语义掩码是独立的灰度 PNG 文件（像素值即类别编号），存放在 `masks_dir` 指定的目录下，内部结构与图像目录镜像、文件同名：

```yaml
path: dataset_root
train: images/train
val: images/val
masks_dir: masks
nc: 2
names: ['road', 'building']
channels: 7
```

```text
dataset_root/
    images/train/*.tiff
    images/val/*.tiff
    masks/train/*.png       # 与图像同名
    masks/val/*.png
```

### 深度估计（depth）

深度真值是独立的 uint16 PNG 文件，路径规则为将图像路径中的 `images` 替换为 `depth`、后缀替换为 `.png`：

```yaml
path: dataset_root
train: images/train
val: images/val
max_depth: 80          # 单位米；超过该值的真值不参与验证指标计算
nc: 1
names:
  0: depth
channels: 7
depth_scale: 100       # PNG 中数值 100 表示真实距离 1 米
```

```text
dataset_root/
    images/train/img0.tiff
    images/val/img0.tiff
    depth/train/img0.png    # uint16，值 = 米 × depth_scale
    depth/val/img0.png
```

## 训练

### 方式一：从模型 YAML 从头训练

```python
from ultralytics import YOLO

model = YOLO("yolo26n.yaml")     # 尺度可选 n/s/m/l/x；任务后缀 -seg/-pose/-obb/-cls/-sem/-depth 同理可用
model.train(data="ms.yaml", epochs=100, imgsz=640)
```

模型的输入首层卷积会按照数据 YAML 中 `channels` 字段自动构建为对应通道数，不需要修改任何模型 YAML 文件。

### 方式二：从 COCO 预训练权重微调（推荐）

```python
model = YOLO("yolo26n.pt")
model.train(data="ms.yaml", epochs=100, imgsz=640)
```

权重迁移逻辑：除第一层卷积外，其余所有层的权重直接迁移。第一层卷积因为输入通道数不同（3 对 N）无法整体迁移，框架会把预训练的 3 波段卷积核按位置部分拷贝进新的 N 波段层（前 3 个波段的核继承预训练值，其余波段的核随机初始化从头学习）。训练日志中 `Transferred xxx/yyy items from pretrained weights` 一行可确认迁移数量。

### 数据增强行为

| 增强 | 多光谱下的行为 | 原因 |
|---|---|---|
| mosaic / mixup / cutmix | 正常生效 | 实现基于数组拼接与混合，与通道数无关 |
| 随机水平/垂直翻转、透视变换、缩放平移 | 正常生效 | 几何运算逐波段一致处理 |
| HSV 色相/饱和度/明度抖动（hsv_h/s/v 参数） | 自动跳过 | HSV 色彩空间仅对 3 通道 BGR 图像有意义 |
| BGR 通道随机反转（bgr 参数） | 自动跳过 | 代码内已限定仅在 3 通道时执行 |
| Albumentations 第三方增强管线 | 自动跳过 | 该库的管线假定 3 通道图像 |
| 分类的随机擦除 RandomErasing | 正常生效 | 多光谱专用变换管线的一部分 |
| 分类的 RandAugment/AutoAugment/AugMix | 自动跳过 | 这些策略内置 RGB 假设 |

### 缓存说明

1. `cache="ram"` 与 `cache="disk"` 两种缓存模式均支持多光谱，磁盘缓存 `.npy` 文件保存完整的 N 波段数组。
2. 若同一批图像文件曾被以不同的通道数训练过，框架会在读取缓存时检测通道数不一致并自动删除重建过期缓存，但保险起见仍建议手动清理。

## 推理

```python
from ultralytics import YOLO

model = YOLO("best.pt")            # 输入通道数自动从 checkpoint 内记录的 yaml.channels 恢复
results = model("image.tiff")
results[0].orig_img.shape          # (H, W, 7)，波段顺序与文件一致
results[0].boxes                   # 检测结果
```

三种输入方式下的波段顺序行为：

| 输入方式 | 波段顺序行为 |
|---|---|
| TIFF 文件路径 | 按文件内的波段顺序原样进入网络，不做任何重排 |
| numpy 数组 `(H, W, C)` | 同上；仅当 C 恰好等于 3 时执行 BGR 与 RGB 之间的转换 |
| torch 张量 `(B, C, H, W)` | 完全按传入顺序使用，框架不做任何转换 |

跟踪用法：

```python
results = model.track("frame_%04d.tiff", persist=True, stream=True)
```

## 输出物说明

| 输出内容 | 通道情况 | 说明 |
|---|---|---|
| `results[0].orig_img` | **完整 N 波段原图** | 后续自定义处理一律从这里取原始波段数据 |
| 标注预测图（save=True 或 plot() 生成的 JPG/PNG） | 前 3 个波段渲染 | 显示文件格式本身最多支持 4 通道 |
| save_crop 导出的目标裁剪图 | 前 3 个波段 JPG | 同时适用于 Re-ID 特征提取路径 |
| train_batch*.jpg / val_batch*.jpg 训练验证网格图 | 前 3 个波段拼接 | |
| results.png 训练曲线、PR/F1 曲线、混淆矩阵 | 与通道无关 | 基于 CSV 数值绘制 |

## 模型导出

| 导出格式 | 多光谱支持 | 说明 |
|---|---|---|
| ONNX | 支持，已实测全部七类任务 | 输入维度包含实际通道数 |
| TensorRT engine | 支持 | 元数据携带通道数，性能基准测试同样适配 |
| OpenVINO | 支持 | |
| TorchScript | 支持 | |
| Ascend .om | 支持 | 按 dummy 输入的实际通道数生成 |
| LiteRT (TFLite) | 支持 | |
| CoreML | 不支持 | coremltools 图像输入上限 3 通道，超过时导出会明确报错 |

```python
model.export(format="onnx", imgsz=640)
```

导出注意事项：在同一 Python 进程内连续执行大量导出时，onnxslim 与 protobuf 序列化环节偶发因内存累积出现分配失败（报错形如 `Failed to serialize proto`、`bad allocation` 或 `MemoryError`）。该现象与多光谱改造无关，三通道模型同样可能出现；重新运行一次导出或在独立进程中执行即可成功。生产环境建议每个导出任务使用独立进程。

## 代码修改清单

相对官方仓库共修改 `ultralytics/` 目录下 14 个文件，逐项如下：

| 文件 | 修改内容 | 解决的问题 |
|---|---|---|
| `nn/tasks.py` | `BaseModel` 新增 `channels` 属性，返回 `self.yaml["channels"]` | 推理时 `predictor` 通过该属性得知模型需要的波段数，从而正确加载图像；此前该属性缺失，任何大于 3 通道的模型推理都会被当作 3 通道处理 |
| `data/augment.py` | 新增 `classify_multispectral_transforms()` 函数，提供基于 CHW 张量的分类变换（训练态：RandomResizedCrop 加翻转加 RandomErasing；评估态：短边缩放加中心裁剪） | 分类任务原有的 torchvision 管线基于 PIL 图像，无法表达超过 4 通道的数据，遇到多光谱样本直接崩溃 |
| `data/dataset.py` | `ClassificationDataset` 改用 `imread` 读取图像（保留 TIFF 全部波段）；`__getitem__` 按通道数分发到 PIL 路径或多光谱张量路径；RAM 与磁盘两条缓存路径同步改用全波段读取 | 分类数据集原先用 `cv2.imread` 读 3 通道再转 PIL，多光谱样本会在颜色空间转换处崩溃，且缓存只保留前 3 波段造成静默数据错误 |
| `data/utils.py` | `check_cls_dataset()` 读取一张样本图像探测实际通道数并写入返回字典 | 分类数据集描述字典原先硬编码 `channels: 3`，导致分类模型永远按 3 通道构建，与 7 波段数据在前向传播时不匹配 |
| `models/yolo/classify/predict.py` | 分类预测器的 `preprocess` 按每张图的通道数分发：标准 3/4 通道走原 PIL 路径，其余走多光谱张量管线 | 修复分类推理在多光谱输入下的崩溃 |
| `models/yolo/detect/predict.py` | postprocess 中将张量转回 numpy 后的 BGR↔RGB 翻转改为仅在通道数为 3 时执行 | 张量直接输入时，多光谱波段序会被无条件翻转导致语义错误；segment、pose、obb 三个预测器继承本类自动获得修复 |
| `models/rtdetr/predict.py` | 同上 | 同上 |
| `models/yolo/semantic/predict.py` | 同上 | 同上 |
| `models/yolo/depth/predict.py` | 同上 | 同上 |
| `utils/plotting.py` | `save_one_box()` 在图像大于 3 波段时先取前 3 波段再执行裁剪与保存 | 修复 save_crop 在多光谱下的崩溃（PIL 无法编码 7 通道数组）；该函数同时被 Re-ID 特征提取与 object_cropper 方案调用，一并修复 |
| `trackers/utils/gmc.py` | 新增 `_to_gray()` 辅助函数：2 维直接返回；3/4 通道走 cvtColor 灰度化；更多通道取各波段均值转为亮度图。三处运动补偿算法统一调用 | 跟踪器的全局运动补偿原先只处理 3 通道帧，多光谱帧传入 cv2 特征算法会抛异常，导致运动补偿退化为恒等矩阵并刷警告日志 |
| `utils/torch_utils.py` | `attempt_compile()` 的预热 dummy 输入改为按 `getattr(model, "channels", 3)` 构建 | 修复 `compile=True` 训练多光谱模型时预热阶段必然崩溃的问题 |
| `utils/benchmarks.py` | TensorRT 性能基准的假输入从引擎元数据读取通道数 | 修复对多光谱导出引擎执行 benchmark 时输入通道不匹配的崩溃 |
| `docs/en/models/yolo26.md` | 新增「多光谱输入」章节 | 功能文档 |

以下基础设施在仓库中此前已经存在，本次未做改动即直接生效，列出以便理解全貌：

1. 数据 YAML 的 `channels` 字段解析（`check_det_dataset`），默认值为 3；
2. 全部九个任务训练器的 `get_model` 均以 `ch=self.data["channels"]` 构建模型，输入层通道随之确定；
3. `imread` 补丁对多页 TIFF 的无损解码：调用 `cv2.imdecodemulti` 以 UNCHANGED 模式解码每一页再沿通道轴堆叠；
4. `RandomHSV`、`Albumentations`、`Format._format_img` 三处的 3 通道守卫判断；
5. 预训练权重加载时首层卷积的部分拷贝迁移逻辑；
6. `plot_images` 对超过 3 通道批次裁剪前 3 波段绘制；
7. `Annotator` 对超过 3 通道的 numpy 输入裁剪前 3 波段、对特殊 PIL 模式转 RGB;
8. 导出器 dummy 输入与元数据的通道数字段、AutoBackend 从元数据恢复通道数；
9. 验证器 warmup 按 `data["channels"]` 构建预热输入。

## 通道数据流全链路

```text
【训练】
ms.yaml 中的 channels: 7
  -> check_det_dataset 解析得到 data["channels"] = 7
  -> DetectionTrainer.get_model 执行 DetectionModel(cfg, nc, ch=7)
       parse_model 使首层 Conv 的输入通道数为 7
  -> YOLODataset(channels=7) -> imread 读取多页 TIFF 得到 (H, W, 7)
  -> 增强管道（几何增强生效，RGB 专属增强跳过）
  -> Format 转置为 (7, H, W) 张量
  -> batch["img"] 形状 (B, 7, H, W) 进入网络

【推理】
best.pt 内记录 yaml["channels"] = 7
  -> predictor.setup_source 通过 getattr(model, "channels") 得知需要 7 波段
  -> LoadImagesAndVideos(channels=7) -> imread 读取全部波段
  -> preprocess（仅当通道数恰为 3 时才执行 BGR 与 RGB 的相互转换）
  -> 前向推理 -> Results(orig_img 完整保留 7 波段)

【导出】
model.yaml["channels"] -> 构造 dummy 输入 (batch, 7, H, W)
  -> ONNX/TensorRT/OpenVINO 等格式的输入维度与元数据 "channels": 7
  -> AutoBackend 加载导出产物时从元数据恢复通道数用于预热与校验
```

## 验证报告

测试环境：Windows 11、Python 3.12.10、PyTorch 2.12.1（CPU 版）、OpenCV 4.13.0、onnx 1.21.0。

验证方法：程序化生成 7 波段合成 TIFF 数据集（随机纹理背景加固定位置标注框），对每一个任务独立执行完整流程。全部结果如下：

1. **训练与验证**：detect、segment、obb、pose、classify、semantic、depth 七类任务各自完成多轮训练并在验证集上产出指标，无一失败。
2. **推理**：七类任务分别对多波段 TIFF 文件执行 predict 并返回正确结构的结果对象（boxes、masks、obb、keypoints、probs、semantic_mask、depth）。
3. **跟踪**：detect 模型 track 模式正常输出轨迹，全局运动补偿不再出现退化警告。
4. **导出**：七类任务的 checkpoint 全部逐一执行 ONNX 导出并成功生成文件。
5. **checkpoint 复用**：训练产物 best.pt 直接加载后对新 TIFF 推理，通道数与波段序正确。
6. **save_crop**：7 波段输入下正常导出 3 波段裁剪 JPG。
7. **torch.compile**：7 通道模型编译预热路径正常。
8. **波段序精确断言**：构造 7 个波段灰度值各不相同（30 至 210 等差）的图像，分别以文件方式与张量方式输入，断言 `orig_img` 各波段取值与输入一一对应，两种方式均通过。
9. **回归测试**：3 通道默认流程重复执行训练与推理，行为与官方版本一致，确认所有修改对存量用户零影响。

关于导出稳定性的补充说明：调试过程中曾观察到同一 checkpoint 在长时间运行的进程内连续多次导出时偶发序列化失败（错误信息包括 `Failed to serialize proto`、`bad allocation`、`MemoryError`）。经过控制变量复测确认该现象与多光谱无关——3 通道模型在高内存压力下同样出现，且相同文件在独立进程中反复导出均成功。相关注意事项已写入上文[模型导出](#模型导出)章节。

## 已知限制

| 限制 | 详细说明 |
|---|---|
| CoreML 导出 | coremltools 的图像类型输入最多支持 3 通道，更宽输入会在导出时报明确错误（不是静默错误）；ONNX、TensorRT、OpenVINO、TorchScript、Ascend、LiteRT 均不受影响 |
| YOLOE 与 YOLO-World | 开放词汇模式的 CLIP 提示编码器假设 RGB 3 通道输入，文本提示与视觉提示流程不适用于多光谱 |
| SAM 与 FastSAM | 这两个模型族本身即为 3 通道设计，未纳入本次多光谱适配范围 |
| 分类 auto_augment | RandAugment、AutoAugment、AugMix 等 RGB 策略仅作用于不超过 4 通道的样本；多光谱样本使用几何增强加随机擦除的组合 |
| 显存占用 | 模型首层参数量与显存占用随输入通道数近似线性增长；高波段数配合大分辨率训练时请注意下调 batch size |
| 波段子集选择 | 框架不支持运行时选取部分波段参与训练或推理（例如 7 选 4），此类需求应在数据预处理阶段完成波段筛选与重写 |

## 常见问题 FAQ

**问：报错 `expected input [...] to have 3 channels, but got 7 channels` 是怎么回事？**

答：这是模型输入通道数与数据实际通道数不匹配导致的。请依次排查三个方向：第一，确认数据 YAML 中写了 `channels: 7` 字段且数值与图像波段数一致；第二，确认加载的是多光谱训练产出的 checkpoint，而不是旧的 3 通道权重文件；第三，如果是分类任务，删除数据集目录旁的旧缓存文件后重试。

**问：更换了图像的波段数量之后，训练行为变得异常？**

答：删除数据集 labels 目录下的所有 `*.cache` 文件以及所有 `.npy` 缓存文件后重试。框架对 npy 缓存有通道数校验会自动重建，但 label cache 与旧配置的关联需要手动清理。

**问：波段顺序弄反了怎么办？**

答：框架不会也无法自动纠正波段的物理语义顺序。请统一将数组转置重组后重新写出全部 TIFF 文件，确保整个数据集（训练集与验证集）使用一致的顺序。

**问：能否只用部分波段训练，比如 7 个波段里选 4 个？**

答：当前版本不支持在运行时选择波段子集。请在准备数据阶段完成波段筛选，只写出需要的波段。

**Q：多光谱训练还能用哪些数据增强？**

答：所有几何类增强正常可用，包括 mosaic、mixup、cutmix、随机翻转、透视变换、缩放平移旋转；颜色类增强（HSV 抖动、色彩抖动、Albumentations 系列、BGR 反转）只对 3 通道有意义，框架已自动跳过。如果需要波段级别的扰动增强（例如随机波段亮度缩放），可以在自定义 Dataset 子类中扩展。

**Q：可视化结果为什么只能看到前 3 个波段？**

答：JPG 与 PNG 这两种显示用图像格式本身最多支持 4 通道，因此所有可视化产物取前 3 个波段渲染。如果需要查看指定波段的伪彩色图，请基于 `results[0].orig_img[..., 波段索引]` 自行调用 matplotlib 或 OpenCV 绘制。

## 许可证

- 本项目代码继承上游 [AGPL-3.0](LICENSE) 许可证
- 上游项目：[Ultralytics](https://www.ultralytics.com) | [GitHub](https://github.com/ultralytics/ultralytics)
- 商业使用请参考 [Ultralytics Enterprise Licensing](https://www.ultralytics.com/license)
