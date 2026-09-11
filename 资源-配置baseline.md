# mmood3d 项目运行环境与成本分析 Baseline

## 1. 项目概况

基于 LiDAR 的 3D 目标检测 OOD（Out-of-Distribution）检测项目，包含两个训练阶段：

| 阶段 | 模型 | 说明 |
|------|------|------|
| 阶段1: Base Detector | CenterPoint (Voxel 0.1m) | SparseEncoder + SECOND + SECONDFPN，完整 3D 检测器 |
| 阶段2: OOD Detector | MLP Head | 冻结 Base Detector，只训练轻量 OOD 分类头 |

数据集：nuScenes（全量约 **350GB**，含 sweeps/samples/annotations）

---

## 2. 计算需求分析

### 阶段1: Base Detector 训练
- 配置名：`8xb4_cyclic_20e` → **8 GPU × batch_size 4 = 总 batch 32**
- Epochs: **20**
- 使用 CBGS 数据增强（类别平衡采样，约 6x 数据量）
- Voxel size: [0.1, 0.1, 0.2]，sparse_shape: [41, 1024, 1024] → **显存密集型**
- 每 GPU 显存需求：约 **14-16 GB**（Voxel 0.1m 的 CenterPoint 是显存大户）
- 论文推荐 **8 GPU** 以保持学习率不变

### 阶段2: OOD Detector 训练
- batch_size: **2**（单 GPU 即可）
- Epochs: **5**
- 使用 AMP 混合精度
- Base detector 冻结（`torch.no_grad()`），只训练 MLP head
- 显存需求：约 **10-12 GB**（需加载冻结的 CenterPoint 做推理）

---

## 3. 推荐运行环境（AWS）

| 项目 | 阶段1: Base Detector | 阶段2: OOD Detector |
|------|---------------------|---------------------|
| **推荐实例** | `p3.16xlarge` | `g5.2xlarge` |
| **GPU** | 8× NVIDIA V100 (16GB) | 1× NVIDIA A10G (24GB) |
| **vCPU** | 64 | 8 |
| **内存** | 488 GB | 32 GB |
| **OS** | Ubuntu 20.04 (Deep Learning AMI) | Ubuntu 20.04 (Deep Learning AMI) |
| **CUDA** | 11.1（代码要求） | 11.1 |
| **Python** | 3.8 | 3.8 |
| **存储** | 500GB EBS gp3 | 同上（复用） |

选择 p3.16xlarge 而非 p4d 的理由：
- 代码指定 PyTorch 1.9 + CUDA 11.1，V100 完全兼容
- CenterPoint 不需要 A100 级别的算力，V100 16GB 刚好够 batch_size=4
- 价格差距巨大（$24.48 vs $32.77/hr）

---

## 4. 运行时长估算

| 阶段 | 计算过程 | 预估时长 |
|------|---------|---------|
| 数据预处理 | `create_data.py` 生成 annotation | ~1-2 小时 |
| 阶段1 训练 | nuScenes ~28k 训练帧 × 6(CBGS) ÷ 32(batch) ≈ 5,250 iter/epoch × 20 epoch，V100×8 约 1.5s/iter | **~20-24 小时** |
| 阶段2 训练 | ~28k 帧 ÷ 2(batch) ≈ 14,000 iter/epoch × 5 epoch，含冻结推理约 2s/iter | **~3-4 小时** |
| 验证/测试 | ~6k 验证帧 | ~0.5 小时 |
| **总计** | | **~25-31 小时** |

---

## 5. 成本估算

### 方案A: On-Demand（按需）

| 项目 | 单价 | 时长 | 费用 |
|------|------|------|------|
| p3.16xlarge（阶段1+预处理） | $24.48/hr | ~25 hr | **$612.00** |
| g5.2xlarge（阶段2+测试） | $1.212/hr | ~5 hr | **$6.06** |
| EBS gp3 500GB（1个月） | $0.08/GB/月 | 1 月 | **$40.00** |
| **总计** | | | **~$658** |

### 方案B: Spot 实例（推荐 ⭐）

| 项目 | 单价 | 时长 | 费用 |
|------|------|------|------|
| p3.16xlarge Spot | ~$6.08/hr (75% off) | ~25 hr | **$152.00** |
| g5.2xlarge Spot | ~$0.36/hr | ~5 hr | **$1.80** |
| EBS gp3 500GB | $0.08/GB/月 | 1 月 | **$40.00** |
| **总计** | | | **~$194** |

### 方案C: 替代方案（4 GPU，需调整学习率）

使用 `p3.8xlarge`（4× V100）替代 8× GPU：
- 单价 $12.24/hr On-Demand，训练时间翻倍约 40-48hr
- 总成本约 $490-590（On-Demand）或 ~$150（Spot）

---

## 6. 总结

| 指标 | 值 |
|------|-----|
| 最佳实例组合 | p3.16xlarge + g5.2xlarge |
| OS | Ubuntu 20.04 + DLAMI |
| GPU 显存需求 | 阶段1: 16GB×8，阶段2: 24GB×1 |
| 总训练时长 | ~25-31 小时 |
| On-Demand 总成本 | ~$658 |
| Spot 总成本（推荐） | ~$194 |
| 存储成本 | ~$40/月 |

关键建议：
- 使用 Spot 实例 + checkpoint 机制（代码已支持每 epoch 保存），可节省 ~70% 成本
- 阶段1 已有预训练权重可下载，如果只需训练 OOD head，成本仅需 **~$2-6**
- 具体定价请以 AWS 官方定价页面为准：https://aws.amazon.com/ec2/pricing/on-demand/

---

> 生成时间：2026-04-23
