# Dynamic Grating SBL (DG-SBL)

[![English](https://img.shields.io/badge/Language-English-blue.svg)](README.md) [![中文](https://img.shields.io/badge/Language-中文-red.svg)](README_zh-CN.md)

本项目针对光谱信号中的非标准峰形及多峰重叠问题，提出了一种基于稀疏贝叶斯学习（SBL）的两阶段处理算法。该系统能够实现对连续动态光谱信号的高精度跟踪与稀疏重建。

> **注意**: 关于算法的详细数学原理、公式推导及伪代码，请参阅 [算法原理指南](ALGORITHM_GUIDE.md)。

## 📦 安装

1.  克隆仓库：
    ```bash
    git clone https://github.com/您的用户名/Dynamic-Grating-SBL.git
    cd Dynamic-Grating-SBL
    ```

2.  安装依赖：
    确保您已安装 Python 3.8+，然后运行：
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 快速开始

### 运行主程序
项目提供了两种主要的运行方式：

1.  **默认运行** (使用默认配置):
    ```bash
    python src/main.py
    ```

2.  **带参数运行** (自定义配置):
    ```bash
    python src/main_with_args.py --config src/config/config_full_data.json
    ```

### 运行可视化脚本
在 `scripts/` 目录下包含了一些用于展示算法效果的脚本：

```bash
# 运行两阶段算法可视化
python scripts/two_stage_visualization.py

# 运行三阶段算法可视化
python scripts/three_stage_visualization.py
```

## 📂 项目结构

```
d:\Dynamic-Grating-SBL\
├── src\
│   ├── config\          # 配置文件 (JSON)
│   ├── core\            # 核心算法实现
│   │   ├── stage1_main.py           # 阶段1: 字典学习与全局参数估计
│   │   ├── optimized_stage2_main.py # 阶段2: 在线跟踪 (SBL)
│   │   ├── ultra_fast_stage3.py     # 阶段3: 高速跟踪
│   │   └── optimized_pytorch_sbl.py # SBL 的 PyTorch 实现
│   ├── modules\         # 辅助模块与组件
│   │   ├── data_reader.py           # 数据读取
│   │   ├── dictionary_learning.py   # 字典学习逻辑
│   │   ├── direction_prediction.py  # 漂移预测
│   │   ├── peak_detection.py        # 寻峰算法
│   │   ├── signal_separation.py     # 信号分离
│   │   ├── signal_tracker.py        # 信号跟踪逻辑
│   │   ├── waveform_reconstruction.py # 波形重建
│   │   └── atom_set_manager.py      # 原子集管理
│   ├── main.py          # 主程序入口
│   └── main_with_args.py# 命令行参数入口
├── scripts\             # 工具脚本与可视化
├── tests\               # 单元测试与集成测试
├── data\                # 输入数据目录 (请在此处放置您的 .npz 或 .csv 数据)
├── output\              # 输出结果目录 (仿真结果、重建波形与日志)
├── ALGORITHM_GUIDE.md   # 详细算法原理文档
└── requirements.txt     # 项目依赖列表
```

## 📄 许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。
