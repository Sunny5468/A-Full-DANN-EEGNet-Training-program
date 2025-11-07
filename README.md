# A-Full-DANN-EEGNet-Training-program
EEGNet_BCI2b/
├── configs/                          # 配置文件
│   ├── hyperparam_config.yaml        # 完整超参数搜索配置
│   ├── hyperparam_config_quick.yaml  # 快速测试配置
│   └── requirements.txt              # Python依赖
│
├── models/                           # 模型定义
│   ├── __init__.py
│   ├── models.py                     # 标准EEGNet模型
│   └── models_dann.py                # DANN-EEGNet模型
│
├── utils/                            # 工具函数
│   ├── __init__.py
│   └── preprocess.py                 # 数据预处理
│
├── scripts/                          # 执行脚本
│   ├── __init__.py
│   ├── main_EEGNet_BCI2b_LOSO.py           # 标准EEGNet训练
│   ├── main_DANN_EEGNet_BCI2b_LOSO.py      # DANN-EEGNet训练
│   ├── hyperparameter_tuning.py             # 超参数调优引擎
│   ├── run_tuning.py                        # 交互式调优启动
│   ├── view_tuning_results.py               # 结果查看器
│   └── USAGE_GUIDE.py                       # 使用指南
│
├── tests/                            # 测试脚本
│   ├── __init__.py
│   └── test_hyperparameter_system.py # 系统测试
│
├── docs/                             # 文档
│   ├── README.md                     # 项目说明
│   ├── README_DANN.md                # DANN实现说明
│   ├── README_HYPERPARAMETER_TUNING.md  # 超参数调优详细文档
│   ├── README_TUNING.md              # 调优导航
│   ├── QUICKSTART.md                 # 快速入门
│   └── PROJECT_SUMMARY.md            # 项目总结
│
├── results_DANN_EEGNet_BCI2b_LOSO/   # DANN训练结果
└── __pycache__/                      # Python缓存
