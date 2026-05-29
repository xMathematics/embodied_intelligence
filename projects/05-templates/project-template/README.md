# 项目模板

## 概述

本目录包含标准项目模板，用于快速创建新项目。

## 使用方法

```bash
cp -r project-template/ ../new-project/
```

## 模板结构

```
project-template/
├── README.md           # 项目说明文档
├── requirements.txt    # Python依赖列表
├── .gitignore          # Git忽略文件
├── LICENSE             # 许可证文件
├── src/               # 源代码目录
│   ├── __init__.py
│   ├── main.py        # 主入口文件
│   └── modules/       # 功能模块
├── data/              # 数据集目录
├── models/            # 模型文件目录
├── results/           # 实验结果目录
├── config/            # 配置文件目录
└── tests/             # 测试目录
```

## 文件说明

| 文件/目录 | 说明 | 是否必需 |
|----------|------|---------|
| README.md | 项目说明文档 | 必需 |
| requirements.txt | Python依赖列表 | 必需 |
| .gitignore | Git忽略配置 | 推荐 |
| LICENSE | 许可证文件 | 推荐 |
| src/ | 源代码目录 | 必需 |
| data/ | 数据集目录 | 可选 |
| models/ | 模型文件目录 | 可选 |
| results/ | 实验结果目录 | 可选 |
| config/ | 配置文件目录 | 可选 |
| tests/ | 测试目录 | 推荐 |
