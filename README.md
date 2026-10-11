# Positional Encoding and Attention Algorithm Comparison

这是一个小型 PyTorch 实验项目，用于比较文本语言模型中的位置编码与注意力算法。模型采用 Decoder-only 架构，通过因果注意力预测下一个 token，并使用 TensorBoard 查看训练和验证曲线。

## 当前功能

- 正余弦位置编码（`sinusoidal`）。
- 旋转位置编码 RoPE（`rope`）。
- 标准 softmax 因果注意力，使用 PyTorch SDPA（`sdpa`）。
- 基于 `ELU(x) + 1` 和前缀累计的因果线性注意力（`linear`）。
- WikiText-2 数据预处理、Byte-level BPE 分词、文章编码缓存。
- 训练与验证 loss、验证困惑度（PPL）；TensorBoard 当前记录训练和验证 loss。

相对位置偏置尚未实现，属于后续实验内容。线性注意力是不同的注意力算法，不与 softmax 注意力等价，也不保证更快或取得更低的 PPL。

## 目录结构

```text
.
├── position_lab/
│   ├── data.py          # 数据准备、分词、缓存、Dataset
│   ├── model.py         # 模型、位置编码、注意力
│   └── train.py         # 训练、验证、TensorBoard 日志
├── data/                # 自动生成，本地保存
│   ├── tokenizer.json
│   ├── articles/
│   └── encoded/
├── runs/                # TensorBoard 日志，本地保存
├── pyproject.toml
├── requirements.txt
└── README.md
```

以下命令均在项目根目录执行。`data/`、`runs/` 和 `.venv/` 已被 `.gitignore` 排除，不随代码上传。

## 环境安装

当前开发环境使用 Python 3.13、PyTorch 2.14。`pyproject.toml` 要求 Python 3.13 或更新版本。

使用 uv 创建环境并安装运行所需依赖：

```bash
uv venv --python 3.13
uv pip install --python .venv/bin/python "torch>=2.14.1" "datasets>=5.1.0" "tokenizers>=0.21,<1" tqdm tensorboard
```

当前 `requirements.txt` 为空，暂时使用上面的显式依赖安装命令。

训练自动选择可用的 Apple MPS，否则使用 CPU。当前代码没有自动选择 CUDA；如在 NVIDIA GPU 上运行，需要调整 `train.py` 的设备选择。

## 数据准备

首次运行需要联网下载 Hugging Face 上的 `Salesforce/wikitext`，配置为 `wikitext-2-raw-v1`。

```bash
./.venv/bin/python -m position_lab.data prepare --train-tokenizer
```

该命令会：

1. 按一级标题重建 train、validation、test 的完整文章。
2. 仅用训练集训练 Byte-level BPE，目标词表大小为 8192。
3. 使用 `<pad>`、`<eos>` 特殊 token，在每篇文章末尾追加 EOS。
4. 保存分词器、文章文本和完整 token ID 序列。

日常训练直接读取缓存，不重新下载、训练分词器或编码文章。缺少缓存时会报错，请先执行准备命令。

检查现有缓存：

```bash
./.venv/bin/python -m position_lab.data load
```

如果要保留现有 tokenizer，只重建文章和编码缓存：

```bash
./.venv/bin/python -m position_lab.data prepare
```

准备命令会覆盖对应缓存；指定 `--train-tokenizer` 还会重新训练并覆盖 tokenizer。对照实验应复用同一份 tokenizer 和缓存。

样本按文章切分，不跨文章拼接。输入与标签错位一个 token；不足长度的输入补 PAD，标签补 `-100`，交叉熵忽略这些标签。注意力保持因果性。

## 训练

在 `position_lab/train.py` 的 `main()` 中修改实验配置，然后运行：

```bash
./.venv/bin/python -m position_lab.train
```

当前默认配置：

| 参数 | 值 |
| --- | --- |
| 随机种子 | 42 |
| 序列长度 | 64 |
| Batch size | 128 |
| Epochs | 5 |
| 模型层数 | 4 |
| 隐藏维度 | 256 |
| 注意力头数 | 4 |
| FFN 维度 | 1024 |
| 位置缓存最大长度 | 1024 |
| LayerNorm 结构 | Post-LN |
| Dropout | 0.0 |
| 优化器 | AdamW，betas=(0.9, 0.95) |
| 学习率 | 3e-4 |
| Weight decay | 0.0 |
| 梯度裁剪范数 | 1.0 |
| 训练文章限制 | None，使用全部训练文章 |
| 位置编码 | rope |
| 注意力算法 | linear |

位置编码与注意力算法分别选择：

```python
position_type = "rope"       # "rope" 或 "sinusoidal"
attention_type = "linear"    # "linear" 或 "sdpa"
```

首次检查可将 `train_article_limit` 改为 `2`、`batch_size` 改为 `2`、`epochs` 改为 `3`。小规模检查用于验证流程，不用于判断模型最终效果。

每轮遍历训练 loader 后评估完整验证集。平均 loss 按有效标签 token 数加权，PPL 为 `exp(validation_loss)`。每次启动训练都会新建模型，当前没有 checkpoint 保存或断点恢复功能。

`len(dataset)` 表示样本数，`len(loader)` 表示 batch 数。默认保留最后一个不足 batch size 的 batch。

## TensorBoard 曲线

训练会为每次运行创建独立的日志目录，名称包含位置编码、注意力算法、随机种子、序列长度、batch size 和时间。

在另一个终端启动：

```bash
./.venv/bin/tensorboard --logdir runs --port 6006
```

浏览器打开 [http://localhost:6006](http://localhost:6006)。

- **Scalars**：查看 `Loss/train` 和 `Loss/validation`，勾选多个运行可叠加比较曲线。
- **Text**：查看 `Config` 中记录的配置。

当前每轮完成训练和验证后写入一个点。第一轮结束前没有 loss 点，两个点之后才有连线。终端里的 `print` 不会自动转成 TensorBoard 指标。

如需记录 PPL，在 epoch 循环中添加：

```python
writer.add_scalar("PPL/validation", valid_ppl, step)
```

## 对照实验

比较位置编码时固定注意力算法；比较注意力算法时固定位置编码：

| 实验 | position_type | attention_type |
| --- | --- | --- |
| 正余弦基线 | sinusoidal | sdpa |
| RoPE 基线 | rope | sdpa |
| 正余弦 + 线性注意力 | sinusoidal | linear |
| RoPE + 线性注意力 | rope | linear |

各组使用同一份数据与 tokenizer，并保持模型规模、seed、batch size、序列长度、优化器参数及训练预算一致，从新初始化的模型开始训练。比较相同 epoch 的验证指标；正式结论建议重复多个随机种子。

速度比较应单独测量相同条件下完整前向、反向和参数更新的耗时或 tokens/s，在 MPS 上测量时同步设备。当前没有自动吞吐量测量，也尚未实现长序列评估或测试集最终评估。

## 线性注意力说明

无 RoPE 时使用正值核特征 `ELU(x) + 1`。RoPE 模式下，先计算核特征，再将旋转后的 Q/K 特征用于分子；分母使用未旋转的正值特征。因果性通过沿序列维的前缀累计实现。

该 RoPE 线性变体的分子可以包含负权重，权重不再具有 softmax 的概率归一化含义。

当前实现显式生成 `[B, H, L, D, Dv]` 的外积和前缀累计张量。虽然固定头维度时计算成本随长度线性增长，中间张量依然较大；内存不足时先减小 batch size。该实现主要用于实验验证，尚未采用专门的分块或融合内核。线性分支没有实现注意力权重 dropout，当前对照实验使用 `dropout=0`。

## 参考资料

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://proceedings.mlr.press/v119/katharopoulos20a.html)
- [PyTorch TensorBoard 文档](https://docs.pytorch.org/docs/2.14/tensorboard.html)
