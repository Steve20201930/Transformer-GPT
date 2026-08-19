<div align="center">

[🇨🇳 中文](#中文) | [🇺🇸 English](#English) | [Examples](#result) | [How to use](#use)

</div>

# English

<a id="english"></a>

# From Scratch Transformer in C++

> **Learning Machine Learning by Building It From Scratch.**

This repository records my journey of learning machine learning and deep learning by implementing the underlying algorithms myself in **C++**, without relying on high-level deep learning frameworks.

The project started from simple matrix operations and neural networks, and gradually evolved into a small Transformer-based language model with:

- BPE tokenization
- Token Embedding
- Positional Embedding
- Multi-Head Self-Attention
- Causal Attention Mask
- Layer Normalization
- Residual Connections
- Feed-Forward Networks
- Backpropagation
- Adam Optimizer
- Gradient Clipping
- Learning Rate Scheduling
- Autoregressive Language Modeling
- Supervised Fine-Tuning (SFT)

The main purpose of this project is **not to build a competitive LLM**, but to understand what is actually happening inside one.

---

## Motivation

I didn't want to learn machine learning only by calling APIs such as:

```python
model.fit(...)
```

or:

```python
optimizer.step()
```

I wanted to understand what these operations actually mean.

So I decided to implement the important components myself.

Instead of treating a neural network as a black box, I wanted to answer questions such as:

- What exactly happens during a forward pass?
- Where do gradients come from?
- How does backpropagation propagate through matrix multiplication?
- How does attention actually work?
- Why does LayerNorm need a backward pass?
- How does Adam update every parameter?
- How does a language model learn to predict the next token?
- What changes when a pretrained language model is fine-tuned?

This repository is the result of that process.

---

# My Learning Journey

The project did not start as a Transformer.

It evolved step by step.

```text
C++
 │
 ├── Matrix Operations
 │
 ├── Neural Network Basics
 │
 ├── Forward Propagation
 │
 ├── Backpropagation
 │
 ├── CNN From Scratch
 │
 ├── Gradient Descent
 │
 ├── Adam Optimizer
 │
 ├── Tokenization
 │
 ├── BPE
 │
 ├── Embeddings
 │
 ├── Self-Attention
 │
 ├── Multi-Head Attention
 │
 ├── Transformer Block
 │
 ├── Full Backpropagation
 │
 ├── Language Model Training
 │
 └── Supervised Fine-Tuning
```

---

# Stage 1 — Learning C++

The project is intentionally written using relatively basic C++ data structures.

For example, matrices are represented as:

```cpp
typedef vector<vector<float>> Matrix;
```

Instead of using libraries such as Eigen, I implemented matrix multiplication myself:

```cpp
Matrix matmul(const Matrix &A, const Matrix &B){
    ...
}
```

This eventually became one of the most important building blocks of the entire project.

I also implemented:

- Matrix multiplication
- Matrix transpose
- Matrix addition
- Matrix concatenation
- File I/O
- Random initialization
- Parameter serialization
- Basic data structures

---

# Stage 2 — Building a Neural Network From Scratch

Before understanding Transformers, I needed to understand ordinary neural networks.

The first important concept was:

> **Forward propagation + loss + backward propagation + parameter update**

The basic training loop became:

```text
Input
  ↓
Forward
  ↓
Prediction
  ↓
Loss
  ↓
Backward
  ↓
Gradient
  ↓
Optimizer
  ↓
Updated Parameters
```

This was where I started understanding that a neural network is essentially a large collection of parameters updated according to gradients.

---

# Stage 3 — CNN From Scratch

I then implemented a small CNN entirely in C++.

The architecture included:

```text
Input
 ↓
Convolution
 ↓
ReLU
 ↓
Max Pooling
 ↓
Flatten
 ↓
Fully Connected Layer
 ↓
Softmax
```

I implemented both forward and backward propagation.

For example, the convolution layer had its own gradient calculation, while max pooling required storing the location of the maximum value so that gradients could be routed back correctly.

This stage taught me an important lesson:

> The difficult part of machine learning is often not the forward pass, but correctly deriving and implementing the backward pass.

---

# Stage 4 — Understanding Backpropagation

After implementing the CNN, I started implementing the derivatives manually.

For matrix multiplication:

```text
C = AB
```

the gradients are:

```text
dA = dC · Bᵀ

dB = Aᵀ · dC
```

This eventually became:

```cpp
void matmul_backward(
    const Matrix &A,
    const Matrix &B,
    const Matrix &dC,
    Matrix &dA,
    Matrix &dB
)
```

I then gradually implemented backward propagation for:

- Matrix multiplication
- ReLU
- Softmax + Cross Entropy
- Max Pooling
- Layer Normalization
- Attention
- Embedding lookup

Understanding these derivatives was one of the most important parts of the project.

---

# Stage 5 — From Classification to Language Modeling

After understanding CNNs, I wanted to build something closer to a language model.

Instead of predicting one class from one input, the task became:

> Given previous tokens, predict the next token.

For example:

```text
The cat is
       ↓
     next?
```

The model learns a probability distribution over the vocabulary:

```text
cat      0.01
dog      0.03
sleeping 0.20
...
```

The training objective is next-token prediction.

The loss is cross entropy:

```text
L = -log P(correct_token)
```

---

# Stage 6 — Tokenization and BPE

A language model cannot directly process raw text.

I therefore implemented a tokenizer using **Byte Pair Encoding (BPE)**.

The pipeline is approximately:

```text
Raw Text
   ↓
UTF-8 Character Splitting
   ↓
Boundary Detection
   ↓
Vocabulary Lookup
   ↓
BPE Merge Operations
   ↓
Token IDs
```

The BPE implementation loads:

```text
bpe_vocab.txt
bpe_merges.txt
```

and reconstructs the merge process during tokenization.

I also added support for special tokens such as:

```text
<|assistant|>
<|user|>
...
```

This later became important when I started experimenting with supervised fine-tuning.

---

# Stage 7 — Embeddings

After tokenization, each token ID is mapped to a vector.

For example:

```text
Token ID
   ↓
Embedding Table
   ↓
384-dimensional vector
```

The project currently uses:

```cpp
int d = 384;
```

The input representation is:

```text
Token Embedding
       +
Position Embedding
       ↓
Transformer Input
```

The embedding table itself is trained through backpropagation.

---

# Stage 8 — Implementing Self-Attention

This was one of the biggest steps.

The fundamental attention equation is:

```text
Attention(Q, K, V)
=
softmax(QKᵀ / √dₖ)V
```

I implemented this directly in C++.

The process is:

```text
X
│
├── XWQ → Q
├── XWK → K
└── XWV → V
       │
       ↓
     QKᵀ
       │
       ↓
   / √dₖ
       │
       ↓
 Causal Mask
       │
       ↓
    Softmax
       │
       ↓
       × V
       │
       ↓
 Attention Output
```

The causal mask ensures that a token cannot see future tokens.

For example:

```text
1  -∞  -∞  -∞
1   1  -∞  -∞
1   1   1  -∞
1   1   1   1
```

This allows the model to perform autoregressive prediction.

---

# Stage 9 — Multi-Head Attention

A single attention mechanism is limited.

I therefore implemented multiple attention heads.

The current configuration is:

```cpp
int d = 384;
int dk = 384;
int h = 6;
```

So each head operates on:

```text
384 / 6 = 64 dimensions
```

Conceptually:

```text
                    ┌── Head 1 ──┐
                    ├── Head 2 ──┤
Input → Q/K/V ──────┼── Head 3 ──┼──→ Concatenate
                    ├── Head 4 ──┤
                    ├── Head 5 ──┤
                    └── Head 6 ──┘
                              ↓
                             Wo
                              ↓
                           Output
```

Each head can learn different relationships between tokens.

---

# Stage 10 — Transformer Block

After implementing attention, I assembled the components into a Transformer block.

The current block is approximately:

```text
                Input
                  │
                  ▼
             LayerNorm
                  │
                  ▼
        Multi-Head Attention
                  │
                  ▼
        Linear Projection (Wo)
                  │
                  ▼
          Residual Connection
                  │
                  ▼
             LayerNorm
                  │
                  ▼
              FFN / MLP
                  │
                  ▼
          Residual Connection
                  │
                  ▼
                Output
```

The Feed-Forward Network is:

```text
X
 ↓
Wup
 ↓
ReLU
 ↓
Wdown
 ↓
Output
```

The implementation uses:

```cpp
int dimension = 4 * d;
```

Therefore the hidden dimension is:

```text
1536
```

for:

```text
d = 384
```

---

# Stage 11 — Building Multiple Transformer Layers

A single Transformer block is not enough.

The project currently uses:

```cpp
int N = 8;
```

Therefore the architecture contains 8 Transformer blocks:

```text
Embedding
   ↓
Transformer Block 1
   ↓
Transformer Block 2
   ↓
Transformer Block 3
   ↓
Transformer Block 4
   ↓
Transformer Block 5
   ↓
Transformer Block 6
   ↓
Transformer Block 7
   ↓
Transformer Block 8
   ↓
Linear Output
   ↓
Softmax
```

The model predicts the next token from the final hidden representation.

---

# Stage 12 — Implementing Transformer Backpropagation

This was probably the most difficult part of the project.

The backward pass needs to travel through:

```text
Loss
 ↓
Softmax
 ↓
Output Projection
 ↓
Residual Connection
 ↓
FFN
 ↓
LayerNorm
 ↓
Residual Connection
 ↓
Attention Output Projection
 ↓
Multi-Head Attention
 ↓
Causal Mask
 ↓
Softmax
 ↓
QKᵀ
 ↓
Q / K / V
 ↓
Embedding
```

For attention, I implemented gradients through:

```text
Attention
    ↓
V
    ↓
Softmax
    ↓
Causal Mask
    ↓
Scaling
    ↓
QKᵀ
    ↓
Q / K
    ↓
WQ / WK / WV
```

This was a major milestone because the model was no longer just a Transformer **forward simulator**.

It became a trainable neural network.

---

# Stage 13 — LayerNorm Backpropagation

LayerNorm is another component that required manual derivative implementation.

The forward operation is approximately:

```text
μ = mean(X)

σ = sqrt(var(X) + ε)

X̂ = (X - μ) / σ

Y = γX̂ + β
```

The implementation stores intermediate values such as:

```cpp
Xhat
sigma
```

during the forward pass.

These values are then reused during backpropagation.

The project therefore separates:

```text
BlockParams
BlockCache
BlockParamGrads
```

This makes it possible to store:

- Parameters
- Forward-pass intermediate values
- Gradients

independently.

---

# Stage 14 — Adam Optimizer

At first, simple gradient descent was enough to understand optimization.

Later I implemented Adam.

The optimizer maintains:

```text
m = first moment
v = second moment
```

and applies bias correction:

```text
m̂ = m / (1 - β₁ᵗ)

v̂ = v / (1 - β₂ᵗ)
```

The final update is:

```text
θ ← θ - α · m̂ / (√v̂ + ε)
```

The implementation stores optimizer states separately for every parameter matrix.

For example:

```cpp
BlockAdamState
```

contains optimizer states for:

```text
WQ
WK
WV
Wo
gamma
beta
gamma2
beta2
Wup
Wdown
```

---

# Stage 15 — Gradient Clipping

While training deeper networks, gradients can become unstable.

I therefore added gradient clipping:

```cpp
void clip(Matrix &grad, float limit)
```

The current limit is:

```cpp
float clipLimit = 1.0f;
```

Each gradient element is restricted to:

```text
[-1, 1]
```

This is a simple but useful technique for preventing excessively large updates.

---

# Stage 16 — Learning Rate Scheduling

The learning rate is not kept constant.

I implemented cosine decay:

```text
lr(e)
=
lr_min
+
(lr_max - lr_min)
×
0.5(1 + cos(πe / E))
```

This allows the learning rate to gradually decrease as training progresses.

For example, the current SFT configuration uses approximately:

```text
maximum learning rate = 2e-5
minimum learning rate = 2e-6
```

---

# Stage 17 — Supervised Fine-Tuning

After implementing ordinary next-token language modeling, I started experimenting with supervised fine-tuning.

The training data is structured around special tokens such as:

```text
<|assistant|>
```

For SFT, the model does not need to learn equally from every token.

Instead, the loss begins from the assistant response.

Conceptually:

```text
<|user|>
Question
<|assistant|>
Answer
```

The loss is calculated primarily on:

```text
Answer
```

rather than the entire sequence.

The implementation tracks:

```cpp
vector<int> sftFrom;
```

to determine where the supervised loss should begin.

This is my first step toward turning the language model into a conversational model.

---

# Current Architecture

The current experimental configuration is approximately:

| Component | Configuration |
|---|---:|
| Language | C++ |
| Model dimension | 384 |
| Attention dimension | 384 |
| Attention heads | 6 |
| Head dimension | 64 |
| Transformer blocks | 8 |
| FFN dimension | 1536 |
| Maximum sequence length | 256 |
| Optimizer | Adam |
| Gradient clipping | 1.0 |
| Attention | Causal Multi-Head Attention |
| Normalization | LayerNorm |
| Tokenizer | BPE |
| Training objective | Next-token prediction |
| Fine-tuning | SFT |

These values are experimental and may change as the project develops.

---

# Model Architecture

```text
                         Input Text
                             │
                             ▼
                          BPE Tokenizer
                             │
                             ▼
                          Token IDs
                             │
                  ┌──────────┴──────────┐
                  │                     │
                  ▼                     ▼
            Token Embedding      Position Embedding
                  │                     │
                  └──────────┬──────────┘
                             │
                             ▼
                    Transformer Block × 8
                             │
          ┌──────────────────┴──────────────────┐
          │                                     │
          ▼                                     │
      LayerNorm                                 │
          │                                     │
          ▼                                     │
 Multi-Head Self-Attention                     │
          │                                     │
          ▼                                     │
      Linear Wo                                │
          │                                     │
          └──────────── Residual ───────────────┘
                             │
                             ▼
                         LayerNorm
                             │
                             ▼
                           FFN
                             │
                             ▼
                         Residual
                             │
                             ▼
                       Repeat × 8
                             │
                             ▼
                       Output Linear
                             │
                             ▼
                          Softmax
                             │
                             ▼
                      Next Token
```

---

# What I Implemented Myself

The goal of this project is to minimize dependence on high-level ML frameworks.

### Linear Algebra

- [x] Matrix multiplication
- [x] Matrix transpose
- [x] Matrix addition
- [x] Matrix concatenation

### Tokenization

- [x] UTF-8 splitting
- [x] Vocabulary loading
- [x] BPE merge loading
- [x] BPE encoding
- [x] Special token handling

### Neural Network

- [x] Embedding lookup
- [x] Linear layers
- [x] ReLU
- [x] Softmax
- [x] Cross-entropy loss
- [x] LayerNorm
- [x] Residual connections

### Transformer

- [x] Q/K/V projection
- [x] Scaled dot-product attention
- [x] Causal masking
- [x] Multi-head attention
- [x] Feed-forward network
- [x] Transformer blocks
- [x] Stacking multiple blocks

### Backpropagation

- [x] Matrix multiplication backward
- [x] Softmax + cross entropy backward
- [x] Embedding backward
- [x] LayerNorm backward
- [x] Attention backward
- [x] FFN backward
- [x] Residual backward

### Optimization

- [x] Gradient descent
- [x] Adam
- [x] Gradient clipping
- [x] Cosine learning-rate decay

### Training

- [x] Random training windows
- [x] Autoregressive next-token prediction
- [x] Model checkpoint saving
- [x] Loss recording
- [x] Training resume
- [x] Supervised fine-tuning experiments

---

# Why C++?

Using C++ makes the project significantly more difficult than using a modern deep-learning framework.

That is intentional.

For example, instead of:

```python
loss.backward()
```

I have to explicitly implement the chain of derivatives.

Instead of:

```python
torch.matmul(A, B)
```

I implemented matrix multiplication manually.

Instead of relying on an automatic differentiation engine, the project explicitly stores intermediate values and propagates gradients through every operation.

This makes the implementation slower and less convenient, but much more educational.

---

# What I Learned

The most important thing I learned is that a Transformer is not magic.

At a fundamental level, it is a composition of relatively understandable operations:

```text
Matrix Multiplication
        +
Normalization
        +
Softmax
        +
Nonlinearity
        +
Residual Connections
        +
Gradient Descent
```

The difficult part is not any single equation.

The difficult part is making **all of them work together correctly**.

A small mistake in:

```text
shape
gradient
mask
normalization
index
tokenization
```

can make the entire model fail to learn.

---

# Important Lessons

### 1. Forward propagation is only half of the model

Implementing the forward pass is relatively straightforward.

The real challenge begins when trying to train the model.

---

### 2. Shapes matter everywhere

For example:

```text
X      : [sequence_length, d]
WQ     : [d, dk]
Q      : [sequence_length, dk]
```

After splitting into 6 heads:

```text
Q_head : [sequence_length, 64]
```

A single incorrect dimension can break the entire computation.

---

### 3. Residual connections make the computational graph more complicated

A residual connection:

```text
Y = X + F(X)
```

means that gradients must flow through both paths:

```text
          ┌──────────────┐
          │              ▼
X ────────┼───────────► Add ───► Y
│         │              ▲
│         └──► F(X) ─────┘
│
└──────────────── gradient path
```

This became much more obvious after implementing the backward pass manually.

---

### 4. Attention is mostly matrix operations

The famous Transformer attention mechanism:

```text
softmax(QKᵀ / √dₖ)V
```

looks complicated when first encountered.

After implementing it from scratch, it becomes much more intuitive.

---

### 5. A language model is fundamentally a next-token predictor

At its core, the model repeatedly learns:

```text
P(token_t | token_1, token_2, ..., token_{t-1})
```

The apparent complexity of a language model emerges from learning this distribution over a huge number of contexts.

---

# Current Limitations

This project is still experimental.

It is **not intended to compete with modern production LLMs**.

Current limitations include:

- Pure C++ matrix implementation is relatively slow.
- `vector<vector<float>>` is not an efficient tensor representation.
- No GPU acceleration in the current training implementation.
- No optimized BLAS implementation.
- No mixed precision.
- No Flash Attention.
- No KV cache for inference.
- Limited dataset size.
- Limited context length.
- Relatively small model capacity.
- Training stability still requires experimentation.
- The tokenizer implementation is intentionally simplified.
- The architecture does not attempt to reproduce every detail of modern LLM implementations.

These limitations are part of the learning process.

---

# Future Plans

## Short Term

- [ ] Improve matrix/tensor memory layout
- [ ] Add faster matrix multiplication
- [ ] Add inference / text generation
- [ ] Implement temperature sampling
- [ ] Implement top-k sampling
- [ ] Implement top-p sampling
- [ ] Add validation loss
- [ ] Improve checkpoint management

## Performance

- [x] OpenMP acceleration
- [ ] SIMD optimization
- [x] CUDA implementation
- [x] Metal implementation
- [ ] GPU matrix multiplication
- [ ] Better memory management

## Model

- [ ] GELU / SwiGLU
- [ ] RoPE
- [ ] KV Cache
- [ ] RMSNorm
- [ ] More efficient attention
- [ ] Larger context length
- [ ] Larger vocabulary

## Training

- [ ] Better dataset pipeline
- [ ] Validation and evaluation
- [ ] Learning-rate warmup
- [ ] Weight decay
- [ ] More robust checkpoint recovery
- [ ] Larger-scale pretraining
- [ ] Better SFT dataset

---

# Project Philosophy

This repository is primarily a **learning project**.

I am not trying to reproduce a billion-parameter model on a laptop.

Instead, I want to understand the fundamental mechanisms behind modern neural networks by implementing them myself.

The philosophy is:

> **Don't just use the framework. Understand what the framework is doing.**

And eventually:

> **If I cannot explain the gradient, I probably don't fully understand the layer.**

---

# From CNN to Transformer

The most meaningful part of this project is not the final code.

It is the progression:

```text
"I want to understand neural networks."
             ↓
"I'll implement a CNN."
             ↓
"I need to understand backpropagation."
             ↓
"I'll derive the gradients myself."
             ↓
"I want to understand language models."
             ↓
"I'll implement tokenization."
             ↓
"I'll implement attention."
             ↓
"I'll build a Transformer."
             ↓
"I'll implement the entire backward pass."
             ↓
"I'll train it myself."
             ↓
"Now I want to understand how an LLM is fine-tuned."
```

This repository is essentially a record of that process.

---

# Disclaimer

This project is experimental and educational.

The implementation is intentionally simple and prioritizes **understanding over performance**.

It should not be considered a production-ready Transformer implementation.

---

# License

This project is released for educational and experimental purposes.

<div align="center">

[🇨🇳 中文](#中文) | [🇺🇸 English](#English) | [Examples](#result) | [How to use](#use)

</div>

<a id="result"></a>

# The training and SFT data used wikiCN2023 and baike2018qa in Chinese language. These are some high quality conversation examples.
# 训练数据使用了wikiCN2023与baike2018qa，以下是一些高质量示例对话。

> 发烧怎么办？

您好：一般来说，由于病情轻重不一、有异物感冒引起的咳嗽和支气管炎等疾病的发生。建议可以用药中西医结合治疗（如百合片）方可治愈；同时注意饮食清淡易消化吸收及外阴干湿润处

(translate)
> What to do for a fever?
> 
Hello: Generally speaking, depending on the severity of the illness, caused by foreign body sensation, colds, cough, and bronchitis may occur. It is recommended to use a combination of traditional Chinese and Western medicine (such as Baihe tablets) for treatment to cure it; at the same time, pay attention to a light and easily digestible diet and keep the external genital area appropriately moist.


> 什么是梦想？

“梦想”，就是实现自己与目标的动力。这是一种理想、有价值和高效发展的前提；我们需要建立一个能让自己的幸福更快乐地成长为未来的人生目标，而不是为了生存而活着。

(translate)
> What is a dream?

"Dreams" are the motivation to achieve yourself and your goals. It's a premise for having ideals, value, and efficient growth; we need to create life goals that let our happiness grow more joyfully for the future, instead of just living to survive.

<div align="center">

[🇨🇳 中文](#中文) | [🇺🇸 English](#English) | [Examples](#result) | [How to use](#use)

</div>

# 中文

<a id="中文"></a>

# 从零开始用 C++ 实现 Transformer

> **通过从零实现机器学习算法来学习机器学习。**

这个仓库记录了我学习机器学习和深度学习的过程。

我尝试使用 **C++** 亲自实现底层算法，而不是依赖高级深度学习框架。

这个项目最初只是从简单的矩阵运算和神经网络开始，随后逐渐发展成了一个基于 Transformer 的小型语言模型，目前包含：

- BPE 分词
- Token Embedding（词元嵌入）
- Positional Embedding（位置嵌入）
- Multi-Head Self-Attention（多头自注意力）
- Causal Attention Mask（因果注意力掩码）
- Layer Normalization（层归一化）
- Residual Connections（残差连接）
- Feed-Forward Networks（前馈神经网络）
- Backpropagation（反向传播）
- Adam Optimizer（Adam 优化器）
- Gradient Clipping（梯度裁剪）
- Learning Rate Scheduling（学习率调度）
- Autoregressive Language Modeling（自回归语言建模）
- Supervised Fine-Tuning（监督微调，SFT）

这个项目的主要目的**不是构建一个具有竞争力的大语言模型**，而是理解一个语言模型内部究竟发生了什么。

---

# 项目动机

我不想仅仅通过调用：

```python
model.fit(...)
```

或者：

```python
optimizer.step()
```

这样的 API 来学习机器学习。

我想知道这些操作背后究竟意味着什么。

所以，我决定自己实现其中的重要组件。

与其把神经网络当作一个黑盒，我更希望能够回答这些问题：

- 一次 Forward Pass（前向传播）究竟发生了什么？
- 梯度究竟是从哪里来的？
- 反向传播是如何穿过矩阵乘法的？
- Attention 究竟是如何工作的？
- 为什么 LayerNorm 需要反向传播？
- Adam 是如何更新每一个参数的？
- 语言模型是如何学习预测下一个 Token 的？
- 一个预训练语言模型进行 Fine-tuning（微调）时究竟发生了什么变化？

这个仓库就是这个学习过程的结果。

---

# 我的学习历程

这个项目并不是一开始就有 Transformer。

它是一步一步发展起来的。

```text
C++
 │
 ├── 矩阵运算
 │
 ├── 神经网络基础
 │
 ├── 前向传播
 │
 ├── 反向传播
 │
 ├── 从零实现 CNN
 │
 ├── 梯度下降
 │
 ├── Adam 优化器
 │
 ├── Tokenization（分词）
 │
 ├── BPE
 │
 ├── Embedding（嵌入）
 │
 ├── Self-Attention（自注意力）
 │
 ├── Multi-Head Attention（多头注意力）
 │
 ├── Transformer Block
 │
 ├── 完整反向传播
 │
 ├── 语言模型训练
 │
 └── Supervised Fine-Tuning（监督微调）
```

---

# 第一阶段 —— 学习 C++

这个项目有意使用了相对基础的 C++ 数据结构。

例如，矩阵被表示为：

```cpp
typedef vector<vector<float>> Matrix;
```

我没有使用 Eigen 等矩阵库，而是自己实现了矩阵乘法：

```cpp
Matrix matmul(const Matrix &A, const Matrix &B){
    ...
}
```

最终，矩阵乘法成为了整个项目最重要的基础组件之一。

我还实现了：

- 矩阵乘法
- 矩阵转置
- 矩阵加法
- 矩阵拼接
- 文件 I/O
- 随机初始化
- 参数序列化
- 基础数据结构

---

# 第二阶段 —— 从零构建神经网络

在理解 Transformer 之前，我首先需要理解普通神经网络。

最重要的概念是：

> **前向传播 + 损失函数 + 反向传播 + 参数更新**

基本训练流程变成：

```text
输入
 ↓
前向传播
 ↓
预测结果
 ↓
损失函数
 ↓
反向传播
 ↓
梯度
 ↓
优化器
 ↓
更新参数
```

在这个阶段，我开始理解：

**神经网络本质上就是大量参数根据梯度不断进行更新。**

---

# 第三阶段 —— 从零实现 CNN

之后，我完全使用 C++ 实现了一个小型 CNN。

网络结构包括：

```text
输入
 ↓
卷积
 ↓
ReLU
 ↓
最大池化
 ↓
Flatten
 ↓
全连接层
 ↓
Softmax
```

我实现了完整的前向传播和反向传播。

例如，卷积层拥有自己的梯度计算，而最大池化则需要记录最大值所在的位置，从而在反向传播时将梯度正确地传回去。

这个阶段让我认识到一个非常重要的问题：

> **机器学习中真正困难的地方往往不是前向传播，而是正确地推导和实现反向传播。**

---

# 第四阶段 —— 理解反向传播

完成 CNN 之后，我开始手动实现各种导数。

对于矩阵乘法：

```text
C = AB
```

其梯度为：

```text
dA = dC · Bᵀ

dB = Aᵀ · dC
```

最终实现成了：

```cpp
void matmul_backward(
    const Matrix &A,
    const Matrix &B,
    const Matrix &dC,
    Matrix &dA,
    Matrix &dB
)
```

之后，我逐渐实现了以下操作的反向传播：

- 矩阵乘法
- ReLU
- Softmax + Cross Entropy
- 最大池化
- LayerNorm
- Attention
- Embedding Lookup

理解这些导数是整个项目中最重要的部分之一。

---

# 第五阶段 —— 从分类走向语言模型

在理解 CNN 之后，我希望构建一个更接近语言模型的东西。

任务不再是：

> 根据一个输入预测一个类别。

而是：

> 根据之前的 Token，预测下一个 Token。

例如：

```text
The cat is
       ↓
     下一个？
```

模型会对整个词表输出一个概率分布：

```text
cat       0.01
dog       0.03
sleeping  0.20
...
```

训练目标就是预测下一个 Token。

损失函数使用交叉熵：

```text
L = -log P(correct_token)
```

---

# 第六阶段 —— Tokenization 与 BPE

语言模型无法直接处理原始文本。

因此，我实现了一个基于 **Byte Pair Encoding（BPE）** 的 Tokenizer。

大致流程如下：

```text
原始文本
   ↓
UTF-8 字符切分
   ↓
边界检测
   ↓
词表查找
   ↓
BPE Merge 操作
   ↓
Token IDs
```

BPE 实现会加载：

```text
bpe_vocab.txt
bpe_merges.txt
```

然后在 Tokenization 过程中重新执行 Merge 流程。

我还加入了对特殊 Token 的支持，例如：

```text
<|assistant|>
<|user|>
...
```

这在之后进行 Supervised Fine-Tuning 时变得非常重要。

---

# 第七阶段 —— Embedding

完成 Tokenization 后，每一个 Token ID 都会被映射成一个向量。

例如：

```text
Token ID
   ↓
Embedding Table
   ↓
384 维向量
```

项目目前使用：

```cpp
int d = 384;
```

输入表示为：

```text
Token Embedding
       +
Position Embedding
       ↓
Transformer Input
```

Embedding Table 本身也会通过反向传播进行训练。

---

# 第八阶段 —— 实现 Self-Attention

这是整个项目中最重要的阶段之一。

Attention 的核心公式是：

```text
Attention(Q, K, V)
=
softmax(QKᵀ / √dₖ)V
```

我在 C++ 中直接实现了这一过程。

整体流程为：

```text
X
│
├── XWQ → Q
├── XWK → K
└── XWV → V
       │
       ↓
     QKᵀ
       │
       ↓
   / √dₖ
       │
       ↓
   Causal Mask
       │
       ↓
    Softmax
       │
       ↓
       × V
       │
       ↓
 Attention Output
```

Causal Mask 可以确保一个 Token 无法看到它后面的 Token。

例如：

```text
1  -∞  -∞  -∞
1   1  -∞  -∞
1   1   1  -∞
1   1   1   1
```

这样模型就可以进行自回归预测。

---

# 第九阶段 —— Multi-Head Attention

单个 Attention 机制的表达能力有限。

因此，我实现了多头注意力。

当前配置为：

```cpp
int d = 384;
int dk = 384;
int h = 6;
```

因此每一个 Head 使用：

```text
384 / 6 = 64 维
```

从概念上来看：

```text
                    ┌── Head 1 ──┐
                    ├── Head 2 ──┤
输入 → Q/K/V ──────┼── Head 3 ──┼──→ Concatenate
                    ├── Head 4 ──┤
                    ├── Head 5 ──┤
                    └── Head 6 ──┘
                              ↓
                             Wo
                              ↓
                            输出
```

不同的 Head 可以学习 Token 之间不同类型的关系。

---

# 第十阶段 —— Transformer Block

完成 Attention 后，我将各个组件组合成了一个 Transformer Block。

目前的 Block 大致如下：

```text
                输入
                  │
                  ▼
             LayerNorm
                  │
                  ▼
        Multi-Head Attention
                  │
                  ▼
        Linear Projection (Wo)
                  │
                  ▼
             残差连接
                  │
                  ▼
             LayerNorm
                  │
                  ▼
              FFN / MLP
                  │
                  ▼
             残差连接
                  │
                  ▼
                输出
```

Feed-Forward Network：

```text
X
 ↓
Wup
 ↓
ReLU
 ↓
Wdown
 ↓
输出
```

实现中使用：

```cpp
int dimension = 4 * d;
```

因此：

```text
d = 384
```

时，隐藏层维度为：

```text
1536
```

---

# 第十一阶段 —— 构建多个 Transformer Layer

一个 Transformer Block 显然是不够的。

目前项目使用：

```cpp
int N = 8;
```

因此整个模型包含 8 个 Transformer Block：

```text
Embedding
   ↓
Transformer Block 1
   ↓
Transformer Block 2
   ↓
Transformer Block 3
   ↓
Transformer Block 4
   ↓
Transformer Block 5
   ↓
Transformer Block 6
   ↓
Transformer Block 7
   ↓
Transformer Block 8
   ↓
Linear Output
   ↓
Softmax
```

模型根据最终的 Hidden Representation 预测下一个 Token。

---

# 第十二阶段 —— 实现 Transformer 反向传播

这可能是整个项目中最困难的部分。

反向传播需要依次穿过：

```text
Loss
 ↓
Softmax
 ↓
Output Projection
 ↓
Residual Connection
 ↓
FFN
 ↓
LayerNorm
 ↓
Residual Connection
 ↓
Attention Output Projection
 ↓
Multi-Head Attention
 ↓
Causal Mask
 ↓
Softmax
 ↓
QKᵀ
 ↓
Q / K / V
 ↓
Embedding
```

对于 Attention，我实现了以下梯度传播过程：

```text
Attention
    ↓
V
    ↓
Softmax
    ↓
Causal Mask
    ↓
Scaling
    ↓
QKᵀ
    ↓
Q / K
    ↓
WQ / WK / WV
```

这是一个非常重要的里程碑。

因为从这一阶段开始，模型不再只是一个 Transformer 的**前向传播模拟器**。

它真正成为了一个可以进行训练的神经网络。

---

# 第十三阶段 —— LayerNorm 反向传播

LayerNorm 也是一个需要手动推导梯度的组件。

前向过程大致为：

```text
μ = mean(X)

σ = sqrt(var(X) + ε)

X̂ = (X - μ) / σ

Y = γX̂ + β
```

实现过程中会在 Forward Pass 保存：

```cpp
Xhat
sigma
```

等中间结果。

然后在 Backpropagation 中重新使用这些值。

因此，项目将：

```text
BlockParams
BlockCache
BlockParamGrads
```

分开管理。

这样可以分别保存：

- 参数
- 前向传播中的中间值
- 梯度

---

# 第十四阶段 —— Adam 优化器

最开始，为了理解优化过程，我使用的是简单的梯度下降。

之后，我实现了 Adam。

Adam 会维护：

```text
m = 一阶矩
v = 二阶矩
```

并进行偏差修正：

```text
m̂ = m / (1 - β₁ᵗ)

v̂ = v / (1 - β₂ᵗ)
```

最终参数更新公式为：

```text
θ ← θ - α · m̂ / (√v̂ + ε)
```

实现中会为每一个参数矩阵单独保存优化器状态。

例如：

```cpp
BlockAdamState
```

会保存以下参数的优化器状态：

```text
WQ
WK
WV
Wo
gamma
beta
gamma2
beta2
Wup
Wdown
```

---

# 第十五阶段 —— Gradient Clipping

在训练更深的网络时，梯度可能会变得不稳定。

因此，我加入了梯度裁剪：

```cpp
void clip(Matrix &grad, float limit)
```

目前使用：

```cpp
float clipLimit = 1.0f;
```

每个梯度元素都会被限制在：

```text
[-1, 1]
```

范围内。

这是一个简单但非常实用的防止梯度过大的方法。

---

# 第十六阶段 —— Learning Rate Scheduling

学习率并不是始终保持不变。

我实现了 Cosine Decay：

```text
lr(e)
=
lr_min
+
(lr_max - lr_min)
×
0.5(1 + cos(πe / E))
```

这样可以让学习率随着训练逐渐降低。

例如，目前 SFT 配置大约使用：

```text
最大学习率 = 2e-5
最小学习率 = 2e-6
```

---

# 第十七阶段 —— Supervised Fine-Tuning

在完成普通的 Next-Token Language Modeling 后，我开始尝试 Supervised Fine-Tuning（监督微调）。

训练数据围绕一些特殊 Token 进行组织，例如：

```text
<|assistant|>
```

在 SFT 中，模型并不需要对所有 Token 进行同等程度的学习。

例如：

```text
<|user|>
Question
<|assistant|>
Answer
```

Loss 主要计算：

```text
Answer
```

而不是整个序列。

项目中使用：

```cpp
vector<int> sftFrom;
```

来确定监督 Loss 应该从哪里开始计算。

这是我让语言模型逐渐具备对话能力的第一步。

---

# 当前模型架构

目前的实验配置大约如下：

| 组件 | 配置 |
|---|---:|
| 编程语言 | C++ |
| 模型维度 | 384 |
| Attention 维度 | 384 |
| Attention Heads | 6 |
| Head Dimension | 64 |
| Transformer Blocks | 8 |
| FFN Dimension | 1536 |
| 最大序列长度 | 256 |
| 优化器 | Adam |
| 梯度裁剪 | 1.0 |
| Attention | Causal Multi-Head Attention |
| Normalization | LayerNorm |
| Tokenizer | BPE |
| 训练目标 | Next-Token Prediction |
| 微调方式 | SFT |

这些参数仍处于实验阶段，并且可能会随着项目发展而发生变化。

---

# 模型架构

```text
                         输入文本
                            │
                            ▼
                         BPE Tokenizer
                            │
                            ▼
                          Token IDs
                            │
                 ┌──────────┴──────────┐
                 │                     │
                 ▼                     ▼
            Token Embedding      Position Embedding
                 │                     │
                 └──────────┬──────────┘
                            │
                            ▼
                   Transformer Block × 8
                            │
         ┌──────────────────┴──────────────────┐
         │                                     │
         ▼                                     │
      LayerNorm                                │
         │                                     │
         ▼                                     │
 Multi-Head Self-Attention                    │
         │                                     │
         ▼                                     │
      Linear Wo                               │
         │                                     │
         └──────────── 残差连接 ───────────────┘
                            │
                            ▼
                        LayerNorm
                            │
                            ▼
                           FFN
                            │
                            ▼
                          Residual
                            │
                            ▼
                        Repeat × 8
                            │
                            ▼
                      Output Linear
                            │
                            ▼
                         Softmax
                            │
                            ▼
                        下一个 Token
```

---

# 我亲自实现的内容

这个项目的目标是尽可能减少对高级机器学习框架的依赖。

### 线性代数

- [x] 矩阵乘法
- [x] 矩阵转置
- [x] 矩阵加法
- [x] 矩阵拼接

### Tokenization

- [x] UTF-8 切分
- [x] 词表加载
- [x] BPE Merge 加载
- [x] BPE 编码
- [x] 特殊 Token 处理

### 神经网络

- [x] Embedding Lookup
- [x] Linear Layers
- [x] ReLU
- [x] Softmax
- [x] Cross-Entropy Loss
- [x] LayerNorm
- [x] Residual Connections

### Transformer

- [x] Q/K/V Projection
- [x] Scaled Dot-Product Attention
- [x] Causal Masking
- [x] Multi-Head Attention
- [x] Feed-Forward Network
- [x] Transformer Blocks
- [x] 多个 Transformer Block 堆叠

### 反向传播

- [x] 矩阵乘法反向传播
- [x] Softmax + Cross Entropy 反向传播
- [x] Embedding 反向传播
- [x] LayerNorm 反向传播
- [x] Attention 反向传播
- [x] FFN 反向传播
- [x] Residual 反向传播

### 优化

- [x] Gradient Descent
- [x] Adam
- [x] Gradient Clipping
- [x] Cosine Learning-Rate Decay

### 训练

- [x] 随机训练窗口
- [x] 自回归 Next-Token Prediction
- [x] 模型 Checkpoint 保存
- [x] Loss 记录
- [x] 训练恢复
- [x] Supervised Fine-Tuning 实验

---

# 为什么选择 C++？

使用 C++ 会让这个项目比使用现代深度学习框架困难很多。

这是故意的。

例如，与其直接使用：

```python
loss.backward()
```

我必须自己明确实现整个导数传播过程。

与其使用：

```python
torch.matmul(A, B)
```

我自己实现了矩阵乘法。

与其依赖自动微分引擎，项目会明确保存 Forward Pass 中的中间结果，并让梯度穿过每一个操作。

这使得实现速度更慢、使用起来也更加麻烦。

但与此同时，它也更加具有学习价值。

---

# 我学到的东西

这个项目让我认识到，Transformer 并没有想象中的那么神秘。

从最基础的角度来看，它其实是由一些相对容易理解的操作组合起来的：

```text
矩阵乘法
    +
归一化
    +
Softmax
    +
非线性函数
    +
残差连接
    +
梯度下降
```

真正困难的并不是某一个单独的公式。

真正困难的是让**所有这些组件正确地组合在一起**。

一个很小的错误，例如：

```text
shape
gradient
mask
normalization
index
tokenization
```

都可能导致整个模型无法学习。

---

# 重要的学习经验

### 1. Forward Propagation 只是模型的一半

实现前向传播相对容易。

真正的挑战从训练模型开始。

---

### 2. Shape 在任何地方都非常重要

例如：

```text
X      : [sequence_length, d]
WQ     : [d, dk]
Q      : [sequence_length, dk]
```

拆分成 6 个 Head 后：

```text
Q_head : [sequence_length, 64]
```

一个错误的维度就可能破坏整个计算过程。

---

### 3. Residual Connection 会让计算图变得更加复杂

一个残差连接：

```text
Y = X + F(X)
```

意味着梯度必须沿着两条路径传播：

```text
          ┌──────────────┐
          │              ▼
X ────────┼───────────► Add ───► Y
│         │              ▲
│         └──► F(X) ─────┘
│
└──────────────── 梯度路径
```

在手动实现反向传播之后，我对这一点有了更加直观的理解。

---

### 4. Attention 本质上主要是矩阵运算

Transformer 中著名的 Attention：

```text
softmax(QKᵀ / √dₖ)V
```

第一次接触时看起来可能非常复杂。

但真正从零实现之后，它会变得直观很多。

---

### 5. 语言模型本质上是一个 Next-Token Predictor

从根本上来说，模型不断学习：

```text
P(token_t | token_1, token_2, ..., token_{t-1})
```

语言模型看起来非常复杂，但其核心其实就是在大量上下文中学习这个概率分布。

---

# 当前限制

这个项目仍然处于实验阶段。

它**并不是为了与现代生产级大语言模型竞争**。

目前存在以下限制：

- 纯 C++ 矩阵实现速度相对较慢
- `vector<vector<float>>` 并不是高效的 Tensor 数据结构
- 当前训练实现没有 GPU 加速
- 没有使用优化过的 BLAS
- 没有 Mixed Precision
- 没有 Flash Attention
- 推理阶段没有 KV Cache
- 数据集规模有限
- Context Length 有限
- 模型容量相对较小
- 训练稳定性仍需要进一步实验
- Tokenizer 实现进行了有意的简化
- 当前架构没有尝试复现现代 LLM 的所有细节

这些限制本身也是学习过程的一部分。

---

# 未来计划

## 短期计划

- [ ] 改进矩阵 / Tensor 内存布局
- [ ] 加速矩阵乘法
- [ ] 添加推理 / 文本生成
- [ ] 实现 Temperature Sampling
- [ ] 实现 Top-K Sampling
- [ ] 实现 Top-P Sampling
- [ ] 添加 Validation Loss
- [ ] 改进 Checkpoint 管理

## 性能优化

- [x] OpenMP 加速
- [ ] SIMD 优化
- [x] CUDA 实现
- [x] Metal 实现
- [ ] GPU 矩阵乘法
- [ ] 更好的内存管理

## 模型

- [ ] GELU / SwiGLU
- [ ] RoPE
- [ ] KV Cache
- [ ] RMSNorm
- [ ] 更高效的 Attention
- [ ] 更长的 Context Length
- [ ] 更大的 Vocabulary

## 训练

- [ ] 更好的数据集 Pipeline
- [ ] Validation 和 Evaluation
- [ ] Learning-Rate Warmup
- [ ] Weight Decay
- [ ] 更可靠的 Checkpoint 恢复
- [ ] 更大规模的预训练
- [ ] 更高质量的 SFT 数据集

---

# 项目理念

这个仓库本质上是一个**学习型项目**。

我并不打算在一台普通电脑上复现一个拥有数十亿参数的模型。

相反，我希望通过亲自实现这些组件，真正理解现代神经网络背后的基本机制。

我的理念是：

> **不要只会使用框架，要理解框架究竟在做什么。**

以及：

> **如果我无法解释一个梯度是怎么来的，那么我可能还没有真正理解这一层。**

---

# 从 CNN 到 Transformer

这个项目最有意义的地方并不是最后的代码。

而是整个过程：

```text
“我想理解神经网络。”
             ↓
“那我自己实现一个 CNN。”
             ↓
“我需要理解反向传播。”
             ↓
“那我自己推导梯度。”
             ↓
“我想理解语言模型。”
             ↓
“那我自己实现 Tokenization。”
             ↓
“我自己实现 Attention。”
             ↓
“我自己构建 Transformer。”
             ↓
“我自己实现完整的反向传播。”
             ↓
“我自己训练它。”
             ↓
“现在我想理解 LLM 是如何进行 Fine-Tuning 的。”
```

这个仓库，本质上就是对这一整个过程的记录。

---

# 声明

这个项目是一个实验性、教育性的项目。

实现过程中会刻意保持简单，并且优先考虑**理解，而不是性能**。

因此，它不应该被视为一个可以直接用于生产环境的 Transformer 实现。

---

# License

本项目仅用于学习和实验目的。

<div align="center">

[🇨🇳 中文](#中文) | [🇺🇸 English](#English) | [Examples](#result) | [How to use](#use)

</div>

<a id="use"></a>

# How to use?
## For user
You should unpack the model or put your own 'train' folder in (usr) folder(Mac) or in the same path with the main program(Windows).
To start it, please using USR_Sweden_sft

When you are in the USR_Sweden_sft, there are several code you need to know:
> #change

For change the pre-setting prompt(front, opposite).
> #temp

For change the temperature of predicting next word.
> #maxlength

For change the maximum generation of tokens.
> #debug

For toggle the debug mode.
> #mem

For toggle the memorizing contexts mode.

# How to train?
## For trainer
1. You should unpack the 'Training Package for xxx' in an appropriate folder.
2. You should create a folder named as 'train'. For Mac user, you'd better create this folder under the (usr) folder.
3. You should put the original training text in the 'train' folder and name it as 'train.txt'.
4. Create a file named as 'para.txt', including total epochs, d, dk, heads quantity, layers quantity, max sequence length. Separating it in different lines but in ordered. (For recommend, 1000000 384 384 6 8 256)
5. Run the program 'genVocab', in order to generate tokens. (genVocab2 has the same function as genVocab but much more faster)
6.  Run the program 'genNew', in order to generate initial parameters.
7.  To start training, run the program 'TSFM_Sweden'. (TSFM_Sweden2 has the same function as TSFM_Sweden but much more faster)
8.  After the training, you can run the program 'TSFM_Sweden_sft' to slight adjust the model. Before you start it, you should create a file named as 'sft.txt' and put the training text in it.
