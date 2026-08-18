<p align="center">
  <a href="README.md">🇺🇸 English</a>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="README_CN.md">🇨🇳 简体中文</a>
</p>

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

Before working on machine learning, I spent a lot of time learning C++ itself.

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

- [ ] OpenMP acceleration
- [ ] SIMD optimization
- [ ] CUDA implementation
- [ ] Metal implementation
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
