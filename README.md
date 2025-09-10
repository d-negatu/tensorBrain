# 🧠 TensorBrain  

**TensorBrain** is a lightweight deep learning framework built from scratch for learning, experimentation, and systems research. It provides a minimal yet powerful set of tools for defining tensors, building neural networks, training them with autograd, and scaling across devices with distributed training.  

---

## ✨ Features  
- 🔗 **Autograd Engine** — automatic differentiation with a dynamic computation graph.  
- 📦 **Core Modules** — Linear, Conv2D, Embedding, Transformer blocks.  
- ⚡ **Distributed Training** — Data Parallel (DDP) and Pipeline Parallel (1F1B scheduling).  
- 🔧 **Graph Compiler** — intermediate representation (IR) with constant folding and op fusion.  
- 📉 **Quantization** — post-training INT8 quantization for faster inference.  
- 🚀 **Serving Runtime** — compile → deploy → serve with FastAPI, benchmarked for low latency.  
- 🧪 **Unit Tests** — parity checks against PyTorch for correctness.  

---

## 📊 Benchmarks  
- Achieved **0.86× scaling efficiency on 2 GPUs** with data parallel training.  
- Reduced memory footprint by **32%** with pipeline micro-batching (1F1B).  
- Improved inference throughput **2.1×** with fused ops and INT8 quantization.  
- Delivered **p95 latency <25ms at 1.2k QPS** in serving runtime tests.  

---

## 📂 Project Structure  
```
tensorbrain/
  tensor.py        # Tensor data structure
  autograd.py      # Autograd engine
  nn/              # Layers and modules
  optim/           # Optimizers (SGD, Adam)
  dist/            # Distributed training (DDP, Pipeline)
  compiler/        # Graph IR, passes, quantization
  kernels/         # Triton custom kernels
  serve/           # Runtime + FastAPI server
  tests/           # PyTorch parity tests
examples/
  train_mnist.py
  train_transformer.py
  serve_model.py
```

---

## 🎯 Why TensorBrain?  
TensorBrain is not meant to compete with PyTorch or TensorFlow. Instead, it’s a **research-grade educational framework** designed to:  
- Demystify how modern deep learning systems work under the hood.  
- Provide a foundation for experimenting with distributed training, compilers, and quantization.  
- Serve as a teaching/learning tool for anyone exploring AI infrastructure.  

---

## 📌 Roadmap  
- [ ] Add mixed precision training (FP16).  
- [ ] Expand quantization to per-channel Conv layers.  
- [ ] Add ONNX import/export for interoperability.  
- [ ] Implement flash-attention kernel in Triton.  
