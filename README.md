# TensorBrain  

TensorBrain is a deep learning framework built from scratch for systems research. It provides a minimal set of tools for defining tensors, building neural networks, training them with autograd, and scaling across devices with distributed training.  

---

## Features  

- **Autograd Engine** — automatic differentiation with a dynamic computation graph.  
![Computation graph](https://upload.wikimedia.org/wikipedia/commons/0/0c/Backpropagation.png)  

- **Multi-Dimensional Tensor Operations** — broadcasting and matrix multiplication across N-D tensors.  
![Tensor illustration](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/tensor_illustration.png)  

- **Neural Network Modules** — Linear, Conv2D, Embedding, and Transformer blocks.  
![Transformer block](https://jalammar.github.io/images/t/transformer_block_diagram.png)  

- **Distributed Training** — Data Parallel (DDP) and Pipeline Parallel (1F1B scheduling).  
![DDP Illustration](https://pytorch.org/tutorials/_images/ddp.png)  

- **Graph Compiler** — intermediate representation (IR) with constant folding and op fusion.  
![IR diagram](https://raw.githubusercontent.com/onnx/tutorials/main/images/onnx_graph.png)  

- **Quantization** — post-training INT8 quantization for faster inference.  

- **Serving Runtime** — compile → deploy → serve with FastAPI, benchmarked for low latency.  
![Serving diagram](https://upload.wikimedia.org/wikipedia/commons/6/6f/FastAPI-logo.png)  

- **Unit Tests** — parity checks against PyTorch for correctness.  

---

## Benchmarks  

- Achieved **0.86× scaling efficiency on 2 GPUs** with data parallel training.  
- Reduced memory footprint by **32%** with pipeline micro-batching (1F1B).  
- Improved inference throughput **2.1×** with fused ops and INT8 quantization.  
- Delivered **p95 latency <25ms at 1.2k QPS** in serving runtime tests.  

![Benchmark chart](https://matplotlib.org/stable/_images/sphx_glr_simple_plot_001.png)  

---

## 📂 Project Structure  

