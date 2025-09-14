# TensorBrain  

TensorBrain is a deep learning framework built from scratch for systems research. It provides a minimal set of tools for defining tensors, building neural networks, training them with autograd, and scaling across devices with distributed training.  

---

## Features  

- **Autograd Engine** — automatic differentiation with a dynamic computation graph.  
![Computation graph](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*1X9lVngy3H9tXz0kscT5Kg.png)  

- **Multi-Dimensional Tensor Operations** — broadcasting and matrix multiplication across N-D tensors.  
![Tensor illustration](https://github.com/pytorch/pytorch/blob/9708fcf92db88b80b9010c68662d634434da3106/docs/source/_static/img/tensor_illustration.png)  

- **Neural Network Modules** — Linear, Conv2D, Embedding, and Transformer blocks.  
![Transformer block](https://jalammar.github.io/images/t/transformer_block_diagram.png)  

- **Distributed Training** — Data Parallel (DDP) and Pipeline Parallel (1F1B scheduling).  
![DDP Illustration](https://pytorch.org/tutorials/_images/ddp.png)  

- **Graph Compiler** — intermediate representation (IR) with constant folding and op fusion.  
![IR diagram](https://raw.githubusercontent.com/onnx/tutorials/main/images/onnx_graph.png)  

- **Quantization** — post-training INT8 quantization for faster inference.  

- **Serving Runtime** — compile → deploy → serve with FastAPI, benchmarked for low latency.  
![Serving diagram](https://miro.medium.com/v2/resize:fit:1200/format:webp/1*FZ6IR7Hj0B0DQ0b7T9HJgA.png)  

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

