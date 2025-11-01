# Quantization & Optimization Guide

## Overview

This guide covers quantization techniques and optimization strategies for efficient model inference and deployment.

## Quantization Basics

Quantization reduces model size and speeds up inference by representing weights and activations with fewer bits.

### Post-Training Quantization (PTQ)

```python
import tensorbrain as tb
from tb.quantization import quantize

# Load pretrained model
model = tb.models.resnet50(pretrained=True)
model.eval()

# Post-training quantization
quantized_model = quantize(
    model,
    qconfig_spec='fbgemm',  # or 'qnnpack'
    inplace=False
)

# Measure inference speed
import time
start = time.time()
for _ in range(100):
    output = quantized_model(input_tensor)
inference_time = time.time() - start
print(f"Inference time: {inference_time:.2f}s")
```

### Quantization-Aware Training (QAT)

```python
from tb.quantization import QuantStub, DeQuantStub, prepare_qat

class QuantizedModel(tb.nn.Module):
    def __init__(self):
        super(QuantizedModel, self).__init__()
        self.quant = QuantStub()
        self.conv = tb.nn.Conv2d(3, 64, 3, padding=1)
        self.dequant = DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.dequant(x)
        return x

# Prepare model for QAT
model = QuantizedModel()
model.train()
model = prepare_qat(model)

# Training loop with QAT
criterion = tb.nn.CrossEntropyLoss()
optimizer = tb.optim.Adam(model.parameters())

for epoch in range(num_epochs):
    for batch_data, batch_labels in train_loader:
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Calibrate and update observers
    model.apply(tb.quantization.enable_observer)
    model.apply(tb.quantization.disable_observer)

# Convert to quantized model
quantized_model = tb.quantization.convert(model)
```

## Pruning

### Structured Pruning

```python
from tb.nn.utils.prune import prune_module_structured

# Prune 30% of channels
prune_module_structured(
    model.conv1,
    pruning_method='magnitude',
    amount=0.3
)

# Remove pruned parameters
prune_module_structured.remove(model.conv1)
```

### Unstructured Pruning

```python
from tb.nn.utils.prune import prune_module

# Prune 50% of weights
prune_module(model.fc, 'weight', pruning_method='magnitude', amount=0.5)

# Iterative pruning
for epoch in range(num_epochs):
    # Training
    train(model, train_loader)
    
    # Pruning
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            prune_module(module, 'weight', amount=0.01)
```

## Knowledge Distillation

```python
class DistillationLoss(tb.nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = tb.nn.KLDivLoss(reduction='batchmean')
        self.ce = tb.nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, target):
        # Distillation loss
        soft_targets = tb.nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        soft_pred = tb.nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        distillation_loss = self.kl_div(soft_pred, soft_targets)
        
        # Cross-entropy loss
        ce_loss = self.ce(student_logits, target)
        
        # Combined loss
        return self.alpha * distillation_loss + (1 - self.alpha) * ce_loss

# Training with distillation
student_model = tb.models.mobilenet_v2()
teacher_model = tb.models.resnet50(pretrained=True)
teacher_model.eval()

distillation_criterion = DistillationLoss(temperature=4.0, alpha=0.7)
optimizer = tb.optim.Adam(student_model.parameters())

for epoch in range(num_epochs):
    for batch_data, batch_labels in train_loader:
        student_output = student_model(batch_data)
        teacher_output = teacher_model(batch_data)
        
        loss = distillation_criterion(student_output, teacher_output, batch_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Model Optimization Techniques

### Batch Normalization Folding

```python
from tb.nn.utils import fuse_modules

# Fuse conv + bn layers
fused_model = fuse_modules(
    model,
    [['conv1', 'bn1'], ['conv2', 'bn2']]
)
```

### Graph Optimization

```python
from tb.jit import script, optimize_for_inference

# JIT compile model
scripted_model = script(model)
optimized_model = optimize_for_inference(scripted_model)
```

## Benchmarking

```python
import time
import tensorbrain as tb

def benchmark_model(model, input_tensor, num_iterations=100):
    model.eval()
    
    # Warmup
    with tb.no_grad():
        for _ in range(10):
            output = model(input_tensor)
    
    # Benchmark
    tb.cuda.synchronize() if tb.cuda.is_available() else None
    start = time.time()
    
    with tb.no_grad():
        for _ in range(num_iterations):
            output = model(input_tensor)
    
    tb.cuda.synchronize() if tb.cuda.is_available() else None
    end = time.time()
    
    avg_time = (end - start) / num_iterations * 1000  # ms
    throughput = input_tensor.shape[0] * num_iterations / (end - start)
    
    print(f"Average time per iteration: {avg_time:.2f} ms")
    print(f"Throughput: {throughput:.2f} samples/s")
    return avg_time, throughput

# Benchmark original vs quantized
original_time, _ = benchmark_model(model, test_tensor)
quantized_time, _ = benchmark_model(quantized_model, test_tensor)
print(f"Speedup: {original_time/quantized_time:.2f}x")
```

## Memory Optimization

```python
# Gradient checkpointing
from tb.utils.checkpoint import checkpoint

class CheckpointedModel(tb.nn.Module):
    def __init__(self):
        super(CheckpointedModel, self).__init__()
        self.layers = tb.nn.ModuleList([...])
    
    def forward(self, x):
        for layer in self.layers:
            x = checkpoint(layer, x)
        return x

# Model parallel
device0 = 'cuda:0'
device1 = 'cuda:1'

model.layer1.to(device0)
model.layer2.to(device1)

x = x.to(device0)
x = model.layer1(x)
x = x.to(device1)
x = model.layer2(x)
```

## Calibration for PTQ

```python
from tb.quantization import get_quantization_statistics

# Calibration data
calibration_loader = tb.utils.data.DataLoader(
    calibration_dataset,
    batch_size=32,
    shuffle=False
)

# Calibrate quantization parameters
stats = get_quantization_statistics(model, calibration_loader)
```

## Best Practices

1. **Choose QAT for Higher Accuracy**: Use QAT when accuracy is critical
2. **Use PTQ for Simplicity**: PTQ is faster and easier for many applications
3. **Calibration Data**: Use representative data for calibration
4. **Batch Size**: Larger batches during calibration for better statistics
5. **Testing**: Always benchmark before and after quantization

## See Also

- [API Reference](./API_REFERENCE.md)
- [Distributed Training Guide](./DISTRIBUTED_GUIDE.md)
- [Serving Guide](./SERVING_GUIDE.md)
- [Contributing Guide](../CONTRIBUTING.md)
