# Model Serving & Deployment Guide

## Overview

This guide covers deploying TensorBrain models to production environments and serving them efficiently.

## TensorFlow Serving

### Export Model to SavedModel Format

```python
import tensorbrain as tb

model = tb.models.resnet50(pretrained=True)
model.eval()

# Export to SavedModel
model_path = '/tmp/resnet50_savedmodel'
model.save(model_path, format='savedmodel')

print(f"Model saved to {model_path}")
```

### Start TensorFlow Serving

```bash
# Install TensorFlow Serving
sudo apt-get update && sudo apt-get install -y tensorflow-model-server

# Start serving
tensorflow_model_server \
  --rest_api_port=8501 \
  --model_name=resnet50 \
  --model_base_path=/tmp/resnet50_savedmodel
```

### Client Request

```python
import requests
import json
import numpy as np

# Prepare input
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Send request
response = requests.post(
    'http://localhost:8501/v1/models/resnet50:predict',
    data=json.dumps({
        'instances': input_data.tolist()
    }),
    headers={'Content-Type': 'application/json'}
)

if response.status_code == 200:
    predictions = response.json()['predictions']
    print(f"Predictions: {predictions}")
else:
    print(f"Error: {response.status_code}")
```

## Docker Deployment

### Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and app
COPY model_server.py .
COPY models /app/models

# Expose port
EXPOSE 8000

# Start server
CMD ["python", "model_server.py"]
```

### Build and Run

```bash
# Build image
docker build -t tensorbrain-server:latest .

# Run container
docker run -p 8000:8000 -v /path/to/models:/app/models tensorbrain-server:latest
```

## Flask API Server

```python
from flask import Flask, request, jsonify
import tensorbrain as tb
import numpy as np

app = Flask(__name__)
model = tb.models.resnet50(pretrained=True)
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.get_json()
        input_tensor = tb.tensor(data['input'])
        
        # Run inference
        with tb.no_grad():
            output = model(input_tensor)
        
        # Return predictions
        return jsonify({
            'predictions': output.cpu().numpy().tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
```

## FastAPI Server

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorbrain as tb
import numpy as np

app = FastAPI()
model = tb.models.resnet50(pretrained=True)
model.eval()

class PredictionRequest(BaseModel):
    input: list

class PredictionResponse(BaseModel):
    predictions: list

@app.post('/predict', response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        input_tensor = tb.tensor(request.input)
        with tb.no_grad():
            output = model(input_tensor)
        
        return PredictionResponse(
            predictions=output.cpu().numpy().tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get('/health')
async def health():
    return {'status': 'healthy'}
```

## TorchServe

### Create Model Handler

```python
from ts.torch_handler.base_handler import BaseHandler
import tensorbrain as tb

class ModelHandler(BaseHandler):
    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties
        model_path = properties.get('model_dir')
        
        self.model = tb.models.resnet50(pretrained=True)
        self.model.load_state_dict(
            tb.load(f'{model_path}/model.pth')
        )
        self.model.eval()
    
    def preprocess(self, data):
        input_tensor = data[0].get('input')
        return tb.tensor(input_tensor)
    
    def inference(self, data):
        with tb.no_grad():
            output = self.model(data)
        return [output]
    
    def postprocess(self, data):
        output = data[0].cpu().numpy().tolist()
        return [{'predictions': output}]
```

### Deploy with TorchServe

```bash
# Package model
torch-model-archiver \
    --model-name=resnet50 \
    --version=1.0 \
    --model-file=model.py \
    --serialized-file=model.pth \
    --handler=handler.py \
    --export-path=model_store

# Start TorchServe
torchserve --start --model-store=model_store --ncs --models=resnet50.mar
```

## Cloud Deployment

### AWS SageMaker

```python
import sagemaker
import tensorbrain as tb

# Upload model to S3
model.save('model.tar.gz')
s3_model_uri = sagemaker.Session().upload_data(
    'model.tar.gz',
    bucket='my-bucket',
    key_prefix='models'
)

# Deploy endpoint
from sagemaker.tensorbrain.model import TensorBrainModel

tb_model = TensorBrainModel(
    model_data=s3_model_uri,
    role=sagemaker.get_execution_role(),
    framework_version='1.0',
    py_version='py39'
)

predictor = tb_model.deploy(
    initial_instance_count=1,
    instance_type='ml.p3.2xlarge'
)

# Make predictions
result = predictor.predict([1, 2, 3, 4])
print(result)
```

### Google Cloud AI Platform

```bash
# Upload model
gsutil cp model.tar.gz gs://my-bucket/models/

# Create model version
gcloud ai-platform models create resnet50
gcloud ai-platform versions create v1 \
    --model=resnet50 \
    --origin=gs://my-bucket/models/model.tar.gz \
    --runtime-version=2.8 \
    --framework=tensorflow

# Make predictions
gcloud ai-platform predict \
    --model=resnet50 \
    --version=v1 \
    --json-instances=input.json
```

## Performance Optimization

### Model Batching

```python
class BatchedModel:
    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size
        self.queue = []
        self.results = {}
    
    def predict(self, input_id, input_data):
        self.queue.append((input_id, input_data))
        
        if len(self.queue) >= self.batch_size:
            return self._process_batch()
        return None
    
    def _process_batch(self):
        ids, inputs = zip(*self.queue)
        batch = tb.stack(inputs)
        
        with tb.no_grad():
            outputs = self.model(batch)
        
        self.queue = []
        return dict(zip(ids, outputs))
```

### Caching

```python
from functools import lru_cache
import hashlib

class CachedModel:
    def __init__(self, model):
        self.model = model
        self.cache = {}
    
    def predict(self, input_data):
        # Create cache key
        key = hashlib.md5(str(input_data).encode()).hexdigest()
        
        if key in self.cache:
            return self.cache[key]
        
        with tb.no_grad():
            output = self.model(tb.tensor(input_data))
        
        self.cache[key] = output.cpu().numpy()
        return self.cache[key]
```

## Monitoring and Logging

```python
import logging
import time
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)
prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

def predict_with_monitoring(model, input_data):
    start_time = time.time()
    
    try:
        output = model(input_data)
        prediction_counter.inc()
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise
    finally:
        latency = time.time() - start_time
        prediction_latency.observe(latency)
        logger.info(f"Prediction latency: {latency:.2f}s")
    
    return output
```

## Best Practices

1. **Model Optimization**: Quantize and compress models before deployment
2. **Batch Processing**: Use batching for better throughput
3. **Load Balancing**: Distribute requests across multiple instances
4. **Monitoring**: Track latency, throughput, and errors
5. **Versioning**: Maintain multiple model versions for A/B testing
6. **Error Handling**: Gracefully handle errors and timeouts

## See Also

- [Quantization Guide](./QUANTIZATION_GUIDE.md)
- [Distributed Training Guide](./DISTRIBUTED_GUIDE.md)
- [API Reference](./API_REFERENCE.md)
- [Contributing Guide](../CONTRIBUTING.md)
