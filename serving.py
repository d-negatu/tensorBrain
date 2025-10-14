#!/usr/bin/env python3
"""
FastAPI Serving Runtime for TensorBrain
Basic model serving with REST API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import time
from typing import List, Dict, Any
import json
import os

from nn import Sequential, Linear, ReLU
from tensor import Tensor


app = FastAPI(title="TensorBrain Serving Runtime", version="1.0.0")

# Global model storage
models: Dict[str, Any] = {}


class PredictionRequest(BaseModel):
    data: List[List[float]]
    model_name: str = "default"


class PredictionResponse(BaseModel):
    predictions: List[List[float]]
    inference_time_ms: float
    model_name: str


class ModelInfo(BaseModel):
    name: str
    input_shape: List[int]
    output_shape: List[int]
    parameters: int
    loaded: bool


def create_default_model():
    """Create a default model for demonstration"""
    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 2),
    )
    return model


def count_parameters(model):
    """Count the number of parameters in a model"""
    total = 0
    for param in model.parameters():
        total += param.data.size
    return total


@app.on_event("startup")
async def startup_event():
    """Initialize default model on startup"""
    print("🚀 Starting TensorBrain Serving Runtime...")
    
    # Create and load default model
    default_model = create_default_model()
    models["default"] = {
        "model": default_model,
        "input_shape": [None, 2],
        "output_shape": [None, 2],
        "parameters": count_parameters(default_model),
        "loaded": True
    }
    
    print(f"✅ Loaded default model with {models['default']['parameters']} parameters")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "TensorBrain Serving Runtime",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "models": "/models",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "models_loaded": len([m for m in models.values() if m["loaded"]])
    }


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all loaded models"""
    model_list = []
    for name, model_info in models.items():
        model_list.append(ModelInfo(
            name=name,
            input_shape=model_info["input_shape"],
            output_shape=model_info["output_shape"],
            parameters=model_info["parameters"],
            loaded=model_info["loaded"]
        ))
    return model_list


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions using a loaded model"""
    start_time = time.time()
    
    # Check if model exists
    if request.model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found")
    
    model_info = models[request.model_name]
    if not model_info["loaded"]:
        raise HTTPException(status_code=400, detail=f"Model '{request.model_name}' not loaded")
    
    try:
        # Convert input to numpy array
        input_data = np.array(request.data, dtype=np.float32)
        
        # Create tensor
        input_tensor = Tensor(input_data, requires_grad=False)
        
        # Make prediction
        model = model_info["model"]
        predictions = model(input_tensor)
        
        # Convert to list for JSON response
        predictions_list = predictions.data.tolist()
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return PredictionResponse(
            predictions=predictions_list,
            inference_time_ms=inference_time,
            model_name=request.model_name
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/models/{model_name}/load")
async def load_model(model_name: str):
    """Load a model (placeholder for future model loading)"""
    # For now, just create a new model
    model = create_default_model()
    models[model_name] = {
        "model": model,
        "input_shape": [None, 2],
        "output_shape": [None, 2],
        "parameters": count_parameters(model),
        "loaded": True
    }
    
    return {
        "message": f"Model '{model_name}' loaded successfully",
        "parameters": models[model_name]["parameters"]
    }


@app.delete("/models/{model_name}")
async def unload_model(model_name: str):
    """Unload a model"""
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    models[model_name]["loaded"] = False
    return {"message": f"Model '{model_name}' unloaded successfully"}


# Benchmarking endpoint
@app.post("/benchmark")
async def benchmark_model(request: PredictionRequest):
    """Benchmark model performance"""
    if request.model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found")
    
    model_info = models[request.model_name]
    if not model_info["loaded"]:
        raise HTTPException(status_code=400, detail=f"Model '{request.model_name}' not loaded")
    
    # Run multiple predictions for benchmarking
    num_runs = 100
    times = []
    
    input_data = np.array(request.data, dtype=np.float32)
    input_tensor = Tensor(input_data, requires_grad=False)
    model = model_info["model"]
    
    for _ in range(num_runs):
        start_time = time.time()
        predictions = model(input_tensor)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Calculate statistics
    avg_time = np.mean(times)
    p95_time = np.percentile(times, 95)
    min_time = np.min(times)
    max_time = np.max(times)
    
    return {
        "model_name": request.model_name,
        "num_runs": num_runs,
        "avg_latency_ms": avg_time,
        "p95_latency_ms": p95_time,
        "min_latency_ms": min_time,
        "max_latency_ms": max_time,
        "throughput_qps": 1000 / avg_time if avg_time > 0 else 0
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting TensorBrain Serving Runtime...")
    print("📖 API Documentation available at: http://localhost:8000/docs")
    print("🔍 Health check at: http://localhost:8000/health")
    print("📊 Models list at: http://localhost:8000/models")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
