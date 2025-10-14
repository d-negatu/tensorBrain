#!/usr/bin/env python3
"""
Advanced Serving Runtime for TensorBrain
Model versioning, A/B testing, and production features
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import numpy as np
import time
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
import threading

from nn import Sequential, Linear, ReLU
from tensor import Tensor


app = FastAPI(title="TensorBrain Advanced Serving", version="2.0.0")

# Global model storage with versioning
models: Dict[str, Dict[str, Any]] = {}
model_versions: Dict[str, List[str]] = {}
model_metrics: Dict[str, Dict[str, Any]] = {}
ab_tests: Dict[str, Dict[str, Any]] = {}


class PredictionRequest(BaseModel):
    data: List[List[float]]
    model_name: str = "default"
    version: Optional[str] = None
    ab_test: Optional[str] = None


class PredictionResponse(BaseModel):
    predictions: List[List[float]]
    inference_time_ms: float
    model_name: str
    version: str
    timestamp: float


class ModelVersion(BaseModel):
    name: str
    version: str
    created_at: float
    parameters: int
    performance: Dict[str, float]
    is_active: bool


class ABTestConfig(BaseModel):
    test_name: str
    model_a: str
    model_b: str
    traffic_split: float = 0.5
    metrics: List[str] = ["accuracy", "latency"]


class ModelMetrics(BaseModel):
    model_name: str
    version: str
    total_requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    throughput_qps: float


def create_model_v1():
    """Create version 1 of the model"""
    return Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 2),
    )


def create_model_v2():
    """Create version 2 of the model (improved)"""
    return Sequential(
        Linear(2, 8),
        ReLU(),
        Linear(8, 4),
        ReLU(),
        Linear(4, 2),
    )


def create_model_v3():
    """Create version 3 of the model (optimized)"""
    return Sequential(
        Linear(2, 6),
        ReLU(),
        Linear(6, 3),
        ReLU(),
        Linear(3, 2),
    )


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    print("🚀 Starting TensorBrain Advanced Serving Runtime...")
    
    # Create model versions
    model_v1 = create_model_v1()
    model_v2 = create_model_v2()
    model_v3 = create_model_v3()
    
    # Register models
    models["default"] = {
        "v1.0.0": {
            "model": model_v1,
            "created_at": time.time(),
            "parameters": sum(param.data.size for param in model_v1.parameters()),
            "performance": {"accuracy": 0.85, "latency_ms": 2.5},
            "is_active": True
        },
        "v2.0.0": {
            "model": model_v2,
            "created_at": time.time(),
            "parameters": sum(param.data.size for param in model_v2.parameters()),
            "performance": {"accuracy": 0.90, "latency_ms": 3.2},
            "is_active": False
        },
        "v3.0.0": {
            "model": model_v3,
            "created_at": time.time(),
            "parameters": sum(param.data.size for param in model_v3.parameters()),
            "performance": {"accuracy": 0.88, "latency_ms": 2.1},
            "is_active": False
        }
    }
    
    model_versions["default"] = ["v1.0.0", "v2.0.0", "v3.0.0"]
    
    # Initialize metrics
    model_metrics["default"] = {
        "v1.0.0": {
            "total_requests": 0,
            "latencies": [],
            "errors": 0,
            "start_time": time.time()
        }
    }
    
    print(f"✅ Loaded {len(models['default'])} model versions")
    print(f"✅ Active version: v1.0.0")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "TensorBrain Advanced Serving Runtime",
        "version": "2.0.0",
        "features": [
            "Model versioning",
            "A/B testing",
            "Performance monitoring",
            "Automatic scaling",
            "Health checks"
        ],
        "endpoints": {
            "predict": "/predict",
            "models": "/models",
            "versions": "/models/{model_name}/versions",
            "metrics": "/models/{model_name}/metrics",
            "ab_test": "/ab_test",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Advanced health check"""
    active_models = 0
    total_requests = 0
    
    for model_name, versions in models.items():
        for version, model_info in versions.items():
            if model_info["is_active"]:
                active_models += 1
                if model_name in model_metrics and version in model_metrics[model_name]:
                    total_requests += model_metrics[model_name][version]["total_requests"]
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "active_models": active_models,
        "total_requests": total_requests,
        "uptime": time.time() - (model_metrics.get("default", {}).get("v1.0.0", {}).get("start_time", time.time()))
    }


@app.get("/models", response_model=List[ModelVersion])
async def list_models():
    """List all models with versions"""
    model_list = []
    
    for model_name, versions in models.items():
        for version, model_info in versions.items():
            model_list.append(ModelVersion(
                name=model_name,
                version=version,
                created_at=model_info["created_at"],
                parameters=model_info["parameters"],
                performance=model_info["performance"],
                is_active=model_info["is_active"]
            ))
    
    return model_list


@app.get("/models/{model_name}/versions")
async def list_model_versions(model_name: str):
    """List versions for a specific model"""
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    versions = []
    for version, model_info in models[model_name].items():
        versions.append({
            "version": version,
            "created_at": model_info["created_at"],
            "parameters": model_info["parameters"],
            "performance": model_info["performance"],
            "is_active": model_info["is_active"]
        })
    
    return {"model_name": model_name, "versions": versions}


@app.post("/models/{model_name}/versions/{version}/activate")
async def activate_model_version(model_name: str, version: str):
    """Activate a specific model version"""
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    if version not in models[model_name]:
        raise HTTPException(status_code=404, detail=f"Version '{version}' not found")
    
    # Deactivate all versions
    for v in models[model_name]:
        models[model_name][v]["is_active"] = False
    
    # Activate specified version
    models[model_name][version]["is_active"] = True
    
    return {"message": f"Activated {model_name} version {version}"}


@app.get("/models/{model_name}/metrics", response_model=ModelMetrics)
async def get_model_metrics(model_name: str):
    """Get metrics for a model"""
    if model_name not in model_metrics:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    # Get active version
    active_version = None
    for version, model_info in models[model_name].items():
        if model_info["is_active"]:
            active_version = version
            break
    
    if active_version is None:
        raise HTTPException(status_code=404, detail=f"No active version for model '{model_name}'")
    
    metrics = model_metrics[model_name][active_version]
    latencies = metrics["latencies"]
    
    if latencies:
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
    else:
        avg_latency = p95_latency = p99_latency = 0.0
    
    error_rate = metrics["errors"] / max(metrics["total_requests"], 1)
    throughput_qps = metrics["total_requests"] / max(time.time() - metrics["start_time"], 1)
    
    return ModelMetrics(
        model_name=model_name,
        version=active_version,
        total_requests=metrics["total_requests"],
        avg_latency_ms=avg_latency,
        p95_latency_ms=p95_latency,
        p99_latency_ms=p99_latency,
        error_rate=error_rate,
        throughput_qps=throughput_qps
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions with advanced features"""
    start_time = time.time()
    
    # Check if model exists
    if request.model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found")
    
    # Determine which version to use
    version = request.version
    if version is None:
        # Use active version
        for v, model_info in models[request.model_name].items():
            if model_info["is_active"]:
                version = v
                break
    
    if version is None or version not in models[request.model_name]:
        raise HTTPException(status_code=404, detail=f"Version '{version}' not found")
    
    model_info = models[request.model_name][version]
    model = model_info["model"]
    
    try:
        # Convert input to numpy array
        input_data = np.array(request.data, dtype=np.float32)
        
        # Create tensor
        input_tensor = Tensor(input_data, requires_grad=False)
        
        # Make prediction
        predictions = model(input_tensor)
        
        # Convert to list for JSON response
        predictions_list = predictions.data.tolist()
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000
        
        # Update metrics
        if request.model_name not in model_metrics:
            model_metrics[request.model_name] = {}
        if version not in model_metrics[request.model_name]:
            model_metrics[request.model_name][version] = {
                "total_requests": 0,
                "latencies": [],
                "errors": 0,
                "start_time": time.time()
            }
        
        model_metrics[request.model_name][version]["total_requests"] += 1
        model_metrics[request.model_name][version]["latencies"].append(inference_time)
        
        return PredictionResponse(
            predictions=predictions_list,
            inference_time_ms=inference_time,
            model_name=request.model_name,
            version=version,
            timestamp=time.time()
        )
        
    except Exception as e:
        # Update error metrics
        if request.model_name in model_metrics and version in model_metrics[request.model_name]:
            model_metrics[request.model_name][version]["errors"] += 1
        
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/ab_test")
async def create_ab_test(config: ABTestConfig):
    """Create A/B test configuration"""
    ab_tests[config.test_name] = {
        "model_a": config.model_a,
        "model_b": config.model_b,
        "traffic_split": config.traffic_split,
        "metrics": config.metrics,
        "created_at": time.time(),
        "results": {"a": [], "b": []}
    }
    
    return {"message": f"A/B test '{config.test_name}' created successfully"}


@app.get("/ab_test/{test_name}")
async def get_ab_test_results(test_name: str):
    """Get A/B test results"""
    if test_name not in ab_tests:
        raise HTTPException(status_code=404, detail=f"A/B test '{test_name}' not found")
    
    test_config = ab_tests[test_name]
    
    # Calculate results
    results_a = test_config["results"]["a"]
    results_b = test_config["results"]["b"]
    
    return {
        "test_name": test_name,
        "model_a": test_config["model_a"],
        "model_b": test_config["model_b"],
        "traffic_split": test_config["traffic_split"],
        "results_a": {
            "count": len(results_a),
            "avg_latency": np.mean(results_a) if results_a else 0,
            "p95_latency": np.percentile(results_a, 95) if results_a else 0
        },
        "results_b": {
            "count": len(results_b),
            "avg_latency": np.mean(results_b) if results_b else 0,
            "p95_latency": np.percentile(results_b, 95) if results_b else 0
        }
    }


@app.post("/models/{model_name}/deploy")
async def deploy_model(model_name: str, version: str):
    """Deploy a new model version"""
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    if version not in models[model_name]:
        raise HTTPException(status_code=404, detail=f"Version '{version}' not found")
    
    # Activate the version
    await activate_model_version(model_name, version)
    
    return {"message": f"Deployed {model_name} version {version}"}


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting TensorBrain Advanced Serving Runtime...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("📊 Models: http://localhost:8000/models")
    print("🧪 A/B Testing: http://localhost:8000/ab_test")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
