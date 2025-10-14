#!/usr/bin/env python3
"""
Production Monitoring and Analytics for TensorBrain
Real-time metrics, alerting, and performance analysis
"""

import numpy as np
import time
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading
import queue

from tensor import Tensor
from nn import Module, Sequential, Linear, ReLU


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: float
    value: float
    tags: Dict[str, str]


@dataclass
class Alert:
    """Alert configuration"""
    name: str
    metric: str
    threshold: float
    condition: str  # "greater", "less", "equal"
    severity: str  # "critical", "warning", "info"
    enabled: bool = True


@dataclass
class AlertEvent:
    """Alert event"""
    alert_name: str
    metric: str
    value: float
    threshold: float
    timestamp: float
    severity: str


class MetricsCollector:
    """Real-time metrics collection"""
    
    def __init__(self, retention_hours: int = 24):
        self.metrics: Dict[str, List[MetricPoint]] = {}
        self.alerts: Dict[str, Alert] = {}
        self.alert_events: List[AlertEvent] = []
        self.retention_hours = retention_hours
        self.lock = threading.Lock()
        
        print(f"🚀 Initialized MetricsCollector:")
        print(f"   Retention: {retention_hours} hours")
        print(f"   Metrics: {len(self.metrics)}")
        print(f"   Alerts: {len(self.alerts)}")
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a metric value"""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = []
            
            metric_point = MetricPoint(
                timestamp=time.time(),
                value=value,
                tags=tags or {}
            )
            
            self.metrics[name].append(metric_point)
            
            # Check alerts
            self._check_alerts(name, value)
            
            # Clean old metrics
            self._cleanup_old_metrics()
    
    def _check_alerts(self, metric_name: str, value: float):
        """Check if any alerts should fire"""
        for alert_name, alert in self.alerts.items():
            if alert.metric == metric_name and alert.enabled:
                should_fire = False
                
                if alert.condition == "greater" and value > alert.threshold:
                    should_fire = True
                elif alert.condition == "less" and value < alert.threshold:
                    should_fire = True
                elif alert.condition == "equal" and value == alert.threshold:
                    should_fire = True
                
                if should_fire:
                    alert_event = AlertEvent(
                        alert_name=alert_name,
                        metric=metric_name,
                        value=value,
                        threshold=alert.threshold,
                        timestamp=time.time(),
                        severity=alert.severity
                    )
                    self.alert_events.append(alert_event)
                    print(f"🚨 ALERT: {alert_name} - {metric_name} = {value} (threshold: {alert.threshold})")
    
    def _cleanup_old_metrics(self):
        """Remove old metric points"""
        cutoff_time = time.time() - (self.retention_hours * 3600)
        
        for metric_name, points in self.metrics.items():
            self.metrics[metric_name] = [
                point for point in points if point.timestamp > cutoff_time
            ]
    
    def add_alert(self, alert: Alert):
        """Add an alert configuration"""
        self.alerts[alert.name] = alert
        print(f"✅ Added alert: {alert.name} - {alert.metric} {alert.condition} {alert.threshold}")
    
    def get_metric_stats(self, metric_name: str, hours: int = 1) -> Dict[str, float]:
        """Get statistics for a metric"""
        if metric_name not in self.metrics:
            return {}
        
        cutoff_time = time.time() - (hours * 3600)
        recent_points = [
            point for point in self.metrics[metric_name] 
            if point.timestamp > cutoff_time
        ]
        
        if not recent_points:
            return {}
        
        values = [point.value for point in recent_points]
        
        return {
            "count": len(values),
            "min": np.min(values),
            "max": np.max(values),
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99)
        }
    
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics"""
        all_stats = {}
        for metric_name in self.metrics:
            all_stats[metric_name] = self.get_metric_stats(metric_name)
        return all_stats
    
    def get_recent_alerts(self, hours: int = 1) -> List[AlertEvent]:
        """Get recent alert events"""
        cutoff_time = time.time() - (hours * 3600)
        return [
            alert for alert in self.alert_events 
            if alert.timestamp > cutoff_time
        ]


class PerformanceProfiler:
    """Performance profiling for models"""
    
    def __init__(self):
        self.profiles: Dict[str, Dict[str, Any]] = {}
    
    def profile_model(self, model: Module, sample_input: Tensor, 
                     num_runs: int = 100) -> Dict[str, float]:
        """Profile model performance"""
        print(f"📊 Profiling model performance...")
        
        # Warmup
        for _ in range(10):
            _ = model(sample_input)
        
        # Profile forward pass
        forward_times = []
        for _ in range(num_runs):
            start_time = time.time()
            output = model(sample_input)
            end_time = time.time()
            forward_times.append((end_time - start_time) * 1000)
        
        # Profile backward pass (if gradients enabled)
        backward_times = []
        if sample_input.requires_grad:
            for _ in range(num_runs):
                output = model(sample_input)
                loss = output.sum()
                
                start_time = time.time()
                loss.backward()
                end_time = time.time()
                backward_times.append((end_time - start_time) * 1000)
        
        # Calculate statistics
        forward_stats = {
            "avg_ms": np.mean(forward_times),
            "min_ms": np.min(forward_times),
            "max_ms": np.max(forward_times),
            "p95_ms": np.percentile(forward_times, 95),
            "p99_ms": np.percentile(forward_times, 99),
            "std_ms": np.std(forward_times)
        }
        
        backward_stats = {}
        if backward_times:
            backward_stats = {
                "avg_ms": np.mean(backward_times),
                "min_ms": np.min(backward_times),
                "max_ms": np.max(backward_times),
                "p95_ms": np.percentile(backward_times, 95),
                "p99_ms": np.percentile(backward_times, 99),
                "std_ms": np.std(backward_times)
            }
        
        # Memory usage
        param_count = sum(param.data.size for param in model.parameters())
        memory_mb = param_count * 4 / (1024 * 1024)
        
        profile_results = {
            "forward_pass": forward_stats,
            "backward_pass": backward_stats,
            "memory_usage_mb": memory_mb,
            "parameter_count": param_count,
            "num_runs": num_runs
        }
        
        print(f"✅ Profiling completed:")
        print(f"   Forward pass: {forward_stats['avg_ms']:.2f}ms avg")
        print(f"   Memory usage: {memory_mb:.2f}MB")
        print(f"   Parameters: {param_count:,}")
        
        return profile_results


class SystemMonitor:
    """System monitoring and health checks"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.profiler = PerformanceProfiler()
        
        # Add default alerts
        self.metrics_collector.add_alert(Alert(
            name="high_latency",
            metric="inference_latency_ms",
            threshold=100.0,
            condition="greater",
            severity="warning"
        ))
        
        self.metrics_collector.add_alert(Alert(
            name="high_error_rate",
            metric="error_rate",
            threshold=0.05,
            condition="greater",
            severity="critical"
        ))
        
        self.metrics_collector.add_alert(Alert(
            name="low_throughput",
            metric="throughput_qps",
            threshold=100.0,
            condition="less",
            severity="warning"
        ))
    
    def monitor_model_performance(self, model: Module, sample_input: Tensor):
        """Monitor model performance in real-time"""
        # Profile model
        profile_results = self.profiler.profile_model(model, sample_input)
        
        # Record metrics
        self.metrics_collector.record_metric(
            "inference_latency_ms",
            profile_results["forward_pass"]["avg_ms"],
            {"model": "default"}
        )
        
        self.metrics_collector.record_metric(
            "memory_usage_mb",
            profile_results["memory_usage_mb"],
            {"model": "default"}
        )
        
        self.metrics_collector.record_metric(
            "parameter_count",
            profile_results["parameter_count"],
            {"model": "default"}
        )
        
        return profile_results
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        all_metrics = self.metrics_collector.get_all_metrics()
        recent_alerts = self.metrics_collector.get_recent_alerts(hours=1)
        
        # Calculate health score
        health_score = 100.0
        
        # Check latency
        if "inference_latency_ms" in all_metrics:
            avg_latency = all_metrics["inference_latency_ms"].get("mean", 0)
            if avg_latency > 50:
                health_score -= 20
            elif avg_latency > 25:
                health_score -= 10
        
        # Check error rate
        if "error_rate" in all_metrics:
            error_rate = all_metrics["error_rate"].get("mean", 0)
            if error_rate > 0.1:
                health_score -= 30
            elif error_rate > 0.05:
                health_score -= 15
        
        # Check alerts
        critical_alerts = [alert for alert in recent_alerts if alert.severity == "critical"]
        warning_alerts = [alert for alert in recent_alerts if alert.severity == "warning"]
        
        health_score -= len(critical_alerts) * 20
        health_score -= len(warning_alerts) * 5
        
        health_score = max(0, health_score)
        
        return {
            "health_score": health_score,
            "status": "healthy" if health_score > 80 else "degraded" if health_score > 50 else "unhealthy",
            "metrics": all_metrics,
            "recent_alerts": len(recent_alerts),
            "critical_alerts": len(critical_alerts),
            "warning_alerts": len(warning_alerts),
            "timestamp": time.time()
        }


def demo_monitoring():
    """Demonstrate monitoring capabilities"""
    print("📊 TensorBrain Production Monitoring Demo")
    print("=" * 50)
    
    # Create system monitor
    monitor = SystemMonitor()
    
    # Create sample model
    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 2),
    )
    
    sample_input = Tensor(np.random.randn(10, 2), requires_grad=False)
    
    # Monitor model performance
    profile_results = monitor.monitor_model_performance(model, sample_input)
    
    # Simulate some metrics
    print("\n🔄 Simulating metrics...")
    for i in range(10):
        # Simulate inference latency
        latency = np.random.normal(25, 5)  # 25ms ± 5ms
        monitor.metrics_collector.record_metric("inference_latency_ms", latency)
        
        # Simulate error rate
        error_rate = np.random.uniform(0, 0.1)
        monitor.metrics_collector.record_metric("error_rate", error_rate)
        
        # Simulate throughput
        throughput = np.random.normal(500, 100)  # 500 QPS ± 100
        monitor.metrics_collector.record_metric("throughput_qps", throughput)
        
        time.sleep(0.1)
    
    # Get system health
    health = monitor.get_system_health()
    
    print("\n📊 System Health Report:")
    print(f"Health Score: {health['health_score']:.1f}/100")
    print(f"Status: {health['status']}")
    print(f"Recent Alerts: {health['recent_alerts']}")
    print(f"Critical Alerts: {health['critical_alerts']}")
    print(f"Warning Alerts: {health['warning_alerts']}")
    
    # Get metric statistics
    print("\n📈 Metric Statistics:")
    all_metrics = monitor.metrics_collector.get_all_metrics()
    for metric_name, stats in all_metrics.items():
        if stats:
            print(f"{metric_name}:")
            print(f"  Count: {stats['count']}")
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  P95: {stats['p95']:.2f}")
            print(f"  P99: {stats['p99']:.2f}")
    
    print("\n🎉 Production monitoring is working!")
    print("📝 Next steps:")
    print("   • Add Grafana integration")
    print("   • Implement Prometheus metrics")
    print("   • Add Slack/email alerting")
    print("   • Implement auto-scaling")
    print("   • Add distributed tracing")
    print("   • Implement circuit breakers")


if __name__ == "__main__":
    demo_monitoring()
