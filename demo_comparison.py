#!/usr/bin/env python3
"""
Demo comparing the old complex system vs new agentic coder
"""

import time
import psutil
import os
from pathlib import Path


def measure_resource_usage(func):
    """Measure CPU and memory usage of a function"""
    process = psutil.Process()

    # Measure before
    cpu_before = process.cpu_percent()
    memory_before = process.memory_info().rss / 1024 / 1024  # MB

    start_time = time.time()
    result = func()
    end_time = time.time()

    # Measure after
    cpu_after = process.cpu_percent()
    memory_after = process.memory_info().rss / 1024 / 1024  # MB

    return {
        "result": result,
        "time": end_time - start_time,
        "cpu_usage": cpu_after - cpu_before,
        "memory_usage": memory_after - memory_before,
        "memory_peak": memory_after,
    }


def demo_old_system():
    """Demo the old complex system"""
    print("🔧 Loading old complex system...")
    try:
        from core.ai import AIEngine
        from core.autonomous_agent import AutonomousAgent

        # Initialize the complex system
        ai_engine = AIEngine()
        agent = AutonomousAgent(ai_engine)

        return "Old system loaded successfully"
    except Exception as e:
        return f"Old system failed to load: {e}"


def demo_new_system():
    """Demo the new agentic coder"""
    print("⚡ Loading new agentic coder...")
    try:
        from agentic_coder import AgenticCoder

        # Initialize the lightweight system
        agent = AgenticCoder(model="mock")  # Use mock to avoid API calls

        return "New system loaded successfully"
    except Exception as e:
        return f"New system failed to load: {e}"


def compare_dependencies():
    """Compare dependencies"""
    print("\n📦 Dependency Comparison:")

    # Old system dependencies
    old_deps = [
        "rich>=10.0.0",
        "prompt_toolkit>=3.0.0",
        "click>=8.0.0",
        "requests>=2.25.0",
        "google-generativeai>=0.3.0",
        "openai>=1.0.0",
    ]

    # New system dependencies
    new_deps = ["openai>=1.0.0  # Optional", "google-generativeai>=0.3.0  # Optional"]

    print(f"Old system: {len(old_deps)} required dependencies")
    for dep in old_deps:
        print(f"  - {dep}")

    print(f"\nNew system: {len(new_deps)} optional dependencies")
    for dep in new_deps:
        print(f"  - {dep}")

    print(
        f"\nDependency reduction: {len(old_deps) - len(new_deps)} fewer required packages"
    )


def compare_file_sizes():
    """Compare file sizes"""
    print("\n📏 File Size Comparison:")

    old_files = [
        "core/ai.py",
        "core/autonomous_agent.py",
        "core/auto_fix.py",
        "core/learning_engine.py",
        "core/reasoning.py",
        "core/task_manager.py",
        "core/memory.py",
        "core/action_logger.py",
    ]

    new_files = ["agentic_coder.py", "simple_memory.py"]

    old_total = 0
    for file_path in old_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            old_total += size
            print(f"  {file_path}: {size:,} bytes")

    print(f"\nOld system total: {old_total:,} bytes")

    new_total = 0
    for file_path in new_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            new_total += size
            print(f"  {file_path}: {size:,} bytes")

    print(f"New system total: {new_total:,} bytes")
    print(f"Size reduction: {((old_total - new_total) / old_total * 100):.1f}%")


def main():
    """Main demo"""
    print("🚀 Agentic Coder vs Complex System Comparison")
    print("=" * 50)

    # Compare dependencies
    compare_dependencies()

    # Compare file sizes
    compare_file_sizes()

    # Compare resource usage
    print("\n⚡ Resource Usage Comparison:")

    print("\nTesting old system...")
    old_metrics = measure_resource_usage(demo_old_system)
    print(f"Result: {old_metrics['result']}")
    print(f"Load time: {old_metrics['time']:.2f}s")
    print(f"Memory peak: {old_metrics['memory_peak']:.1f} MB")

    print("\nTesting new system...")
    new_metrics = measure_resource_usage(demo_new_system)
    print(f"Result: {new_metrics['result']}")
    print(f"Load time: {new_metrics['time']:.2f}s")
    print(f"Memory peak: {new_metrics['memory_peak']:.1f} MB")

    # Summary
    print("\n📊 Summary:")
    print(
        f"Load time improvement: {((old_metrics['time'] - new_metrics['time']) / old_metrics['time'] * 100):.1f}%"
    )
    print(
        f"Memory usage improvement: {((old_metrics['memory_peak'] - new_metrics['memory_peak']) / old_metrics['memory_peak'] * 100):.1f}%"
    )

    print("\n✨ Key Improvements:")
    print("  ✅ Minimal dependencies (only stdlib + optional LLM clients)")
    print("  ✅ Single-file architecture for core functionality")
    print("  ✅ Lightweight memory system instead of complex learning engine")
    print("  ✅ Direct tool calling without heavy abstraction layers")
    print("  ✅ Efficient context management (last 20 messages)")
    print("  ✅ Resource-conscious design")
    print("  ✅ Same agentic capabilities with much less overhead")


if __name__ == "__main__":
    main()
