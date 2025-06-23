#!/usr/bin/env python
"""
performance_monitor.py
Simple performance monitoring for your RAG system
"""
import time
import logging
from functools import wraps
from contextlib import contextmanager
import json

logger = logging.getLogger("PerformanceMonitor")


class SimplePerformanceMonitor:
    """Lightweight performance monitor for RAG operations"""

    def __init__(self):
        self.timings = {}
        self.counters = {}

    def time_operation(self, operation_name):
        """Decorator to time operations"""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    success = True
                    return result
                except Exception as e:
                    success = False
                    raise
                finally:
                    elapsed = time.time() - start_time
                    self.record_timing(operation_name, elapsed, success)

            return wrapper

        return decorator

    def record_timing(self, operation, elapsed, success=True):
        """Record timing for an operation"""
        if operation not in self.timings:
            self.timings[operation] = []

        self.timings[operation].append(
            {"time": elapsed, "success": success, "timestamp": time.time()}
        )

        # Keep only last 100 measurements
        if len(self.timings[operation]) > 100:
            self.timings[operation] = self.timings[operation][-100:]

        status = "✅" if success else "❌"
        logger.info(f"{status} {operation}: {elapsed:.2f}s")

    def increment_counter(self, counter_name, value=1):
        """Increment a counter"""
        if counter_name not in self.counters:
            self.counters[counter_name] = 0
        self.counters[counter_name] += value

    def get_stats(self, operation=None):
        """Get performance statistics"""
        if operation and operation in self.timings:
            times = [t["time"] for t in self.timings[operation] if t["success"]]
            if times:
                return {
                    "operation": operation,
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "total_calls": len(self.timings[operation]),
                    "success_rate": sum(
                        1 for t in self.timings[operation] if t["success"]
                    )
                    / len(self.timings[operation]),
                }

        # Return all stats
        stats = {}
        for op, timing_list in self.timings.items():
            times = [t["time"] for t in timing_list if t["success"]]
            if times:
                stats[op] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "total_calls": len(timing_list),
                    "success_rate": sum(1 for t in timing_list if t["success"])
                    / len(timing_list),
                }

        stats["counters"] = self.counters.copy()
        return stats

    def print_summary(self):
        """Print performance summary"""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 60)

        for operation, data in stats.items():
            if operation == "counters":
                continue

            print(f"\n🔍 {operation}:")
            print(f"   ⏱️  Average: {data['avg_time']:.2f}s")
            print(f"   ⚡ Fastest: {data['min_time']:.2f}s")
            print(f"   🐌 Slowest: {data['max_time']:.2f}s")
            print(f"   ✅ Success: {data['success_rate']:.1%}")
            print(f"   📞 Calls: {data['total_calls']}")

        if "counters" in stats and stats["counters"]:
            print(f"\n📊 Counters:")
            for counter, value in stats["counters"].items():
                print(f"   {counter}: {value}")

    def save_stats(self, filename):
        """Save stats to file"""
        stats = self.get_stats()
        with open(filename, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"📊 Stats saved to {filename}")


# Global monitor instance
monitor = SimplePerformanceMonitor()


@contextmanager
def time_block(operation_name):
    """Context manager to time code blocks"""
    start_time = time.time()
    try:
        yield
        success = True
    except Exception:
        success = False
        raise
    finally:
        elapsed = time.time() - start_time
        monitor.record_timing(operation_name, elapsed, success)


# Convenience decorators for your specific operations
def time_retrieval(func):
    """Decorator specifically for retrieval operations"""
    return monitor.time_operation("retrieval")(func)


def time_full_text(func):
    """Decorator specifically for full text operations"""
    return monitor.time_operation("full_text_retrieval")(func)


def time_agent_generation(agent_name):
    """Decorator for agent generation"""

    def decorator(func):
        return monitor.time_operation(f"agent_{agent_name}_generation")(func)

    return decorator


# Add these to your existing CitedRAGENT class methods:
"""
USAGE IN YOUR EXISTING CODE:

1. Add to your imports:
from performance_monitor import monitor, time_block, time_retrieval, time_full_text

2. Add decorators to your Retriever methods:

@time_retrieval
def retrieve_abstracts(self, query: str, top_k: int = None):
    # your existing code

@time_full_text  
def get_full_texts(self, doc_ids: list, db=None):
    # your existing code

3. Add time blocks to your CitedRAGENT.answer_query:

def answer_query(self, query, db=None, choices=None):
    with time_block("total_query_processing"):
        # ... your existing code
        
        with time_block("retrieve_abstracts"):
            retrieved_abstracts = self.retriever.retrieve_abstracts(query, top_k=5)
        
        with time_block("agent1_processing"):
            # Agent 1 code
            
        with time_block("agent2_evaluation"):
            # Agent 2 code
            
        with time_block("get_full_texts"):
            full_texts = self.retriever.get_full_texts(filtered_doc_ids, db=db)
        
        with time_block("agent3_generation"):
            # Agent 3 code
    
    # Print stats every 5 queries
    if len(monitor.timings.get('total_query_processing', [])) % 5 == 0:
        monitor.print_summary()
        
    return cited_answer, debug_info

4. Add cache hit tracking:
# In your retrieve_abstracts method:
if cache_key in self._abstract_cache:
    monitor.increment_counter("abstract_cache_hits")
    return self._abstract_cache[cache_key]
else:
    monitor.increment_counter("abstract_cache_misses")
"""


def benchmark_before_after(old_func, new_func, test_cases, iterations=3):
    """Compare performance of old vs new implementation"""
    print("🔬 BENCHMARKING OLD vs NEW IMPLEMENTATION")
    print("-" * 50)

    old_times = []
    new_times = []

    for test_case in test_cases:
        print(f"\nTest case: {str(test_case)[:50]}...")

        # Benchmark old implementation
        old_case_times = []
        for i in range(iterations):
            start = time.time()
            try:
                old_func(*test_case if isinstance(test_case, tuple) else (test_case,))
                elapsed = time.time() - start
                old_case_times.append(elapsed)
            except Exception as e:
                print(f"Old implementation failed: {e}")
                old_case_times.append(float("inf"))

        # Benchmark new implementation
        new_case_times = []
        for i in range(iterations):
            start = time.time()
            try:
                new_func(*test_case if isinstance(test_case, tuple) else (test_case,))
                elapsed = time.time() - start
                new_case_times.append(elapsed)
            except Exception as e:
                print(f"New implementation failed: {e}")
                new_case_times.append(float("inf"))

        old_avg = sum(old_case_times) / len(old_case_times)
        new_avg = sum(new_case_times) / len(new_case_times)

        if old_avg != float("inf") and new_avg != float("inf"):
            speedup = old_avg / new_avg
            print(
                f"   Old: {old_avg:.2f}s, New: {new_avg:.2f}s, Speedup: {speedup:.1f}x"
            )

        old_times.extend(old_case_times)
        new_times.extend(new_case_times)

    # Overall summary
    valid_old = [t for t in old_times if t != float("inf")]
    valid_new = [t for t in new_times if t != float("inf")]

    if valid_old and valid_new:
        overall_speedup = (sum(valid_old) / len(valid_old)) / (
            sum(valid_new) / len(valid_new)
        )
        print(f"\n🎯 OVERALL SPEEDUP: {overall_speedup:.1f}x")


if __name__ == "__main__":
    # Example usage
    print("Performance Monitor ready!")
    print("Add the decorators and time blocks to your code as shown in the docstring.")
