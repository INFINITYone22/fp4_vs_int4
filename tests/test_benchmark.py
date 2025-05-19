import os
import json
import subprocess
import sys

def test_benchmark_script_runs():
    cmd = [sys.executable, "src/benchmark.py"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Benchmark script failed: {result.stderr}"
    res_path = "research/results.json"
    assert os.path.exists(res_path), "results.json not found"
    data = json.load(open(res_path))
    for key in ["FP4", "INT4"]:
        assert key in data, f"{key} missing in results"
        assert "perplexity" in data[key] and "latency" in data[key], f"{key} metrics incomplete" 