# FP4 vs INT4 Comparison

This project compares **FP4 (nf4)** and **INT4** quantization methods for transformer models.

## Recommended Quantization
I recommend using **FP4 (nf4)** quantization as the default choice for inference due to its strong balance between accuracy retention and performance improvements.

## Author
**Rohith Garapati** (<https://github.com/INFINITYone22>)

## Repository
<https://github.com/INFINITYone22/fp4_vs_int4_comparison>

## Running Tests

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the test suite:

```bash
pytest
```

## Benchmarking

To run automated benchmarks:

```bash
python src/benchmark.py
```

Results will be saved to `research/results.json`.
