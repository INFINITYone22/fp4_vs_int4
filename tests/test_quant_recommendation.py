import pytest


def test_recommend_fp4():
    with open("research/report.md", "r", encoding="utf-8") as f:
        content = f.read()
    assert "FP4" in content, "The research report should recommend FP4 quantization." 