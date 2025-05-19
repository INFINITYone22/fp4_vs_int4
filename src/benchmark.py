#!/usr/bin/env python3
import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import math

def load_quantized(model_name, format):
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_format=format,
        bnb_4bit_compute_dtype=torch.float16
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer, device

def evaluate_perplexity(model, tokenizer, device, dataset, num_samples=100):
    losses = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        inputs = tokenizer(sample["text"], return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
        losses.append(loss)
    return math.exp(sum(losses)/len(losses))

def measure_latency(model, tokenizer, device, prompt="Hello, world!", gen_tokens=50, num_runs=10):
    model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model.generate(**inputs, max_new_tokens=gen_tokens)
        times.append(time.perf_counter() - start)
    return sum(times)/len(times)

def main():
    model_name = "distilgpt2"
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    results = {}
    for fmt in ["int4", "nf4"]:
        label = "INT4" if fmt=="int4" else "FP4"
        model, tokenizer, device = load_quantized(model_name, fmt)
        try:
            perplexity = evaluate_perplexity(model, tokenizer, device, dataset)
        except Exception as e:
            print(f"Error evaluating perplexity for {label}: {e}")
            perplexity = None
        try:
            latency = measure_latency(model, tokenizer, device)
        except Exception as e:
            print(f"Error measuring latency for {label}: {e}")
            latency = None
        results[label] = {"perplexity": perplexity, "latency": latency}
        print(f"{label}: Perplexity={perplexity}, Latency={latency}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    with open("research/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to research/results.json")

if __name__ == "__main__":
    main() 