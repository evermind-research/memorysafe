# MemorySafe – Benchmark Results

This document explains the contents of `results.json`, which stores the raw experimental outputs of the MemorySafe validation benchmarks.

The purpose of these results is to provide **auditable and reproducible evidence** for the claims made in the MemorySafe documentation and NVIDIA application.

---

## What is in `results.json`

The results file contains aggregated metrics for each benchmark evaluated using:

- the same unchanged MemorySafe policy  
- identical memory capacity  
- no dataset-specific tuning  

Each entry corresponds to one dataset.

---

## Key Metrics

### Task-0 Protection
Accuracy on the first task after learning all subsequent tasks.

This measures **long-term memory retention**.

### MVI (Memory Vulnerability Index)
Average vulnerability score for Task-0 memories.

Lower values indicate stronger protection.

### Rare-Class Recall
Recall on low-frequency or safety-critical classes.

Used for medical and anomaly datasets.

---

## Summary Results

| Dataset        | Tasks | Task-0 Protection | Rare Recall |
|----------------|------:|------------------:|------------:|
| MNIST          | 5     | 1.00              | N/A         |
| Fashion-MNIST  | 5     | 1.00              | N/A         |
| CIFAR-10       | 5     | 1.00              | N/A         |
| CIFAR-100      | 10    | 1.00              | N/A         |
| Omniglot       | 20    | 1.00              | N/A         |
| Permuted MNIST | 10    | 1.00              | N/A         |
| PneumoniaMNIST | –     | –                 | 0.917       |

---

## Experimental Protocol

All benchmarks were executed with:

- identical policy parameters  
- identical memory capacity  
- fixed random seeds  
- no dataset-specific logic  

This ensures that performance differences reflect:

> task complexity and data modality, not parameter tuning.

---

## Interpretation

These results indicate that MemorySafe learns a **general memory governance policy** that transfers across:

- different task counts  
- different data modalities  
- different continual learning regimes  

This supports the claim that memory allocation can be treated as a **policy problem rather than a dataset-specific engineering problem**.
