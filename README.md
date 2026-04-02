# Brevity Constraints Reverse Performance Hierarchies in Language Models

**ICML 2026 Submission - Code Repository**

This repository contains the complete implementation for reproducing all experiments and analyses from our paper "Brevity Constraints Reverse Performance Hierarchies in Language Models."

##  Overview

Our code implements a comprehensive evaluation framework that:
- Evaluates 31 language models (0.5B-405B parameters) across 5 benchmarks
- Identifies 115 "inverse scaling" problems where small models outperform large ones
- Conducts causal intervention experiments testing brevity constraints
- Performs contamination analysis to validate findings
- Generates all statistics, figures, and tables from the paper

##  Key Findings Reproduced

- **7.7%** of benchmark problems exhibit inverse scaling (Section 4.2)
- **26.3pp** accuracy improvement from brevity constraints on large models (Section 4.3)
- **67%** reduction in performance gaps through prompt intervention (Section 4.3)
- Complete gap reversals on GSM8K and MMLU-STEM benchmarks (Figure 3B)

---

##  Repository Structure

```
├── main.py                          # Main evaluation pipeline
├── casual_intervention.py           # Causal intervention experiments (Section 4.3)
├── ablation_study.py               # Comprehensive analysis & statistics
├── cross_model_prober.py           # Core model evaluation engine
├── divergence_analysis.py          # Reasoning divergence analysis
├── model_manager.py                # Model loading & inference
├── validator.py                    # Answer extraction & validation
├── prompt_formater.py              # Standard prompts (control condition)
├── casual_intervention_prompt.py   # Intervention prompts (brief/direct)
├── reasoning_extractor.py          # Step-by-step reasoning extraction
├── utils/
│   └── load_dataset.py            # Dataset loaders (GSM8K, BoolQ, etc.)
├── inverse_scaling_problem_ids.json # 115 identified inverse problems
└── README.md                       # This file
```

---

##  Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (16GB+ VRAM recommended)
- Hugging Face account with access to gated models

### Setup

```bash
pip install -r requirements.txt

# Login to Hugging Face (required for gated models)
huggingface-cli login
# Enter your token when prompted
```


---

##  Acknowledgments

We thank:
- Hugging Face for model hosting infrastructure
- Authors of benchmark datasets (GSM8K, BoolQ, ARC, CommonsenseQA, MMLU)
- ICML reviewers for constructive feedback

---

