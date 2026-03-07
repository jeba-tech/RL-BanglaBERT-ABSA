# RL-BanglaBERT-ABSA
Reinforcement Learning-Augmented Fine-Tuning for Aspect-Based Sentiment Analysis in Bangla

---

This repository contains the code, dataset, and results for my research paper on **Bangla Aspect-Based Sentiment Analysis (ABSA)**.

The goal is to teach a language model to read a Bangla restaurant review and answer two questions at the same time:
- **What aspect is being talked about?** (food quality, service, price, ambiance, or cleanliness)
- **What is the sentiment toward that aspect?** (positive, negative, or neutral)

The key contribution is using **Reinforcement Learning (PPO)** on top of the standard fine-tuning approach to make the model more robust — meaning it performs well even when the input text is noisy or slightly changed.

---


## Paper

> **"Reinforcement Learning-Augmented Fine-Tuning for Aspect-Based Sentiment Analysis in Bangla Using Pre-Trained Language Models"**
>

---

## Results at a Glance

Experiments were run with **3 random seeds (42, 123, 456)** and reported as mean ± standard deviation.

| Model | Clean Accuracy | Clean F1 | Adv Accuracy | Adv F1 | Robustness |
|---|---|---|---|---|---|
| Baseline (BanglaBERT) | 0.9455 ± 0.0013 | 0.9725 ± 0.0007 | 0.8782 ± 0.0092 | 0.9375 ± 0.0047 | 0.9289 ± 0.0085 |
| mBERT | 0.9410 ± 0.0047 | 0.9696 ± 0.0019 | 0.8654 ± 0.0125 | 0.9291 ± 0.0051 | 0.9197 ± 0.0177 |
| **RL-BanglaBERT (Ours)** | **0.9493 ± 0.0016** | **0.9742 ± 0.0009** | **0.9687 ± 0.0096** | **0.9833 ± 0.0054** | **1.0204 ± 0.0117** |

**Key finding:**
RL-BanglaBERT is the **only model with Robustness Score > 1.0**, meaning it performs better on adversarial inputs than on clean inputs — resolving the typical accuracy-robustness trade-off.

| Improvement | Over Baseline | Over mBERT |
|---|---|---|
| Adversarial Accuracy | **+9.05 percentage points** | **+10.33 percentage points** |
| Adversarial F1 | **+4.58 percentage points** | **+5.42 percentage points** |

---

## Dataset

Bangla restaurant reviews collected from online food delivery platforms in Bangladesh.

| Split | Samples | Used For |
|---|---|---|
| Train | 2,819 | Baseline + mBERT + RL-BanglaBERT |
| Validation | 604 | All models |
| Test | 605 | Final evaluation |
| Adversarial | 520 | Robustness evaluation only |
| Mixed Train | 3,339 | RL-BanglaBERT only (clean + 20% adversarial) |

**5 Aspect classes:** Ambiance · Cleanliness · Food Quality · Price · Service

**3 Sentiment classes:** Negative · Neutral · Positive

**3 Complexity levels:** Simple · Medium · Complex



---
## Seeds and Reproducibility

All experiments use three fixed random seeds for statistical reliability:

```python
SEEDS = [42, 123, 456]
```

Each seed controls: Python random, NumPy, PyTorch, CUDA, and DataLoader shuffling.

Results are reported as **mean ± standard deviation** across all three seeds.
The low standard deviation (e.g., ±0.0016 for clean accuracy) confirms stable and reproducible training.

---

## Adversarial Data Generation

Adversarial samples were generated using four **word-level perturbation** techniques applied to original training samples. All perturbations are **label-preserving** — aspect and sentiment labels do not change.

| Method | What It Does |
|---|---|
| Random Deletion | Removes each word with probability p=0.10 |
| Random Swap | Swaps N random word pairs (N = max(1, len÷8)) |
| Random Insertion | Duplicates and inserts N random words (N = max(1, len÷10)) |
| Mixed | Applies deletion (p=0.08) then swap (N=1) together |

To generate adversarial data yourself:
```bash
python src/adversarial_generation.py
```


---

## Training Details

| Hyperparameter | Baseline / mBERT | RL-BanglaBERT |
|---|---|---|
| Base model | BanglaBERT / mBERT | BanglaBERT |
| Epochs | 10 / 5 | 20 |
| Learning rate | 2e-5 | 5e-6 |
| Batch size | 32 | 32 |
| PPO clip ε | — | 0.1 |
| KL coefficient λ | — | 0.4 |
| Clean scale | — | 1.8 |
| Adversarial scale | — | 1.2 |
| Clean acc floor | — | 0.9254 |
| Best checkpoint | Epoch with best val F1 | Epoch 15 (combined score 1.2518) |

---
## Requirements

```
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

Install all with:
```bash
pip install -r requirements.txt
```

---


## Pre-trained Models Used

| Model | Source | Purpose |
|---|---|---|
| BanglaBERT | [sagorsarker/bangla-bert-base](https://huggingface.co/sagorsarker/bangla-bert-base) | Backbone for Baseline and RL model |
| mBERT | [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased) | Multilingual comparison baseline |

Both models are downloaded automatically via HuggingFace `transformers` when the notebook runs.

---
## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

*Last updated: March 2026*
