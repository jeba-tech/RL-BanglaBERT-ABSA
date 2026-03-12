# Dataset Documentation
## Bangla Restaurant Review Dataset for Aspect-Based Sentiment Analysis (ABSA)

---

## Overview

This dataset contains **Bangla restaurant reviews** collected from online food delivery platforms and restaurant review sites in Bangladesh. Each review is annotated with an **aspect category** and a **sentiment polarity** for use in Aspect-Based Sentiment Analysis (ABSA) research.


This dataset was created to support the paper:

> **"Reinforcement Learning-Augmented Fine-Tuning for Aspect-Based Sentiment Analysis in Bangla Using Pre-Trained Language Models"**  
> [Jeba Fawjia], [2025] — Under Review

---

## Files in This Folder

```
data/
├── README_data.md                       - This file
├── ABSA_Cleaned_Original_Raw_Data.csv   - Original 4,028 Raw data 
└── processed_data/ 
     └── ├── label_mappings.json                  - Aspect and sentiment ID mappings
         ├── train.csv                            - 2,819 training samples (clean)
         ├── val.csv                              - 604 validation samples
         ├── test.csv                             - 605 test samples

└── adversarial_data/
    └── train_with_adversarial.csv       - 3,339 mixed samples (clean + adversarial)
```

---


## Dataset Statistics

| Split | Samples | Description |
|---|---|---|
| Train | 2,819 | Clean training samples |
| Validation | 604 | Clean validation samples |
| Test | 605 | Clean test samples |
| Adversarial | 520 | Word-level perturbed samples |
| **Mixed Train** | **3,339** | Clean + 20% adversarial (used for RL-BanglaBERT) |

---


## Column Descriptions

### train.csv / val.csv / test.csv

| Column | Type | Description |
|---|---|---|
| `review_id` | int | Unique identifier for each review |
| `text` | string | Original Bangla review text |
| `aspect` | string | Aspect category label (text) |
| `sentiment` | string | Sentiment polarity label (text) |
| `aspect_id` | int | Numeric ID for aspect (0–4) |
| `sentiment_id` | int | Numeric ID for sentiment (0–2) |
| `complexity` | string | Sentence complexity: simple / medium / complex |

### train_with_adversarial.csv (additional columns)

| Column | Type | Description |
|---|---|---|
| `original_text` | string | Original text before perturbation (NaN for clean samples) |
| `is_adversarial` | bool | True if sample is adversarially generated, False otherwise |

---

## Label Mappings

### Aspect Categories (5 classes)

| Label | ID | Description |
|---|---|---|
| `ambiance` | 0 | Atmosphere, decoration, environment of the restaurant |
| `cleanliness` | 1 | Hygiene and cleanliness of the restaurant |
| `food_quality` | 2 | Taste, quality, and presentation of food |
| `price` | 3 | Value for money, pricing, affordability |
| `service` | 4 | Staff behavior, speed of service, hospitality |

### Sentiment Polarities (3 classes)

| Label | ID | Description |
|---|---|---|
| `negative` | 0 | Negative opinion expressed toward the aspect |
| `neutral` | 1 | Neutral or mixed opinion toward the aspect |
| `positive` | 2 | Positive opinion expressed toward the aspect |

### Complexity Levels

| Level | Description | Example |
|---|---|---|
| `simple` | Short, direct single-clause sentence | "সেবা ভালো ছিল।" (The service was good.) |
| `medium` | Moderate length, single main clause | "সব মিলিয়ে খুব ভালো, মূল্য চমৎকার।" |
| `complex` | Multi-clause or conditional sentence | "আশা করেছিলাম ভালো হবে কিন্তু খাবারের মান ছিল বিরক্তিকর।" |

---

## Sample Data

### Clean Samples

| review_id | text | aspect | sentiment | complexity |
|---|---|---|---|---|
| 1425 | এখানকার সেবা বেশ চমৎকার ও উপভোগ্য। | service | positive | simple |
| 3468 | আশা করেছিলাম ভালো হবে কিন্তু অভিজ্ঞ শেফ থাকা সত্ত্বেও খাবারের মান ছিল বিরক্তিকর। | food_quality | negative | complex |
| 2164 | আমরা পদগুলো প্রশংসনীয় পেয়েছি এবং সরাসরি চলে গিয়েছিলাম। | food_quality | positive | medium |
| 408 | সেবার মান নিম্নমানের ছিল। | service | negative | simple |
| 3222 | সব মিলিয়ে খুব ভালো লেগেছে এবং কর্মচারীদের ব্যবহার ছিল প্রশংসনীয়। | service | positive | medium |

### Adversarial Samples (word-level perturbations)

| review_id | text (perturbed) | original_text | aspect | sentiment | is_adversarial |
|---|---|---|---|---|---|
| adv_0_1 | এখানকার চমৎকার সেবা বেশ ও উপভোগ্য। | এখানকার সেবা বেশ চমৎকার ও উপভোগ্য। | service | positive | True |
| adv_1_3 | আশা করেছিলাম ভালো কিন্তু খাবারের মান ছিল বিরক্তিকর। | আশা করেছিলাম ভালো হবে কিন্তু খাবারের মান ছিল বিরক্তিকর। | food_quality | negative | True |

---

## Adversarial Data Generation

The adversarial samples were generated programmatically using **four word-level perturbation techniques**. All perturbations are **label-preserving** — the aspect and sentiment labels remain identical to the original sample.

| Method | Description | Parameters |
|---|---|---|
| **Random Deletion** | Each word deleted with probability p | p = 0.10 |
| **Random Swap** | N random pairs of words swapped | N = max(1, len ÷ 8) |
| **Random Insertion** | N words duplicated and inserted randomly | N = max(1, len ÷ 10) |
| **Mixed** | Random Deletion followed by Random Swap | p = 0.08, then 1 swap |

The generation script is located at `src/adversarial_generation.py`.

Key parameters used:
```
TARGET_ADV_COUNT = 600
AUG_PER_EXAMPLE  = 2
AUGMENTATION_RATIO_IN_TRAINING = 20%  (520 of 3,339 total)
```

---

## Data Collection

- **Source:** Online restaurant review platforms and food delivery apps in Bangladesh
- **Language:** Bangla (Bengali script)
- **Domain:** Restaurant reviews (food, service, price, ambiance, cleanliness)
- **Annotation:** Manual annotation by the authors
- **Collection period:** 2025


---

## Important Notes for Reproducibility

1. All CSV files use **UTF-8-sig encoding** to correctly handle Bangla Unicode characters
2. The `test.csv` and adversarial samples were **never used during training** for Baseline and mBERT models
3. For RL-BanglaBERT, only `train_with_adversarial.csv` was used for training — not the raw adversarial file
4. The adversarial test evaluation uses a **separate 520-sample held-out set** from the adversarial data
5. Label distributions are **approximately balanced** across all splits

---

## Citation

If you use this dataset in your research, please cite:

```bibtex
@article{fawjia2025rlbanglabert,
  title   = {Reinforcement Learning-Augmented Fine-Tuning for 
             Aspect-Based Sentiment Analysis in Bangla Using 
             Pre-Trained Language Models},
  author  = {Fawjia, Jeba},
  journal = {Under Review at Neurocomputing},
  year    = {2025},
  doi     = {10.5281/zenodo.18986512},
  url     = {https://doi.org/10.5281/zenodo.18986512}
}
```

---

## License

This dataset is released under the **MIT License**.  
See the [LICENSE](../LICENSE) file in the root of the repository.

