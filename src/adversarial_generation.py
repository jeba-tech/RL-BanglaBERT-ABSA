
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import random

# ─── Config ───────
DATA_DIR = "/kaggle/working/data"
os.makedirs(DATA_DIR, exist_ok=True)

INPUT_TRAIN_CSV   = "/kaggle/input/datasets-bangla/train.csv" 
OUTPUT_ADV_CSV   = os.path.join(DATA_DIR, "adversarial_samples.csv")
OUTPUT_MIXED_CSV = os.path.join(DATA_DIR, "train_with_adversarial.csv")

TARGET_ADV_COUNT = 600
AUG_PER_EXAMPLE = 2

# ─── Label mappings ───────
aspect_labels = {
    "ambiance": 0,
    "cleanliness": 1,
    "food_quality": 2,
    "price": 3,
    "service": 4
}

sentiment_labels = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

# ─── Load data ───────
print("Loading original training data...")
df = pd.read_csv(INPUT_TRAIN_CSV, encoding='utf-8-sig')

print(f"Original size: {len(df):,} rows")
print("Sentiment distribution:")
print(df['sentiment'].value_counts(dropna=False))

# ─── Add IDs if not present ───────
if 'aspect_id' not in df.columns:
    df['aspect_id'] = df['aspect'].map(aspect_labels)
    print("✓ Added aspect_id column")

if 'sentiment_id' not in df.columns:
    df['sentiment_id'] = df['sentiment'].map(sentiment_labels)
    print("✓ Added sentiment_id column")

# Verify mappings
print("\nVerifying IDs:")
print(f"  Aspects mapped: {df['aspect_id'].notna().sum()}/{len(df)}")
print(f"  Sentiments mapped: {df['sentiment_id'].notna().sum()}/{len(df)}")

# ─── Augmentation functions ────────
def random_deletion(words, p=0.1):
    """Randomly delete words with probability p"""
    if len(words) == 1:
        return words
    new_words = [word for word in words if random.random() > p]
    return new_words if new_words else [random.choice(words)]

def random_swap(words, n=1):
    """Randomly swap n pairs of words"""
    new_words = words.copy()
    for _ in range(n):
        if len(new_words) < 2:
            return new_words
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words

def random_insertion(words, n=1):
    """Randomly insert n duplicate words"""
    new_words = words.copy()
    for _ in range(n):
        if not new_words:
            return words
        random_word = random.choice(new_words)
        random_idx = random.randint(0, len(new_words))
        new_words.insert(random_idx, random_word)
    return new_words

def augment_text(text, n_aug=2):
    """Generate n_aug augmented versions of text"""
    words = text.split()
    if len(words) < 3:
        return []
    
    augmented_texts = []
    
    for _ in range(n_aug):
        # Randomly choose augmentation method
        aug_type = random.choice(['delete', 'swap', 'insert', 'mixed'])
        
        if aug_type == 'delete':
            new_words = random_deletion(words, p=0.1)
        elif aug_type == 'swap':
            new_words = random_swap(words, n=max(1, len(words)//8))
        elif aug_type == 'insert':
            new_words = random_insertion(words, n=max(1, len(words)//10))
        else:  # mixed
            new_words = random_deletion(words, p=0.08)
            new_words = random_swap(new_words, n=1)
        
        augmented_text = ' '.join(new_words)
        if augmented_text != text:
            augmented_texts.append(augmented_text)
    
    return augmented_texts

# ─── Test augmentation ──────────
print("\nTesting augmentation...")
test_text = str(df.iloc[0]['text']).strip()
print(f"Original: {test_text}")

test_augs = augment_text(test_text, n_aug=2)
for i, aug in enumerate(test_augs, 1):
    print(f"Aug {i}: {aug}")

print("Augmentation ready!\n")

# ─── Generate adversarial examples ─────
print(f"Generating {TARGET_ADV_COUNT:,} adversarial samples...")

adversarial_records = []
sample_indices = np.random.choice(
    len(df),
    size=min(TARGET_ADV_COUNT // AUG_PER_EXAMPLE, len(df)),
    replace=False
)

for idx in tqdm(sample_indices, desc="Augmenting"):
    original_text = str(df.iloc[idx]['text']).strip()
    
    if len(original_text.split()) < 3:
        continue
    
    augmented_texts = augment_text(original_text, n_aug=AUG_PER_EXAMPLE)
    
    for aug_text in augmented_texts:
        adversarial_records.append({
            'review_id'     : f"adv_{idx}_{len(adversarial_records)}",
            'text'          : aug_text,
            'aspect'        : df.iloc[idx]['aspect'],
            'sentiment'     : df.iloc[idx]['sentiment'],
            'aspect_id'     : int(df.iloc[idx]['aspect_id']),      
            'sentiment_id'  : int(df.iloc[idx]['sentiment_id']),   
            'original_text' : original_text,
            'is_adversarial': True
        })

adv_df = pd.DataFrame(adversarial_records)

# ─── Save and mix ───────
if not adv_df.empty:
    adv_df.to_csv(OUTPUT_ADV_CSV, index=False, encoding='utf-8-sig')
    print(f"\nGenerated {len(adv_df):,} adversarial samples")
    
    print(f"\nAdversarial CSV columns: {adv_df.columns.tolist()}")
    
    # Mix into training data
    n_to_add = min(int(len(df) * 0.20), len(adv_df))
    add_df = adv_df.sample(n=n_to_add, random_state=42)
    
    train_mixed = pd.concat([df, add_df], ignore_index=True)
    train_mixed = train_mixed.sample(frac=1, random_state=42).reset_index(drop=True)
    train_mixed.to_csv(OUTPUT_MIXED_CSV, index=False, encoding='utf-8-sig')
    
    print(f"\nMixed training set: {len(train_mixed):,} rows")
    print(f"  Original: {len(df):,}")
    print(f"  Added: {len(add_df):,} ({len(add_df)/len(df)*100:.1f}%)")
    
    print(f"\nMixed CSV columns: {train_mixed.columns.tolist()}")
    
    print("\nSentiment distribution:")
    print(train_mixed['sentiment'].value_counts(normalize=True).round(3))
    
    # Verify IDs in mixed data
    print("\n--- ID Verification in Mixed Data ---")
    print(f"Aspect IDs present: {train_mixed['aspect_id'].notna().sum()}/{len(train_mixed)}")
    print(f"Sentiment IDs present: {train_mixed['sentiment_id'].notna().sum()}/{len(train_mixed)}")
    
    print("\nAspect ID distribution:")
    print(train_mixed['aspect_id'].value_counts().sort_index())
    
    print("\nSentiment ID distribution:")
    print(train_mixed['sentiment_id'].value_counts().sort_index())
    
    # Show examples with IDs
    print("\nSample augmentations with IDs:")
    for i in range(min(3, len(adv_df))):
        print(f"\n{'='*70}")
        print(f"Original: {adv_df.iloc[i]['original_text']}")
        print(f"Aug:      {adv_df.iloc[i]['text']}")
        print(f"Aspect:   {adv_df.iloc[i]['aspect']} (ID: {adv_df.iloc[i]['aspect_id']})")
        print(f"Sentiment: {adv_df.iloc[i]['sentiment']} (ID: {adv_df.iloc[i]['sentiment_id']})")

else:
    print("\nNo adversarial examples generated!")

print("\nFinished!")

# ─── Final verification ────────
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

if os.path.exists(OUTPUT_MIXED_CSV):
    final_df = pd.read_csv(OUTPUT_MIXED_CSV, encoding='utf-8-sig')
    print(f"\ntrain_with_adversarial.csv ready for training!")
    print(f"   Total rows: {len(final_df):,}")
    print(f"   Columns: {final_df.columns.tolist()}")
    print(f"\n   Sample row:")
    print(final_df.iloc[0][['text', 'aspect', 'aspect_id', 'sentiment', 'sentiment_id']])

['text', 'aspect', 'sentiment', 'aspect_id', 'sentiment_id', 'review_id', 'original_text', 'is_adversarial']