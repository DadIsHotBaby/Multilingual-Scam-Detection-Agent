import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch
import os

# === Step 1: Load the Preprocessed Dataset ===
data_path = r"C:\Users\KIIT0001\Desktop\Scam Data\final_combined_dataset.csv"
df_all = pd.read_csv(data_path)
df_all = df_all.dropna().reset_index(drop=True)

# === Step 2: Split into Train and Test Sets ===
X_train, X_test, y_train, y_test = train_test_split(
    df_all["text"].tolist(),
    df_all["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df_all["label"]
)

# === Step 3: Load Tokenizer ===
model_name = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# === Step 4: Tokenize the Text ===
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)

# === Step 5: Prepare Torch Datasets ===
train_dataset = {
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": torch.tensor(y_train)
}

test_dataset = {
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "labels": torch.tensor(y_test)
}

# === Step 6: Save Datasets ===
output_dir = r"C:\Users\KIIT0001\Desktop\Scam Data\tokenized_data"
os.makedirs(output_dir, exist_ok=True)

torch.save(train_dataset, os.path.join(output_dir, "train_dataset.pt"))
torch.save(test_dataset, os.path.join(output_dir, "test_dataset.pt"))

# === Step 7: Save Tokenizer ===
tokenizer.save_pretrained(output_dir)

print("✅ Tokenization complete. Datasets and tokenizer saved.")
