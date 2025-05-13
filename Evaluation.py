# === Evaluation.py ===
import os
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, Trainer
from sklearn.metrics import classification_report
import numpy as np
import wandb

# === W&B Initialization ===
os.environ["WANDB_MODE"] = "online"  # Ensure W&B is online
wandb.init(project="Multilingual-Scam-Detection", name="Evaluation-Run")

# === Path Configuration ===
BASE_DIR = r"C:\Users\KIIT0001\Desktop\Scam Data\tokenized_data"
TEST_PATH = Path(BASE_DIR) / "test_dataset.pt"
MODEL_DIR = Path(BASE_DIR).parent / "final_model"

# === Load Test Dataset ===
def load_dataset_safely(path):
    print(f"\n=== Loading Test Dataset ===\n")
    print(f"Loading dataset from: {path}")
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        # Load with safety checks
        torch.serialization.add_safe_globals([])
        data = torch.load(path, weights_only=False)
        print(f"✅ Test Dataset Loaded. Total Samples: {len(data['labels'])}")
        return data
    except Exception as e:
        print(f"\n❌ Failed to load test dataset: {str(e)}")
        raise

test_data = load_dataset_safely(TEST_PATH)

class ScamDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.input_ids = data['input_ids']
        self.attention_mask = data.get('attention_mask', [1]*len(self.input_ids))
        self.labels = data['labels']

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx]),
            'attention_mask': torch.tensor(self.attention_mask[idx]),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.labels)

test_dataset = ScamDataset(test_data)

# === Loading Fine-Tuned Model ===
print("\n=== Loading Fine-Tuned Model ===")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
print("✅ Model Loaded.")

# === Evaluating Model on Test Dataset ===
print("\n🚀 Evaluating Model on Test Dataset...")
trainer = Trainer(model=model)

# Generate predictions
preds = trainer.predict(test_dataset)
y_true = test_data['labels']
y_pred = np.argmax(preds.predictions, axis=1)

# === Generate Classification Report ===
print("\n🧪 Classification Report:")
report = classification_report(y_true, y_pred, target_names=["Not Scam", "Scam"])
print(report)

# === Save Classification Report with UTF-8 Encoding ===
REPORT_PATH = Path(BASE_DIR).parent / "evaluation_report.txt"
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("🧪 Classification Report:\n")
    f.write(report)

print(f"\n✅ Classification report saved at: {REPORT_PATH}")

# === Log Metrics to W&B ===
wandb.log({"Classification Report": report})
wandb.finish()
