# === FineTuning.py ===
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# === Path Configuration ===
BASE_DIR = r"C:\Users\KIIT0001\Desktop\Scam Data\tokenized_data"
TRAIN_PATH = Path(BASE_DIR) / "train_dataset.pt"
TEST_PATH = Path(BASE_DIR) / "test_dataset.pt"
OUTPUT_DIR = Path(BASE_DIR).parent / "final_model"

# === Universal Dataset Class ===
class ScamDataset(Dataset):
    def __init__(self, data):
        """Handles all possible dataset formats"""
        print(f"\nInitial dataset type: {type(data)}")
        if hasattr(data, '__dict__'):
            print("Dataset attributes:", vars(data).keys())

        if isinstance(data, dict):
            self._init_from_dict(data)
        elif hasattr(data, 'input_ids'):
            self._init_from_object(data)
        else:
            raise ValueError("Unsupported dataset format")

        self._validate_dataset()

    def _init_from_dict(self, data):
        """Initialize from dictionary format"""
        if 'input_ids' in data:
            self.input_ids = data['input_ids']
            self.attention_mask = data.get('attention_mask', [1]*len(self.input_ids))
            self.labels = data['labels']
        elif 'encodings' in data:
            self.input_ids = data['encodings']['input_ids']
            self.attention_mask = data['encodings'].get('attention_mask', [1]*len(self.input_ids))
            self.labels = data['labels']
        else:
            raise ValueError("Dictionary missing required keys")

    def _init_from_object(self, data):
        """Initialize from object format"""
        self.input_ids = data.input_ids
        self.attention_mask = getattr(data, 'attention_mask', [1]*len(self.input_ids))
        self.labels = data.labels

    def _validate_dataset(self):
        """Validate dataset structure"""
        required = ['input_ids', 'attention_mask', 'labels']
        for attr in required:
            if not hasattr(self, attr):
                raise AttributeError(f"Dataset missing required attribute: {attr}")
            if len(getattr(self, attr)) == 0:
                raise ValueError(f"Empty attribute: {attr}")

        lens = [len(getattr(self, attr)) for attr in required]
        if not all(l == lens[0] for l in lens):
            raise ValueError("All dataset fields must have same length")

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx]),
            'attention_mask': torch.tensor(self.attention_mask[idx]),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.labels)

# === Enhanced Dataset Loading ===
def load_dataset_safely(path):
    print(f"\nLoading dataset from: {path}")
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        torch.serialization.add_safe_globals([ScamDataset])
        data = torch.load(path, weights_only=False)

        print(f"Loaded data type: {type(data)}")
        if isinstance(data, dict):
            print("Dictionary keys:", data.keys())
            if 'encodings' in data:
                print("Encodings keys:", data['encodings'].keys())

        return ScamDataset(data)
    except Exception as e:
        print(f"Failed to load dataset: {str(e)}")
        raise

# === Main Execution ===
if __name__ == "__main__":
    print("=== Starting Scam Detection Fine-Tuning ===")

    try:
        print("\n" + "="*40)
        print("Loading TRAIN dataset...")
        train_dataset = load_dataset_safely(TRAIN_PATH)

        print("\n" + "="*40)
        print("Loading TEST dataset...")
        test_dataset = load_dataset_safely(TEST_PATH)

        print("\nTrain sample:", train_dataset[0])
        print("Test sample:", test_dataset[0])
    except Exception as e:
        print(f"\n❌ Failed to load datasets: {str(e)}")
        exit(1)

    # Model setup
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-multilingual-cased",
        num_labels=2
    )

    # Training configuration with checkpointing
    training_args = TrainingArguments(
        output_dir="./results",               # Directory to save checkpoints
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="steps",                # Save model periodically
        save_steps=500,                       # Save checkpoint every 500 steps
        save_total_limit=3,                   # Keep only the 3 most recent checkpoints
        resume_from_checkpoint=True           # Resume training if interrupted
    )

    # Training with automatic checkpoint resumption
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=lambda pred: {
            "accuracy": accuracy_score(pred.label_ids, pred.predictions.argmax(-1)),
            "precision": precision_recall_fscore_support(
                pred.label_ids, pred.predictions.argmax(-1), average='binary')[0]
        },
    )

    print("\n🚀 Starting training...")
    try:
        trainer.train(resume_from_checkpoint=True)
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        exit(1)

    # Save the final model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    print(f"\n✅ Model successfully saved to {OUTPUT_DIR}")
