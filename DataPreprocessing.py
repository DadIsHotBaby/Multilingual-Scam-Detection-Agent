import os
import pandas as pd

# === Step 1: Convert TALLIP .txt files into a DataFrame ===
data = []
base_path = r"C:\Users\KIIT0001\Desktop\Scam Data\TALLIP-FakeNews-Dataset"

# Walk through the TALLIP Fake News folder
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)
            label = os.path.basename(root).lower()  # e.g., "fake" or "real"
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
                data.append({
                    "text": content,
                    "label": 1 if label == "fake" else 0  # 1 = fake, 0 = real
                })

# Create a DataFrame for TALLIP data
df3 = pd.DataFrame(data)

# Clean TALLIP dataset: remove irrelevant lines (headers, etc.)
df3 = df3[~df3["text"].str.startswith(("Domain", "This Dataset", "Label"))]
df3 = df3[df3["text"].notna()]
df3["text"] = df3["text"].str.strip()
df3.reset_index(drop=True, inplace=True)

# Save cleaned TALLIP dataset to CSV
df3_path = os.path.join(base_path, "data_file.csv")
df3.to_csv(df3_path, index=False)
print(f"✅ Created CSV with {len(df3)} rows at: {df3_path}")


# === Step 2: Load the other two datasets ===
df1 = pd.read_csv(r"C:\Users\KIIT0001\Desktop\Scam Data\data-en-hi-de-fr.csv")
df2 = pd.read_csv(r"C:\Users\KIIT0001\Desktop\Scam Data\sms_scam_detection_dataset_merged_with_lang.csv")

# === Step 3: Clean and standardize columns ===

# df1: multilingual message dataset
df1 = df1.rename(columns={"labels": "label", "text": "text_en"})
df1["label"] = df1["label"].map({"ham": 0, "spam": 1})
df1["text"] = df1["text_en"]  # use English for now
df1 = df1[["text", "label"]]

# df2: SMS + language dataset
df2 = df2.rename(columns={"label": "label", "text": "text"})
df2["label"] = df2["label"].map({"ham": 0, "spam": 1})
df2 = df2[["text", "label"]]

# df3: already cleaned and normalized (TALLIP data)
df3 = df3[["text", "label"]]

# === Step 4: Merge all datasets ===
df_all = pd.concat([df1, df2, df3], ignore_index=True)

# Drop nulls and reset index
df_all = df_all.dropna().reset_index(drop=True)

# === Step 5: Save the final merged dataset ===
final_dataset_path = r"C:\Users\KIIT0001\Desktop\Scam Data\final_combined_dataset.csv"
df_all.to_csv(final_dataset_path, index=False)

print(f"✅ Final merged dataset shape: {df_all.shape}")
print(f"✅ Final merged dataset saved at: {final_dataset_path}")
print(df_all.sample(5))  # Print sample rows for verification
