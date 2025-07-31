import os
import regex
from tqdm import tqdm

def normalize_label(label):
    return regex.sub(r'[^\p{L}\p{N}]', '', label)

def process_all_labels(data_path):
    labels_dir = os.path.join(data_path, "labels")
    for filename in tqdm(os.listdir(labels_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(labels_dir, filename)
            with open(filepath, "r+", encoding="utf-8") as f:
                original = f.read().strip()
                normalized = normalize_label(original)
                f.seek(0)
                f.write(normalized)
                f.truncate()

if __name__ == "__main__":
    data_path = "../Datasets" 
    process_all_labels(data_path)
    print("All labels normalized successfully!")