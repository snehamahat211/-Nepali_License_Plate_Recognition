import os
import random

# Change this to your dataset folder path
data_path = "../Datasets"

images_folder = os.path.join(data_path, "images")
train_file = os.path.join(data_path, "annotation.train.txt")
val_file = os.path.join(data_path, "annotation.val.txt")

image_files = [os.path.join("images", f) for f in os.listdir(images_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

random.shuffle(image_files)

split_ratio = 0.8
split_index = int(len(image_files) * split_ratio)

train_images = image_files[:split_index]
val_images = image_files[split_index:]

with open(train_file, "w") as f:
    for item in train_images:
        f.write(item + "\n")

with open(val_file, "w") as f:
    for item in val_images:
        f.write(item + "\n")

print(f"Total images: {len(image_files)}")
print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
print("Train/Val split files created successfully!")
