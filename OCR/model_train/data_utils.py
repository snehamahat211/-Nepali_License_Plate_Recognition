import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from albumentations import Compose, RandomBrightnessContrast, GaussianBlur, ShiftScaleRotate
import tensorflow as tf

import matplotlib
matplotlib.rcParams['font.family'] = 'Lohit Devanagari'


def clean_labels(data_path):
    labels_dir = os.path.join(data_path, "labels")
    for filename in os.listdir(labels_dir):
        if filename.endswith(".txt"):
            path = os.path.join(labels_dir, filename)
            with open(path, "r+", encoding="utf-8") as f:
                text = f.read().strip().replace(" ", "")
                f.seek(0)
                f.write(text)
                f.truncate()

def standardize_label(label):
    num_map = {'0':'०', '1':'१', '2':'२', '3':'३', '4':'४',
               '5':'५', '6':'६', '7':'७', '8':'८', '9':'९'}
    return ''.join([num_map.get(c, c) for c in label])

def get_augmenter():
    return Compose([
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        GaussianBlur(blur_limit=(1,3)),
        ShiftScaleRotate(rotate_limit=5, scale_limit=0.1)
    ])

def validate_dataset(dataset, num_samples=3):
    for img_path, label in dataset[:num_samples]:
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(f"Label: {label}")
        plt.axis('off')
        plt.show()

def decode_predictions(preds, vocab):
    input_length = np.ones(preds.shape[0]) * preds.shape[1]
    decoded, _ = tf.keras.backend.ctc_decode(preds, input_length=input_length, greedy=True)
    decoded_sequences = decoded[0].numpy()

    texts = []
    for seq in decoded_sequences:
        text = ''.join([vocab[c] if c < len(vocab) else '' for c in seq])
        texts.append(text)
    return texts

def visualize_predictions(images, true_texts, pred_texts, num=5):
    for i in range(num):
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"True: {true_texts[i]}\nPredicted: {pred_texts[i]}")
        plt.axis('off')
        plt.show()
