import sys
import os
import numpy  as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# GPU Configuration
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

try:
    [tf.config.experimental.set_memory_growth(gpu, True)
     for gpu in tf.config.experimental.list_physical_devices("GPU")]
except:
    pass

from model_train.ctc_loss import CTCloss
from inferencemodel import demo_random_val_sample
from model_train.grayscalereader import GrayscaleImageReader
from model_train.model import train_model
from metrics.conf_matrix import ConfMatrixCallback
from metrics.char_prf1 import CharPRF1
from model_train.data_utils import decode_predictions, visualize_predictions


from model_train.np_image_resizer import NumpyImageResizer
from mltu.transformers import LabelIndexer, LabelPadding
from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.metrics import CWERMetric
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger

from configs.config import ModelConfiguration


import matplotlib
matplotlib.rcParams['font.family'] = 'Lohit Devanagari'

class DebugCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch} | Loss: {logs.get('loss')} | Val Loss: {logs.get('val_loss')}")
        if np.isnan(logs.get('loss')):
            print("Warning: Loss is NaN!")

class SamplePredictionCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data_provider, vocab):
        super().__init__()
        self.val_data_provider = val_data_provider
        self.vocab = vocab

    def on_epoch_end(self, epoch, logs=None):
        sample_batch = next(iter(self.val_data_provider))
        sample_images, sample_labels = sample_batch[0], sample_batch[1]
        predictions = self.model.predict(sample_images) 

        decoded = []
        blank_index = len(self.vocab)
        for pred in predictions:
            decoded.append(ctc_beam_search_decode(pred, self.vocab))


        true_texts = labels_to_texts(sample_labels, self.vocab)

        print(f"\nEpoch {epoch+1} Sample Predictions:")
        for t, p in zip(true_texts[:3], decoded[:3]):
            print(f" → True: {t}\n → Pred: {p}\n")

    
def plot_sample(image, true_text, pred_text):
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"True: {true_text}\nPredicted: {pred_text}")
    plt.axis('off')
    plt.show()




class NaNStoppingCallback(tf.keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            loss = logs.get('loss')
            if loss is not None and np.isnan(loss):
                print(f"NaN detected in loss at batch {batch}, stopping training.")
                self.model.stop_training = True

# Clean labels 
def clean_labels(data_path):
    labels_dir = os.path.join(data_path, "labels")
    for filename in os.listdir(labels_dir):
        if filename.endswith(".txt"):
            path = os.path.join(labels_dir, filename)
            with open(path, "r+", encoding="utf-8") as f:
                text = f.read().strip()
                cleaned = text.replace(" ", "")
                f.seek(0)
                f.write(cleaned)
                f.truncate()
    print("All labels cleaned! Removed spaces.")

def standardize_label(label):
    num_map = {'0':'०', '1':'१', '2':'२', '3':'३', '4':'४',
               '5':'५', '6':'६', '7':'७', '8':'८', '9':'९'}
    return ''.join([num_map.get(c, c) for c in label])


# Read Dataset and Extract Labels
def read_dataset(image_list_path, data_path, clean_labels_first=True):
    if clean_labels_first:
        clean_labels(data_path)
        
    dataset, vocab, max_len = [], set(), 0

    with open(image_list_path, "r", encoding="utf-8") as f:
        image_paths = [line.strip() for line in f]

    for img_rel_path in image_paths:
        img_path = os.path.join(data_path, img_rel_path)
        label_filename = os.path.basename(img_path).replace(".jpg", ".txt")
        label_path = os.path.join(data_path, "labels", label_filename)

        with open(label_path, "r", encoding="utf-8") as lf:
            label = lf.read().strip()
            label = standardize_label(label)  # standardize digits here
        
        if not label:
            print(f"Empty label in {label_path}")
            continue

        dataset.append([img_path, label])
        vocab.update(label)
        max_len = max(max_len, len(label))

    return dataset, sorted(vocab), max_len


def labels_to_texts(labels_batch, vocab):
    texts = []
    pad_token = len(vocab)
    for label_seq in labels_batch:
        text = ''.join([vocab[idx] for idx in label_seq if idx != pad_token])
        texts.append(text)
    return texts


def check_for_nans_and_infs(dataset):
    for img_path, _ in dataset:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print("Image not loaded correctly:", img_path)
            continue  
        
        img = img.astype("float32") / 255.0
        
        if np.isnan(img).any() or np.isinf(img).any():
            raise ValueError(f"NaN/Inf found in {img_path}")

def ctc_beam_search_decode(predictions, vocab, beam_width=10):
    import tensorflow as tf
    import numpy as np

    if predictions.ndim == 2:
        predictions = predictions[np.newaxis, ...]

    if np.max(predictions) > 1.0:
        predictions = tf.nn.softmax(predictions, axis=-1).numpy()

    input_length = np.array([predictions.shape[1]])

    decoded, _ = tf.keras.backend.ctc_decode(
        predictions,
        input_length=input_length,
        greedy=False,
        beam_width=beam_width,
        top_paths=1
    )

    decoded_indices = decoded[0].numpy()[0]

    # Default blank index is the last class
    blank_index = len(vocab)
    decoded_text = ''.join([vocab[idx] for idx in decoded_indices if idx != -1 and idx != blank_index])
    return decoded_text




def main():
    config = ModelConfiguration()

    # Defining paths
    data_path = "../Datasets"
    train_annotation_path = os.path.join(data_path, "annotation.train.txt")
    val_annotation_path = os.path.join(data_path, "annotation.val.txt")

    # Read Dataset
    print("Loading training data...")
    train_dataset, train_vocab, max_train_len = read_dataset(train_annotation_path, data_path)
    print("Loading validation data...")
    val_dataset, val_vocab, max_val_len = read_dataset(val_annotation_path, data_path)
    
    config.vocab = sorted(train_vocab)
    char_to_num = {c: i for i, c in enumerate(config.vocab)}

    def encode_label(label):
        return [char_to_num[c] for c in label]

    print(encode_label('ग१ख४५६२'))

    config.max_text_length = max(max_train_len, max_val_len)
    config.save()

    check_for_nans_and_infs(train_dataset)
    check_for_nans_and_infs(val_dataset)


    # Create data provider for model training
    train_data_provider = DataProvider(
        dataset=train_dataset,
        skip_validation=True,
        shuffle=True,
        batch_size=config.batch_size,
        data_preprocessors=[GrayscaleImageReader(visualize=True)],
        transformers=[
            NumpyImageResizer(config.width, config.height),
            LabelIndexer(config.vocab),
            LabelPadding(max_word_length=config.max_text_length, padding_value=len(config.vocab))
        ]
    )

    # Create data provider for model validation
    val_data_provider = DataProvider(
        dataset=val_dataset,
        skip_validation=True,
        batch_size=config.batch_size,
        data_preprocessors=[GrayscaleImageReader(visualize=True)],
        transformers=[
            NumpyImageResizer(config.width, config.height),
            LabelIndexer(config.vocab),
            LabelPadding(max_word_length=config.max_text_length, padding_value=len(config.vocab))
        ]
    )

    for i, (batch_x, batch_y) in enumerate(train_data_provider):
        if i >= 10:
            break
        if tf.math.reduce_any(tf.math.is_nan(batch_x)) or tf.math.reduce_any(tf.math.is_inf(batch_x)):
            raise ValueError("NaN or Inf detected in input images.")



    # Model Initialization
    model = train_model(
        input_dimen=(config.height, config.width, 1),
        output_dimen=len(config.vocab)
    )

    # Compile model
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate, clipnorm=1.0),
        loss=CTCloss(blank_index=len(config.vocab)),
        metrics=[
            CWERMetric(padding_token=-1),
            CharPRF1(pad=-1)
        ],  
        run_eagerly=False
    )

    model.summary(line_length=110)

    print("Max label length:", config.max_text_length)

    dummy_input = tf.random.uniform((1, config.height, config.width, 1))
    dummy_output = model(dummy_input)
    print("Model output shape:", dummy_output.shape) 
    print("Output sequence length (time steps):", dummy_output.shape[1])


    # Prepare output directory
    os.makedirs(config.model_path, exist_ok=True)

    conf_matrix_cb = ConfMatrixCallback(
    val_data=val_data_provider,
    charset=config.vocab,
    pad=len(config.vocab),
    log_dir=f"{config.model_path}/logs"
)

    # Define callbacks
    callbacks = [
    EarlyStopping(monitor="val_char_f1", patience=15, mode="max"), 
    ModelCheckpoint(f"{config.model_path}/model.h5", monitor="val_char_f1", save_best_only=True, mode="max"), conf_matrix_cb,  
    TrainLogger(config.model_path),
    TensorBoard(log_dir="/home/lamin/-Nepali_License_Plate_Recognition/OCR/model_train/logs", update_freq='epoch', write_graph=True), 
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6),
    Model2onnx(f"{config.model_path}/model.h5"),
    NaNStoppingCallback(),
    SamplePredictionCallback(val_data_provider, config.vocab)
]

    
    #print("\nSample Validation:")
    #for i in range(3):
        #img = cv2.imread(train_dataset[i][0])
        #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #plt.title(f"Label: {train_dataset[i][1]}")
        #plt.show()

    # Train the model

    for i, (images, labels) in enumerate(train_data_provider):
        print("Sample encoded labels:", labels[0])
        print("Decoded:", labels_to_texts([labels[0]], config.vocab))
        break


    print("\nStarting training...")
    try:
        model.fit(
            train_data_provider,
            validation_data=val_data_provider,
            epochs=config.train_epochs,
            callbacks=callbacks
        )
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        model.save(os.path.join(config.model_path, "interrupted_model.h5"))
        raise

    


    # Save final datasets
    train_data_provider.to_csv(os.path.join(config.model_path, "train.csv"))
    val_data_provider.to_csv(os.path.join(config.model_path, "val.csv"))

    print("\nTraining completed successfully!")

    demo_random_val_sample(model, val_dataset, config.vocab)


    # After training or in validation loop:
    batch = next(iter(val_data_provider))
    images, labels = batch

    predictions = model.predict(images)
    pred_texts = [ctc_beam_search_decode(pred, config.vocab) for pred in predictions]
    pred_probs = tf.nn.softmax(predictions, axis=-1) 
    print(pred_probs)
    

    # Debug output tokens
    pred_classes = np.argmax(predictions[0], axis=-1)
    blank_index = len(config.vocab)
    blank_count = np.sum(pred_classes == blank_index)
    print(f"Blank predictions: {blank_count}/{len(pred_classes)}")

    print("Predicted class indices:", pred_classes)

    true_texts = labels_to_texts(labels, config.vocab)

    visualize_predictions(images, true_texts, pred_texts, num=5)
    plot_sample(images[0], true_texts[0], pred_texts[0])

    softmax_probs = tf.nn.softmax(predictions[0], axis=-1).numpy()
    plt.figure(figsize=(15, 5))
    sns.heatmap(softmax_probs, cmap='viridis')
    plt.title("Softmax Heatmap")
    plt.xlabel("Classes")
    plt.ylabel("Timesteps")
    plt.show()

    for path, _ in val_dataset:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Missing or unreadable image: {path}")
        elif np.isnan(img).any():
            print(f"Image with NaNs: {path}")
        elif np.max(img) == 0:
            print(f"Image is completely black: {path}")


if __name__ == "__main__":
    main()

