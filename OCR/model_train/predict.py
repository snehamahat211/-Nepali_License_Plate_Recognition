import tensorflow as tf
import os
from .inferencemodel import predict_single_image
from configs.config import ModelConfiguration
from .ctc_loss import CTCloss
from metrics.char_prf1 import CharPRF1
from mltu.tensorflow.metrics import CWERMetric

CONFIG_PATH = "model_train/models/OCR/nepali_lpr_20250731_192208/configs.yaml"
MODEL_PATH = "model_train/models/OCR/nepali_lpr_20250731_192208/model.h5"
IMAGE_PATH = "./Datasets/img.png"

print("Config exists?", os.path.exists(CONFIG_PATH))
print("Model exists?", os.path.exists(MODEL_PATH))
print("Image exists?", os.path.exists(IMAGE_PATH))

config = ModelConfiguration.load(CONFIG_PATH)

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "CTCloss": CTCloss(blank_index=len(config.vocab)),
        "CharPRF1": CharPRF1(pad=-1),
        "CWERMetric": lambda **kwargs: CWERMetric(padding_token=-1),
    }
)

predict_single_image(model, IMAGE_PATH, config.vocab, width=config.width, height=config.height)
