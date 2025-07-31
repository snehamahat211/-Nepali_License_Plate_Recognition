import numpy as np
import cv2
from mltu.transformers import Transformer

class NumpyImageResizer(Transformer):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, image: np.ndarray, label=None):
        resized = cv2.resize(image, (self.width, self.height))
        
        if resized.ndim == 2:
            resized = np.expand_dims(resized, axis=-1)

        return resized, label
