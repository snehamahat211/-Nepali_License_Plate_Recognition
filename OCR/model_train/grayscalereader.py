import cv2
import numpy as np
from albumentations import Compose, RandomBrightnessContrast, GaussianBlur, ShiftScaleRotate
from mltu.preprocessors import ImageReader
import matplotlib.pyplot as plt

def get_augmenter():
    return Compose([
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        GaussianBlur(blur_limit=(1, 3), p=0.3),
        ShiftScaleRotate(rotate_limit=5, scale_limit=0.1, p=0.5),
    ])

class GrayscaleImageReader(ImageReader):
    def __init__(self, visualize=False):
        super().__init__(image_class=np.ndarray)
        self.augmenter = None
        self.visualize = visualize
        self.has_visualized = False  

    def __call__(self, image_path: str, annotation=None):
        color_img = cv2.imread(image_path)
        if color_img is None:
            raise ValueError(f"Could not read image at {image_path}")

        # Only apply augmentation if self.augmenter is not None
        if self.augmenter is not None:
            color_img = self.augmenter(image=color_img)['image']

        gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        gray_img = gray_img.astype(np.float32) / 255.0  
        gray_img = np.expand_dims(gray_img, axis=-1)

        if self.visualize and not self.has_visualized:
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
            plt.title("Augmented Color Image")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(gray_img.squeeze(), cmap="gray")
            plt.title("Grayscale Normalized")
            plt.axis("off")

            plt.suptitle(f"Preview: {image_path}")
            plt.show()

            self.has_visualized = True

        return gray_img, annotation
