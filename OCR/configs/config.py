import os
from datetime import datetime
from pathlib import Path
from mltu.configs import BaseModelConfigs


def _validate_vocab(chars: str) -> list[str]:
    unique_chars = sorted(set(chars))
    if " " not in unique_chars:
        unique_chars.append(" ")
    return unique_chars


class ModelConfiguration(BaseModelConfigs):
    def __init__(self):
        super().__init__()

        self._model_path = None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_path = Path("models/OCR") / f"nepali_lpr_{timestamp}"

        self.vocab = _validate_vocab(
            "०१२३४५६७८९"
            "अआइईउऊएऐओऔ"
            "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"
            "क्षत्र्ज्ञश्र"
            "्ािीुूृेैोौंःँ"
            " "
        )

        self.height = 32
        self.width = 256
        self.max_text_length = 15
        self.batch_size = self._get_optimal_batch_size()
        self.learning_rate = 1e-4
        self.train_epochs = 100
        self.train_workers = min(os.cpu_count(), 8)
        self.aug_rotate_limit = 7
        self.aug_blur_limit = 3

    @property
    def model_path(self) -> str:
        if getattr(self, "_model_path", None) is None:
            self._model_path = "models/OCR/default_model"
        return str(self._model_path)

    @model_path.setter
    def model_path(self, value):
        try:
            self._model_path = str(Path(value)) if value else "models/OCR/default_model"
            os.makedirs(self._model_path, exist_ok=True)
        except (TypeError, OSError):
            self._model_path = "models/OCR/fallback_model"
            os.makedirs(self._model_path, exist_ok=True)

    def _get_optimal_batch_size(self) -> int:
        try:
            img_count = len(os.listdir("../Datasets/images"))
            return 16 if img_count > 5000 else 8
        except Exception:
            return 8

    def serialize(self):
        clean = {}
        for k, v in self.__dict__.items():
            if callable(v):
                continue
            clean[k] = str(v) if isinstance(v, Path) else v
        return clean

    # ---------- save ----------
    def save(self, verbose: bool = True):
        os.makedirs(self.model_path, exist_ok=True)

        try:
            super().save()
            if verbose:
                print(f"Configuration saved to: {self.model_path}")
        except Exception as e:
            print(f"Failed to save config: {e}")
            print("Attempting backup location...")
            self.model_path = "models/OCR/backup_model"
            super().save()
            if verbose:
                print(f"Configuration saved to backup: {self.model_path}")
