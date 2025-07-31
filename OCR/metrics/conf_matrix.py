import io, itertools
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix

class ConfMatrixCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch} ended. Generating confusion matrix.")

    def __init__(self, val_data, charset, pad=-1, log_dir="logs"):
        super().__init__()
        self.val_data = val_data
        self.charset = list(charset) + ["<blank>"]
        self.pad = pad
        self.file_writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        all_pred, all_true = [], []
        for i, batch in enumerate(self.val_data):
            if i >= 20:
                break
        x, y = batch
        logits = self.model(x, training=False)  
        preds = tf.argmax(logits, axis=-1)

        preds = preds[:, :y.shape[1]]

        mask  = tf.not_equal(y, self.pad)  

        all_true.extend(tf.boolean_mask(y, mask).numpy())
        all_pred.extend(tf.boolean_mask(preds, mask).numpy())


        cm = confusion_matrix(all_true, all_pred, labels=range(len(self.charset)))
        fig = self.plot_confusion_matrix(cm)
        buf = io.BytesIO()
        plt.savefig(buf, format='png'); plt.close(fig)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        with self.file_writer.as_default():
            tf.summary.image("ConfusionMatrix", image[None], step=epoch)

    def plot_confusion_matrix(self, cm):
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(cm, interpolation='nearest')
        ax.set_xticks(range(len(self.charset)))
        ax.set_yticks(range(len(self.charset)))
        ax.set_xticklabels(self.charset, rotation=90, fontsize=6)
        ax.set_yticklabels(self.charset, fontsize=6)
        ax.set_ylabel('True'); ax.set_xlabel('Pred')
        ax.set_title('Character Confusion Matrix')
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if cm[i, j] > 0:
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=4)
        fig.tight_layout(); return fig
