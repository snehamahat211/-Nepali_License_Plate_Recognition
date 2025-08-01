import tensorflow as tf

class CTCloss(tf.keras.losses.Loss):
    def __init__(self, blank_index=-1, reduction=tf.keras.losses.Reduction.AUTO, name='CTCLoss'):
        super().__init__(reduction=reduction, name=name)
        self.blank_index = blank_index

    def call(self, y_true, y_pred):
        batch_size = tf.shape(y_true)[0]
        time_steps = tf.shape(y_pred)[1]

        input_length = tf.fill([batch_size], tf.cast(time_steps, tf.int32))
        
        # Convert padded tokens (blank_index) to -1 for tf.nn.ctc_loss
        labels = tf.cast(y_true, tf.int32)
        labels = tf.where(labels == self.blank_index, -1, labels)
        
        label_length = tf.math.count_nonzero(tf.cast(labels != -1, tf.int32), axis=1)

        loss = tf.nn.ctc_loss(
            labels=labels,
            logits=y_pred,
            label_length=label_length,
            logit_length=input_length,
            logits_time_major=False,
            blank_index=self.blank_index
        )

        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            "blank_index": self.blank_index
        })
        return config
