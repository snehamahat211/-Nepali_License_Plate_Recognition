import tensorflow as tf

class CharPRF1(tf.keras.metrics.Metric):
    def __init__(self, pad=-1, name="char_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pad = pad
        self.tp = self.add_weight(name="tp", shape=(), initializer="zeros")
        self.fp = self.add_weight(name="fp", shape=(), initializer="zeros")
        self.fn = self.add_weight(name="fn", shape=(), initializer="zeros")


    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_ids = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred_ids, [-1])

        y_true_flat = tf.cast(y_true_flat, tf.int32)


        mask = tf.not_equal(y_true_flat, -1)   
        y_true_valid = tf.boolean_mask(y_true_flat, mask)
        y_pred_valid = tf.boolean_mask(y_pred_flat, mask)

        matches = tf.cast(tf.equal(y_true_valid, y_pred_valid), tf.float32)
        self.tp.assign_add(tf.reduce_sum(matches))
        self.fp.assign_add(tf.reduce_sum(1.0 - matches))
        self.fn.assign_add(tf.reduce_sum(1.0 - matches))

    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall    = self.tp / (self.tp + self.fn + 1e-8)
        return 2 * precision * recall / (precision + recall + 1e-8)

    def reset_states(self):         
        for var in (self.tp, self.fp, self.fn):
            var.assign(0.0)
