from keras import layers
from keras.models import Model
import tensorflow as tf

from residual_block import residual_block

def train_model(input_dimen, vocab_size, *, dropout=0.15):
    h, w, _ = input_dimen
    inputs = layers.Input(shape=input_dimen, name="input")        

    input_norm = inputs

    x1 = residual_block(input_norm, 8, strides=(1, 1), dropout=dropout)
    x2 = residual_block(x1, 8, strides=(2,1), dropout=dropout)
   
    x3 = residual_block(x2, 16, strides=(1, 1), dropout=dropout)
    x4 = residual_block(x3, 16, strides=(2,2), dropout=dropout)
   
    x5 = residual_block(x4, 32, strides=(1, 1), dropout=dropout)
    x6 = residual_block(x5, 32, strides=(2,2), dropout=dropout)
  
    x7 = residual_block(x6, 64, strides=(1, 1), dropout=dropout)
    x8 = residual_block(x7, 64, strides=(2,1), dropout=dropout)

    x9 = residual_block(x8, 128, strides=(1, 1), dropout=dropout)
    x10 = residual_block(x9, 128, strides=(1,1), dropout=dropout)

    h_final, w_final, c_final = tf.keras.backend.int_shape(x10)[1:]
    assert h_final == 2, f"Expected height 2, got {h_final}"

    squeezed = layers.Reshape((w_final, c_final * h_final))(x10)
   
    blstm1 = layers.Bidirectional(layers.LSTM(32, return_sequences=True, dropout=dropout))(squeezed)
    blstm2 = layers.Bidirectional(layers.LSTM(32, return_sequences=True, dropout=dropout))(blstm1)

    #attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)(blstm2, blstm2)
    #merged = layers.Concatenate()([blstm2, attention])

    merged = blstm2

    dense = layers.Dense(32, activation='relu')(merged)
    output = layers.Dense(vocab_size + 1, activation=None, name='output')(dense)

    model = Model(inputs=inputs, outputs=output)
    return model

