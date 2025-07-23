from keras import layers
from keras.models import Model

from residual_block import residual_block

def train_model(input_dimen, output_dimen, *, dropout=0.25):

    h, w, _ = input_dimen
    inputs = layers.Input(shape=input_dimen, name="input")        

    input_norm = layers.Rescaling(1./255, offset=-0.5)(inputs)

    x1 = residual_block(input_norm, 32, strides=(1, 1), dropout=dropout)
    x2 = residual_block(x1, 32, strides=(2,1), dropout=dropout)
   
    x3 = residual_block(x2, 64, strides=(1, 1), dropout=dropout)
    x4 = residual_block(x3, 64, strides=(2,2), dropout=dropout)
   
    x5 = residual_block(x4, 128, strides=(1, 1), dropout=dropout)
    x6 = residual_block(x5, 128, strides=(2,2), dropout=dropout)
  
    x7 = residual_block(x6, 256, strides=(1, 1), dropout=dropout)
    x8 = residual_block(x7, 256, strides=(2,1), dropout=dropout)

    _, h_final, w_final, c_final = x8.shape
    assert h_final == 2

    squeezed = layers.Reshape((w_final, c_final * h_final))(x8)
   
    blstm1 = layers.Bidirectional(layers.LSTM(512, return_sequences=True, dropout=dropout))(squeezed)
    blstm2 = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=dropout))(blstm1)

    
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)(blstm2, blstm2)
    merged = layers.Concatenate()([blstm2, attention])

    dense = layers.Dense(128, activation='relu')(merged)
    output = layers.Dense(output_dimen + 1, name="output")(dense)
   
    model = Model(inputs=inputs, outputs=output)
    return model
