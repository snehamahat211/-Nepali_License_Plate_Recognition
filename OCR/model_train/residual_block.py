from keras import layers
from keras.regularizers import l2


def residual_block(x: layers.Layer, filters: int, activation='leaky_relu', strides: tuple=(1,1), dropout: float = 0.25, l2_strength: float = 1e-4) -> layers.Layer:
    shortcut = x

    act = layers.LeakyReLU(0.1) if activation == "leaky_relu" else layers.Activation(activation)


    x = layers.Conv2D(filters, kernel_size=3, strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_strength), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = act(x)
    x = layers.SpatialDropout2D(dropout)(x)

    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_strength), use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if strides != (1, 1) or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=3, strides=strides, padding='same', kernel_initializer='he_normal', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Add()([x, shortcut])
        x = act(x)


    if filters >= 128:
        se = layers.GlobalAveragePooling2D()(x)
        se = layers.Dense(filters // 8, activation='relu')(se)
        se = layers.Dense(filters, activation='sigmoid')(se)
        se = layers.Reshape((1, 1, filters))(se)
        x = layers.Multiply()([x, se])


    return x
