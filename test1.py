import tensorflow as tf
from tensorflow.keras import layers as L

# Encoder
enc_input = L.Input(shape=(61, 40))  # Input: sequence length=61, features=40
enc0 = L.Conv1D(filters=64, kernel_size=5, strides=2, padding='same', activation='relu')(enc_input)  # 61 -> 31
enc1 = L.Conv1D(filters=128, kernel_size=4, strides=2, padding='same', activation='relu')(enc0)      # 31 -> 16
enc2 = L.Conv1D(filters=256, kernel_size=3, strides=2, padding='same', activation='relu')(enc1)      # 16 -> 8

enc_flat = L.Flatten()(enc2)        # Flatten to connect to dense layer
enc_dense = L.Dense(256, activation='tanh')(enc_flat)  # Hidden dense layer
enc_out = L.Dense(1, activation='linear')(enc_dense)   # Latent space: 1 dimension
encoder = tf.keras.Model(inputs=enc_input, outputs=enc_out, name='Encoder')

# Decoder
dec_input = L.Input(shape=(1,))
    
# Match flattened shape from the encoder
dec_dense1 = L.Dense(256, activation='tanh')(dec_input)
dec_dense2 = L.Dense(8 * 256, activation='relu')(dec_dense1)  # Match the encoder's flattened size
dec_reshape = L.Reshape((8, 256))(dec_dense2)

# Symmetric Conv1DTranspose layers for upsampling
dec2 = L.Conv1DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation='relu')(dec_reshape)  # 8 -> 16
dec1 = L.Conv1DTranspose(filters=64, kernel_size=4, strides=2, padding='same', activation='relu')(dec2)          # 16 -> 32
dec0 = L.Conv1DTranspose(filters=40, kernel_size=3, strides=2, padding='valid', activation='linear')(dec1)       # 32 -> 61

decoder = tf.keras.Model(inputs=dec_input, outputs=dec0, name='Decoder')

# Summaries
encoder.summary()
decoder.summary()


# Combine encoder and decoder into an autoencoder
autoencoder_input = enc_input
autoencoder_output = decoder(encoder(autoencoder_input))
autoencoder = tf.keras.Model(inputs=autoencoder_input, outputs=autoencoder_output, name="Autoencoder")

# Summary
autoencoder.summary()
