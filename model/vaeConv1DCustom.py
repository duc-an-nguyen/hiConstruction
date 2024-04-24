import os, warnings, sys
import tensorflow                   as tf
from re                             import T
from tensorflow.keras.layers        import Conv1D,  Flatten, Dense, Conv1DTranspose, Reshape, Input
from tensorflow.keras.models        import Model
from tensorflow.keras.backend       import random_normal
from tensorflow.keras.optimizers    import Adam

from utility.utils                  import get_mnist_data, draw_orig_and_post_pred_sample, plot_latent_space
from model.vae_base                 import BaseVariationalAutoencoder, Sampling 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 



# ================================
# This class is used to create 
# variantial auto encoder decoder
# model
# ================================
class vaeConv1DCustom(BaseVariationalAutoencoder):
    def __init__(self,  config, **kwargs ):   
        

        super(vaeConv1DCustom, self).__init__(config,**kwargs)

        self.encLayer           = config["encLayer"]
        self.decLayer           = config["decLayer"]
        
        self.encoder            = self._get_encoder()
        self.decoder            = self._get_decoder() 

    # =====================
    # Encoder side
    # ---------------------
    def _get_encoder(self):
        encoder_inputs  = Input(shape=(self.seq_len, self.inFeatureDim), name='encoder_input')
        x               = encoder_inputs
        for i, num_filters in enumerate(self.encLayer):
            x = Conv1D( filters         = num_filters, 
                        kernel_size     = 3, 
                        strides         = 2, 
                        activation      = 'relu', 
                        padding         = 'same',
                        name            = f'enc_conv_{i}')(x)

        x = Flatten(name='enc_flatten')(x)

        # ----------------------------------
        # save the dimensionality of this last
        #  dense layer before the latent state 
        # layer. We need it in the decoder. (because
        # after flatten, we don't know the dimention)
        # --------------------------
        self.encoder_last_dense_dim = x.get_shape()[-1]        

        # --------------------------
        # Create latent space 
        # --------------------------
        z_mean              = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var           = Dense(self.latent_dim, name="z_log_var")(x)
        encoder_output      = Sampling()([z_mean, z_log_var])     
        self.encoder_output = encoder_output
        
        encoder = Model(encoder_inputs, [z_mean, z_log_var, encoder_output], name="encoder")
        # encoder.summary()
        return encoder

    # =====================
    # Decoder side
    # ---------------------
    def _get_decoder(self):
        decoder_inputs  = Input(shape=(self.latent_dim), name='decoder_input')
        x               = decoder_inputs
        x               = Dense(self.encoder_last_dense_dim, name = "dec_dense", activation = 'relu')(x)           
        x               = Reshape(target_shape = (-1, self.decLayer[0]), name = "dec_reshape")(x)
                                    
        for i, num_filters in enumerate(reversed(self.decLayer[1:])):
            x = Conv1DTranspose(    filters     = num_filters, 
                                    kernel_size = 3, 
                                    strides     = 2, 
                                    padding     = 'same',
                                    activation  = 'relu', 
                                    name        = f'dec_deconv_{i}')(x)
        # ---------------------
        # last de-convolution
        # ---------------------
        x = Conv1DTranspose(filters     = self.outFeatureDim, 
                            kernel_size = 3, 
                            strides     = 2, 
                            padding     = 'same',
                            activation  = 'relu', 
                            name        = f'dec_deconv__{i+1}')(x)
                    
        x                       = Flatten(name = 'dec_flatten')(x)
        x                       = Dense(self.seq_len * self.outFeatureDim, name= "decoder_dense_final")(x)
        self.decoder_outputs    = Reshape(target_shape=(self.seq_len, self.outFeatureDim))(x)
        decoder                 = Model(decoder_inputs, self.decoder_outputs, name="decoder")
        return decoder



# ================================
#       TEST FUNCTION
# ================================

if __name__ == '__main__':

    mnist_digits = get_mnist_data()
    print('data shape:', mnist_digits.shape)
    N, T, D = mnist_digits.shape


    vae = vaeConv1DCustom(
        seq_len=T,
        feat_dim=D,
        latent_dim = 2,
        hidden_layer_sizes=[100, 200],
    )

    vae.compile(optimizer=Adam())
    # vae.summary() ; sys.exit()

    
    r = vae.fit(mnist_digits, epochs=3, batch_size=64, shuffle=True)


    x_decoded = vae.predict(mnist_digits)
    print('x_decoded.shape', x_decoded.shape)


    # compare original and posterior predictive (reconstructed) samples
    draw_orig_and_post_pred_sample(mnist_digits, x_decoded, n=5)


    # generate prior predictive samples by sampling from latent space
    plot_latent_space(vae, 30, figsize=15)