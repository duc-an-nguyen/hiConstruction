from keras.layers                   import Conv1D,  Flatten, Dense, Conv1DTranspose, Reshape, Input, Layer, GRU, LSTM, Concatenate, Embedding, MultiHeadAttention
from keras                          import backend as K
from keras.models                   import Model
from keras.optimizers               import Adam
from model.vaeBase                  import BaseVariationalAutoencoder, Sampling 
from re                             import T
from keras.metrics                  import Mean
from utility.logger                 import aiLogger
from scipy.stats                    import spearmanr
import  numpy                       as np
import  tensorflow                  as tf
import time
import  os, warnings, sys
import  joblib, random, json
class VAEConvInterpretable(BaseVariationalAutoencoder):    
    def __init__(   self,  
                    hidden_layer_sizes, 
                    config      = None,
                    **kwargs   ):
        '''
            hidden_layer_sizes  : list of number of filters in convolutional layers in encoder and residual connection of decoder. 
            trend_poly          : integer for number of orders for trend component. e.g. setting trend_poly = 2 will include linear and quadratic term. 
            num_gen_seas        : Number of sine-waves to use to model seasonalities. Each sine wae will have its own amplitude, frequency and phase. 
            custom_seas         : list of tuples of (num_seasons, len_per_season). 
            num_seasons         : number of seasons per cycle. 
            len_per_season      : number of epochs (time-steps) per season.
            use_residual_conn   : boolean value indicating whether to use a residual connection for reconstruction in addition to trend, generic and custom seasonalities.       
        '''

        super(VAEConvInterpretable, self).__init__(**kwargs)

        self.hidden_layer_sizes = hidden_layer_sizes
        self.encoder            = self.get_encoder()
        self.decoder            = self.get_decoder() 
        self.config             = config

    # ================================
    # Full AE
    # --------------------------------
    def getUnsymetricDecoder(self,kernelSize = 3, stride = 1): 
        z   = Input(shape=(self.latent_dim,), name="decoder_input")
        wIn = Input(shape=(self.seq_len,self.config["manipulationDim"]), name="manipulatedVar")

        x = tf.keras.layers.RepeatVector(self.seq_len,name='repeat')(z)
        x = tf.keras.layers.Concatenate(axis  =-1)([x, wIn])

        for numFilters in reversed(self.hidden_layer_sizes):
            x = tf.keras.layers.Conv1D(filters      =   numFilters,
                                       kernel_size  =   kernelSize,
                                       stride      =   stride,
                                       activation   =   'relu',
                                       padding = 'same')(x)
        
        reconsData  = tf.keras.layers.Dense(self.config["efficiencyDim"],name='X')(x)   
    
        decoder     = tf.keras.models.Model([z, wIn],reconsData)
        return decoder
    
    # ================================
    # Attention encoder
    # --------------------------------
    # def getAttentionEncoder(self, kernelSize = 3, stride = 2):
    #     encoder_inputs  = Input(shape   = (self.seq_len, self.feat_dim), 
    #                             name    = 'encoder_input')
    #     x               = encoder_inputs
    #     # --------------------
    #     # Repeat to create Conv1D
    #     # --------------------
    #     for i, num_filters in enumerate(self.hidden_layer_sizes):
    #         x = AttentionMaskedConv1D(  filters     = num_filters, 
    #                                     kernel_size = kernelSize, 
    #                                     strides     = stride, 
    #                                     activation  = 'relu', 
    #                                     padding     = 'same',
    #                                     name        = f'enc_conv_{i}')(x)
    #     sysHILayer = AttentionMaskedConv1D(     filters     = 5, 
    #                                             kernel_size = kernelSize, 
    #                                             strides     = stride, 
    #                                             activation  = 'relu', 
    #                                             padding     = 'same',
    #                                             name        = 'extractedSysHiLayer')(x)
        
    #     print("The shape of sysHILayer",sysHILayer.shape)
    #     # --------------------
    #     # Flat and create latent
    #     # --------------------
    #     x                       = Flatten(name='enc_flatten')(x)
    #     self.encLastDenseDim    = x.get_shape()[-1]                                 # save the dimensionality of this last dense layer before the hidden state layer. We need it in the decoder.
        
    #     # ----------------------
    #     # For VAE
    #     # ----------------------
    #     encoder_output          = Dense(self.latent_dim, name="encoder_output")(x)
    #     self.encoder_output     = encoder_output
        
    #     encoder                 = Model(encoder_inputs, [encoder_output, sysHILayer], name="encoder")
    #     return encoder
    # ================================
    # Create module Encoder separately
    # --------------------------------
    def get_encoder(self, kernelSize = 3, stride = 2):
        # --------------------
        # Input
        # --------------------
        encoder_inputs  = Input(shape   = (self.seq_len, self.feat_dim), 
                                name    = 'encoder_input')
        x               = encoder_inputs
        # --------------------
        # Repeat to create Conv1D
        # --------------------
        for i, num_filters in enumerate(self.hidden_layer_sizes):
            x = Conv1D( filters     = num_filters, 
                        kernel_size = kernelSize, 
                        strides     = stride, 
                        activation  = 'relu', 
                        padding     = 'same',
                        name        = f'enc_conv_{i}')(x)
        sysHILayer = Conv1D(    filters     = 5, 
                                kernel_size = kernelSize, 
                                strides     = stride, 
                                activation  = 'relu', 
                                padding     = 'same',
                                name        = 'extractedSysHiLayer')(x)
        print("The shape of sysHILayer",sysHILayer.shape)
        # --------------------
        # Flat and create latent
        # --------------------
        x                       = Flatten(name='enc_flatten')(x)
        self.encLastDenseDim    = x.get_shape()[-1]                                 # save the dimensionality of this last dense layer before the hidden state layer. We need it in the decoder.
        
        # ----------------------
        # For VAE
        # ----------------------
        encoder_output          = Dense(self.latent_dim, name="encoder_output")(x)
        self.encoder_output     = encoder_output
        
        encoder                 = Model(encoder_inputs, [encoder_output, sysHILayer], name="encoder")
        print(encoder.summary())
        return encoder

    # =================================
    # ---------------------------------
    def get_decoder(self, kernelSize = 3, stride = 2):
        decoder_inputs = Input(shape=(self.latent_dim,), name="decoder_input")

        x = decoder_inputs
        x = Dense(self.encLastDenseDim, name="dec_dense", activation="relu")(x)


        x = Reshape(target_shape=(-1, self.hidden_layer_sizes[-1]), name="dec_reshape")(x)
     

        for i, num_filters in enumerate(reversed(self.hidden_layer_sizes[:-1])):
            x = Conv1DTranspose(    filters     = num_filters,
                                    kernel_size = kernelSize,
                                    strides     = stride,
                                    padding     = "same",
                                    activation  = "relu",
                                    name        = f"dec_deconv_{i}",
                                )(x)
       

        # last de-convolution
        self.decoder_outputs = Conv1DTranspose(        filters     = self.feat_dim,
                                    kernel_size = kernelSize,
                                    strides     = stride,
                                    padding     = "same",
                                    activation  = "relu",
                                    name        = f"dec_deconv__{i+1}",
                                )(x)

        # x = Dense(self.config["efficiencyDim"],name='X')(x)   
        # x                       = Flatten(name="dec_flatten")(x)
        # x                       = Dense(self.seq_len * self.feat_dim, name="decoder_dense_final")(x)
        # self.decoder_outputs    = Reshape(target_shape=(self.seq_len, self.feat_dim))(x)
        decoder                 = Model(decoder_inputs, self.decoder_outputs, name="decoder")
        print(decoder.summary())
        return decoder

with open('utility/modelConfig.json', 'r') as configFile:
    config = json.load(configFile)
myConfig   = config["system"]

vae                     = VAEConvInterpretable( seq_len             = myConfig["seqLength"],  
                                                feat_dim            = myConfig["inFeatureDim"], 
                                                latent_dim          = myConfig["latentDim"], 
                                                hidden_layer_sizes  = myConfig["hiddenVAESize"], # [50,100,200]      #[80, 200, 250] 
                                                config              = myConfig
                                                )
        
vae.get_encoder(kernelSize = 3, stride = 2)
vae.get_decoder(kernelSize = 3, stride = 2)