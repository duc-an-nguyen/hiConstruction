from model.vae_conv_I_model   import VariationalAutoencoderConvInterpretable as TimeVAE
from utility.config     import rulVAECNNConfig
import joblib 
import os
import tensorflow       as tf
from tensorflow.keras.layers    import Conv1D,  Flatten, Dense, Conv1DTranspose, Reshape, Input, Layer,InputLayer
from tensorflow.keras.layers    import MultiHeadAttention,AveragePooling1D
from tensorflow.keras.models    import Model 
from utility.logger             import aiLogger


class systemHiAndRul(Model):
    def __init__(self,
                 config             = rulVAECNNConfig(),
                 pretrainedEncoder  = True,
                 freezeEncoder      = True):
        
        super(systemHiAndRul, self).__init__()

        self.config     = config

        # ---------------------------------------
        # Encoder model
        # ---------------------------------------
        self.reacInput   =   InputLayer(input_shape=(None,128,7))
        reacVae = TimeVAE(  seq_len             = config.seqLength,
                            feat_dim            = config.reacInput, 
                            latent_dim          = config.latentDim, 
                            hidden_layer_sizes  = config.hiddenSize, # [50,100,200]      #[80, 200, 250] 
                            reconstruction_wt   = 3.0,
                            use_residual_conn   = True
                        )  
        self.encReac =  reacVae._get_encoder()
        
        self.sepInput   =   InputLayer(input_shape=(None,128,5))
        sepVae = TimeVAE(   seq_len             = config.seqLength,
                            feat_dim            = config.sepInput, 
                            latent_dim          = config.latentDim, 
                            hidden_layer_sizes  = config.hiddenSize, # [50,100,200]      #[80, 200, 250] 
                            reconstruction_wt   = 3.0,
                            use_residual_conn   = True
                        )   
        
        self.encSep     = sepVae._get_encoder()
        

        self.stripInput   =   InputLayer(input_shape=(None,128,5))
        stripVae = TimeVAE( seq_len             = config.seqLength,
                            feat_dim            = config.stripInput, 
                            latent_dim          = config.latentDim, 
                            hidden_layer_sizes  = config.hiddenSize, # [50,100,200]      #[80, 200, 250] 
                            reconstruction_wt   = 3.0,
                            use_residual_conn   = True
                        )   
        self.encStrip   = stripVae._get_encoder()

        if pretrainedEncoder:
            self.loadEncWeights()

        if freezeEncoder:
            self.freezeEncoder()

        # ---------------------------------------
        # Multihead Attention and Dense layers
        # ---------------------------------------
        self.attention  = MultiHeadAttention(num_heads=config.num_heads, key_dim=config.latentDim)
        
        if config.useComponentAsFeature:
            self.fc1 = Dense(3, activation='relu', input_shape=(config.numComponent,))
            self.fc2 = Dense(1, activation='relu')

        else:
            self.avg_pool = AveragePooling1D(pool_size=config.numComponent)
            self.fc1 = Dense(2, activation='relu', input_shape=(config.latentDim,))
            self.fc2 = Dense(1, activation='relu')
            

    def call(self,input,returnLatent = False):
      
        # print("=======================================")
        reacInput       = self.reacInput(input[0])
        sepInput        = self.sepInput(input[1])
        stripInput      = self.stripInput(input[2])
        
        _, _, zReac     = self.encReac(reacInput)
        _, _, zSep      = self.encSep(sepInput)
        _, _, zStrip    = self.encStrip(stripInput)
        
        
        latentVector    = tf.stack([zReac, zSep, zStrip], axis=1)
        
        
        attentionOut    = self.attention(   query   = latentVector, 
                                            key     = latentVector,
                                            value   = latentVector)
                                         
                                        

        # -----------------------
        # Use the seq as features,
        # move the seq to the end
        if self.config.useComponentAsFeature:
            permutedTensor = tf.transpose(attentionOut, perm=[0, 2, 1])
            pooled_output  = permutedTensor
            
        # -----------------------
        # Use component as features,
        # move feature dim to the end
        else:
            pooled_output   = self.avg_pool(attentionOut)
            

        rulLayer1   = self.fc1(pooled_output)
        rul         = self.fc2(rulLayer1)

        if returnLatent:
            return rul, zReac, zSep, zStrip
        else:
            return rul
        
    def loadEncWeights(self,):
        model_dir       = "checkpointNew/bestCheckpoint"
        encReacWts      = joblib.load(os.path.join(model_dir, f'{"Reactor"}encoder_wts.h5'))
        encSepWts       = joblib.load(os.path.join(model_dir, f'{"Separator"}encoder_wts.h5'))
        encStripWts     = joblib.load(os.path.join(model_dir, f'{"Stripper"}encoder_wts.h5'))

        self.encReac.set_weights(encReacWts)
        self.encSep.set_weights(encSepWts)
        self.encStrip.set_weights(encStripWts)

    def freezeEncoder(self,):
        for layer in self.encReac.layers:
            layer.trainable = False
        
        for layer in self.encSep.layers:
            layer.trainable = False

        for layer in self.encStrip.layers:
            layer.trainable = False


