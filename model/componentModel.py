# ====================================
#  IMPORT LIBRARIES
# ====================================
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


# -----------------------
# Filter the level of 
# debuging
# -----------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 
tf.random.set_seed(1)
 
# ====================================
#  The custom model checkpoint for 
#   subclass method
# ====================================
class SubclassModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, customSaveFunction, config= None, *args, **kwargs):
        super(SubclassModelCheckpoint, self).__init__(*args, **kwargs)

        self.config             = config
        self.customSaveFunction = customSaveFunction

    # --------------------------------
    # Override the default saving function
    # --------------------------------
    def _save_model(self, epoch, logs, *args, **kwargs):
        logs    = logs or {}
        current = logs.get(self.monitor)                                       # Get the monitored metric value (e.g., val_loss)

        if current is None:
            print(f"Warning: Metric {self.monitor} is not available.")
            return
        
        if self.save_best_only: 
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    print(f" \n Epoch {epoch + 1}: {self.monitor} improved from {self.best} to {current}. Saving model...")
                
                self.best   = current
                filepath    = self.filepath.format(epoch=epoch + 1, **logs)         # Format the file path and get the directory
                model_dir   = os.path.dirname(filepath)                             # Extract the directory
                self.customSaveFunction(model_dir, config=self.config)              # Call the custom save function
        else:
            filepath    = self.filepath.format(epoch=epoch + 1, **logs)
            model_dir   = os.path.dirname(filepath)                             # Extract the directory
            self.customSaveFunction(model_dir, config=self.config)              # Call the custom save function
            

# ====================================
# Attention convolution
# ====================================
class AttentionMaskedConv1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=2,  activation='relu', padding = 'same', **kwargs):
        super(AttentionMaskedConv1D, self).__init__(**kwargs)
        self.conv = Conv1D(filters, kernel_size, strides=strides, padding=padding, activation=activation)
        self.attention_weights = Dense(filters, activation='sigmoid')
        
    def call(self, inputs):
        conv_output = self.conv(inputs)
        attn_weights = self.attention_weights(conv_output)
        attended_output = conv_output * attn_weights
        return attended_output
    
# ====================================
# VAE Time (VariationalAutoencoderConv)
# ====================================
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
    def getAttentionEncoder(self, kernelSize = 3, stride = 2):
        encoder_inputs  = Input(shape   = (self.seq_len, self.feat_dim), 
                                name    = 'encoder_input')
        x               = encoder_inputs
        # --------------------
        # Repeat to create Conv1D
        # --------------------
        for i, num_filters in enumerate(self.hidden_layer_sizes):
            x = AttentionMaskedConv1D(  filters     = num_filters, 
                                        kernel_size = kernelSize, 
                                        strides     = stride, 
                                        activation  = 'relu', 
                                        padding     = 'same',
                                        name        = f'enc_conv_{i}')(x)
        sysHILayer = AttentionMaskedConv1D(     filters     = 5, 
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
        return encoder
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
        x = Conv1DTranspose(        filters     = self.feat_dim,
                                    kernel_size = kernelSize,
                                    strides     = stride,
                                    padding     = "same",
                                    activation  = "relu",
                                    name        = f"dec_deconv__{i+1}",
                                )(x)

        x                       = Flatten(name="dec_flatten")(x)
        x                       = Dense(self.seq_len * self.feat_dim, name="decoder_dense_final")(x)
        self.decoder_outputs    = Reshape(target_shape=(self.seq_len, self.feat_dim))(x)
        decoder                 = Model(decoder_inputs, self.decoder_outputs, name="decoder")
        return decoder
    
    # =================================
    # This one is decoder model. Using it
    # when the input is the power of 2
    # ---------------------------------
    def decoderNoDense(self, kernelSize = 3, stride = 2):
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

        # x                       = Flatten(name="dec_flatten")(x)
        # x                       = Dense(self.seq_len * self.feat_dim, name="decoder_dense_final")(x)
        # self.decoder_outputs    = Reshape(target_shape=(self.seq_len, self.feat_dim))(x)
        decoder                 = Model(decoder_inputs, self.decoder_outputs, name="decoder")
        return decoder
    
    
# ====================================
#  Hi system block and RUL
# ====================================
class HiSystemAndRul(tf.keras.Model):
    def __init__(self):
        super(HiSystemAndRul, self).__init__()
        # ------------------------------
        # Input from latent space
        # ------------------------------
        inputRUL1 = Input(shape=(8, 5))
        # inputRUL2 = Input(shape=(8, 5))
        # inputRUL3 = Input(shape=(8, 5))

        # inputHISys1 = Input(shape=(1,))
        # inputHISys2 = Input(shape=(1,))
        # inputHISys3 = Input(shape=(1,))


        # concatenatedHiInput = Concatenate(axis=-1)([inputHISys1, inputHISys2, inputHISys3])
        # hiSystem            = tf.keras.layers.Dense(1)(concatenatedHiInput)
        
        # print(hiSystem.shape)
        
        # ------------------------------
        # RUL
        # ------------------------------
        # concatenatedRULInput = Concatenate(axis=-1)([inputRUL1, inputRUL2, inputRUL3])
        rulLayer1   = GRU(32, return_sequences=True)(inputRUL1)
        rulLayer2   = GRU(16, return_sequences=True)(rulLayer1)
        rulLayer3   = GRU(4, return_sequences=False)(rulLayer2)
        # rulLayer4   = Concatenate(axis=-1)(rulLayer3)
        rul         = tf.keras.layers.Dense(1, activation='linear')(rulLayer3)

        # print(rulLayer4.shape)
        # print(rulLayer3.shape)
        # print(hiSystem.get_shape())
        # # # ------------------------------
        # # # rul
        
        # Create the model
        self.model = Model(inputs=inputRUL1, outputs=rul)
        


    def call(self, inputs):
        # print("----------------------------------")
        # print(inputs.shape)
        return self.model(inputs)
    
# ====================================
#  The sensor selection block
# ====================================
class SensorSelection(tf.keras.Model):
    def __init__(self, inFeature = 45, unchangeFeatures = 0, baseline=0.01):
        '''
        The input size should be (batchSize, timeStep, featureDim)        
        '''
        super(SensorSelection, self).__init__()
        self.baseline    = baseline
        self.mask_logits = self.add_weight(shape        = (inFeature,),
                                           initializer  = 'zeros',
                                           trainable    = True,
                                           name         = 'mask_logits')

    def call(self,rawData):
                          
        probs           = tf.sigmoid(self.mask_logits)
        random_uniform  = tf.random.uniform(tf.shape(probs))
        binary_mask     = tf.cast(tf.less(random_uniform, probs), tf.float32)
        mask            = binary_mask + probs - tf.stop_gradient(probs)
        mask            = tf.expand_dims(mask, axis=0)

        # maskedData      = rawData * mask
        maskedData      = rawData * probs

        return maskedData, probs, binary_mask

    # def call(self, rawData):
    #     # Calculate probabilities
    #     probs = tf.sigmoid(self.mask_logits)
    #     random_uniform = tf.random.uniform(tf.shape(probs))
    #     binary_mask = tf.cast(tf.less(random_uniform, probs), tf.float32)
        
    #     # Force first `unchangeFeatures` to be selected
    #     fixed_mask = tf.concat([
    #         tf.ones(self.unchangeFeatures),  # Force these features to be selected
    #         tf.zeros(tf.shape(probs)[0] - self.unchangeFeatures)  # Others are trainable
    #     ], axis=0)
    #     fixed_mask = tf.cast(fixed_mask, tf.float32)
        
    #     # Combine fixed mask with trainable mask
    #     binary_mask = tf.maximum(binary_mask, fixed_mask)  # Ensure fixed features are always selected
        
    #     # Straight-through estimator
    #     mask = binary_mask + probs - tf.stop_gradient(probs)
    #     mask = tf.expand_dims(mask, axis=0)
        
    #     # Apply mask
    #     maskedData = rawData * mask
        
    #     return maskedData, probs, binary_mask
# ==================================================
#                   SYSTEM MODEL
# ==================================================
class mainModel(tf.keras.Model):
    def __init__(self,config = None):
        super(mainModel, self).__init__()
        # ----------------------------
        # For model
        # ----------------------------
        self.config             = config
        
        vae                     = VAEConvInterpretable( seq_len             = config["seqLength"],  
                                                        feat_dim            = config["in" + config["component"]], 
                                                        latent_dim          = config["latentDim"], 
                                                        hidden_layer_sizes  = config["hiddenVAESize"], # [50,100,200]      #[80, 200, 250] 
                                                        config              = config
                                                        )
        
        self.mask           = SensorSelection(inFeature= config["in" + config["component"]])

        # self.encoder        = vae.getAttentionEncoder(kernelSize = 3, stride = 2)
        self.encoder        = vae.get_encoder(kernelSize = 3, stride = 2)
        self.decoder        = vae.decoderNoDense(kernelSize = 3, stride = 2)

        # self.encoder     = vae.get_encoder(kernelSize = 85, stride = 2)
        # self.decoder     = vae.getUnsymetricDecoder(kernelSize = 85, stride = 1)
        self.hiSystemAndRul  = HiSystemAndRul()

        # ----------------------------
        # For loss tracker
        # ----------------------------
        self.totalLossTracker               = Mean(name="totalLoss")
        self.cr1ConstructionTracker         = Mean(name="cstrLoss" )
        self.cr1ZLossTracker                = Mean(name = "zLoss")
        self.maskLossTracker                = Mean(name= "maskLoss")
        self.rulLossTracker                 = Mean(name= "rulLoss")

        self.bestLoss                      = tf.Variable(float('inf'), trainable=False)


    # ==========================================
    # Call function
    # ------------------------------------------
    def call(self,input):

        # ------------------------------------
        # Encoder and Decoder for 3 components
        # ------------------------------------
        # print("Encoder ")
        # print(input.shape)
        binaryMask = None
        probs      = None
        if self.config["useMask"]:
            encIn, probs, binaryMask    = self.mask(input)
            zCr1,  sysHILayerCr1        = self.encoder(encIn)
            rul                         = self.hiSystemAndRul(sysHILayerCr1)
        else:
            probs                       = 1
            encIn                       = input
            zCr1,  sysHILayerCr1        = self.encoder(input)
            rul                         = self.hiSystemAndRul(sysHILayerCr1)
    
        return zCr1, binaryMask,probs, rul

    # ==========================================
    # Get custom layer
    # ------------------------------------------
    def getHIandRUL(self,input):
        # ------------------------------------
        # Encoder and Decoder for 3 components
        # ------------------------------------
        # print("Encoder ")
        # print(input.shape)
        binaryMask = None
        probs = None
        if self.config["useMask"]:
            encIn, probs, binaryMask    = self.mask(input)
            zCr1,  sysHILayerCr1        = self.encoder(encIn)
            rul                         = self.hiSystemAndRul(sysHILayerCr1)
        else:
            encIn                       = input
            zCr1,  sysHILayerCr1        = self.encoder(input)
            rul                         = self.hiSystemAndRul(sysHILayerCr1)
    
        return zCr1, binaryMask,probs, rul
    # ==========================================
    # Training (replace the default)
    # ------------------------------------------
    def train_step(self, input):
        with tf.GradientTape() as tape:

            # ====================================
            #           MODEL FLOW
            # ------------------------------------
            seqEff, seqCycle, env           = input[0]
            rulY                            = input[1]

            # ------------------------------------
            # Encoder and Decoder for 3 components
            # ------------------------------------
            if self.config["useMask"]:
                encIn, probsMask, binaryMask= self.mask(seqEff)
                zCr1,  sysHILayerCr1        = self.encoder(encIn)
                reconsCr1                   = self.decoder(zCr1)

                decOut,_,_                  = self.mask(reconsCr1)
            else:
                encIn                       = seqEff
                zCr1,  sysHILayerCr1        = self.encoder(seqEff)
                decOut                      = self.decoder(zCr1)

            # reconsCr1                   = self.decoder([zCr1,seqMani])
            # ------------------------------------
            # Create RUL and HI System
            # ------------------------------------
            
            # ====================================
            #               LOSS
            # ------------------------------------
            # ------------------------------------
            # Loss for reconstruction
            # ------------------------------------
            reconLossCr1    = tf.keras.losses.mae(encIn,decOut)
            reconLossCr1    = tf.reduce_mean(reconLossCr1)

            totalLoss       = reconLossCr1
            # ------------------------------------
            # Loss for RUL
            # ------------------------------------
            if self.config["useRUL"]:
                predictedRul    = self.hiSystemAndRul(sysHILayerCr1)
                rulLoss         = tf.keras.losses.mae(predictedRul,rulY)
                rulLoss         = tf.reduce_mean(rulLoss)
                totalLoss       += rulLoss
            else:
                rulLoss = 0
            
            if self.config["useMask"] :
                maskLoss                    = 10* tf.reduce_sum(probsMask * (1 - probsMask))
                enforceUnchangedFeatureLoss = 10 * tf.reduce_sum((1 - probsMask[:self.config["unchangeFeatures"]]) ** 2) if self.config["unchangeFeatures"] != 0 else 0

                totalLoss += (maskLoss + enforceUnchangedFeatureLoss)
            else:
                maskLoss = 0

            # ------------------------------------
            # Add the constrain to the latent space
            # ------------------------------------
            if self.config["latentConstrains"] :
                # zReacLoss     = self._getConstrain(inputData = [zCr1,    seqCycle], type = "trendability", alpha=10)
                zReacLoss,_   = self._getConstrain(inputData = [zCr1,    seqCycle, env], type = "exportCurveWithEnv", alpha=10)
                totalLoss    += zReacLoss
            else:
                zReacLoss = 0
            
        # --------------------------------
        # Update gradient
        # --------------------------------
        grads = tape.gradient(totalLoss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    
        # --------------------------------
        # Update losses
        # --------------------------------
        self.totalLossTracker.update_state(totalLoss)

        # Construction loss 
        self.cr1ConstructionTracker.update_state(reconLossCr1)

        # RUL loss
        if self.config["useRUL"]:
            self.rulLossTracker.update_state(rulLoss)

        # Latent loss
        if self.config["latentConstrains"] :
            self.cr1ZLossTracker.update_state(zReacLoss)

        # Mask loss
        if self.config["useMask"] :
            self.maskLossTracker.update_state(maskLoss)
       

        return {
            "totalLoss"             : self.totalLossTracker.result(),
            "rulLoss"               : self.rulLossTracker.result(),        
            "cstrLoss"              : self.cr1ConstructionTracker.result(),
            "zLoss"                 : self.cr1ZLossTracker.result(),
            "maskLoss"              : self.maskLossTracker.result()
        }

    # -----------------------------
    # Testing (replace the default)
    # -----------------------------
    def test_step(self, input): 

        # ====================================
        #           MODEL FLOW
        # ------------------------------------
        seqEff,seqCycle             = input[0]
        rulY                        = input[1]
        # ------------------------------------
        # Encoder and Decoder for 3 components
        # ------------------------------------
        if self.config["useMask"]:
            encIn, probsMask, binaryMask= self.mask(seqEff)

            zCr1,  sysHILayerCr1        = self.encoder(encIn)
            reconsCr1                   = self.decoder(zCr1)

            decOut,_,_                  = self.mask(reconsCr1)
        else:
            encIn                       = seqEff
            zCr1,  sysHILayerCr1        = self.encoder(seqEff)
            decOut                      = self.decoder(zCr1)

        # reconsCr1                   = self.decoder([zCr1,seqMani])
        # ------------------------------------
        # Create RUL and HI System
        # ------------------------------------
        # tf.print("Sum of binary mask:", tf.reduce_sum(binaryMask))
        # tf.print("Mask values:", binaryMask)
        # print("HI and system rul")
        predictedRul = self.hiSystemAndRul(sysHILayerCr1)

        # print("Loss funcion")
        # ====================================
        #               LOSS
        # ------------------------------------
        # L1 regularization to promote sparsity
        # l1_reg = tf.reduce_sum(probs)
        # Binarization regularization
        maskLoss = 10* tf.reduce_sum(probsMask * (1 - probsMask))
        # maskLoss = l1_reg + binary_reg
        # ------------------------------------
        # Loss for reconstruction
        # ------------------------------------
        # reconLossCr1   = self._get_reconstruction_loss(seqEff, reconsCr1)
        # print(encIn.shape, decOut.shape)
        reconLossCr1    = tf.keras.losses.mae(encIn,decOut)
        reconLossCr1    = tf.reduce_mean(reconLossCr1)
        # ------------------------------------
        # Loss for RUL
        # ------------------------------------
        rulLoss = tf.keras.losses.mae(predictedRul,rulY)
        rulLoss = tf.reduce_mean(rulLoss)

        totalLoss = reconLossCr1   +  rulLoss + maskLoss
                    
        # ------------------------------------
        # Add the constrain to the latent space
        # ------------------------------------
        if self.config["latentConstrains"] :
            zReacLoss     = self._getConstrain(inputData = [zCr1,    seqCycle], type = "expertCurve", alpha=10)
            # zReacLoss     = self._getConstrain(inputData = [zCr1,    seqCycle], type = "reliability", alpha=1)
            totalLoss    += zReacLoss
        else:
            zReacLoss = 0
            
    
        # --------------------------------
        # Update losses
        # --------------------------------
        self.totalLossTracker.update_state(totalLoss)
        self.rulLossTracker.update_state(rulLoss)

        # Construction loss 
        self.cr1ConstructionTracker.update_state(reconLossCr1)
        
        # Latent loss
        if self.config["latentConstrains"] :
            self.cr1ZLossTracker.update_state(zReacLoss)
        self.maskLossTracker.update_state(maskLoss)
       

        return {
            "totalLoss"             : self.totalLossTracker.result(),
            "rulLoss"               : self.rulLossTracker.result(),        
            "cstrLoss"            : self.cr1ConstructionTracker.result(),
            "zLoss"               : self.cr1ZLossTracker.result(),
            "maskLoss"              : self.maskLossTracker.result()
        }
    
    # --------------------------------
    # The reconstruction loss
    # --------------------------------
    def _get_reconstruction_loss(self, X, X_recons): 
        def get_reconst_loss_by_axis(X, X_c, axis): 
            x_r     = tf.reduce_mean(X, axis = axis)
            x_c_r   = tf.reduce_mean(X_recons, axis = axis)
            err     = tf.math.squared_difference(x_r, x_c_r)
            loss    = tf.reduce_sum(err)
            return loss

        # overall    
        err             = tf.math.squared_difference(X, X_recons)
        reconst_loss    = tf.reduce_sum(err)
      
        reconst_loss += get_reconst_loss_by_axis(X, X_recons, axis=[2])     # by time axis        
        # reconst_loss += get_reconst_loss_by_axis(X, X_recons, axis=[1])    # by feature axis
        return reconst_loss
    
        # --------------------------------
        # Latent space constrains
        # --------------------------------
    def _getConstrain(self,inputData, type="expertCurve",alpha = 1, initCurve = 0, kRate = 0.064, tMovement = 20):
        if type == "trendability":
            x, t = inputData
            x_mu = tf.reduce_mean(x)        # Calculate means
            t_mu = tf.reduce_mean(t)        # Calculate means
            
            # -------------------------
            # Calculate covariance and 
            # standard deviation
            # -------------------------
            cov = tf.reduce_sum((x - x_mu) * (t - t_mu))
            

            # Calculate standard deviations
            std = tf.math.sqrt(tf.reduce_sum(tf.square(x - x_mu))) * tf.math.sqrt(tf.reduce_sum(tf.square(t - t_mu)))
            # -------------------------
            # Check if std is finite and 
            # greater than zero
            # -------------------------
            loss = tf.cond(tf.math.is_finite(std) & (std > 0), lambda: cov / (std), lambda: self.bestLoss)  # Use last loss if std is not finite
            self.add_update(self.bestLoss.assign(loss))         # Update lastLoss if std is valid 
            # loss = tf.abs(1/loss)                               # Making positive values and increase



            # std_x = tf.math.sqrt(tf.reduce_sum(tf.square(x - x_mu)))
            # std_t = tf.math.sqrt(tf.reduce_sum(tf.square(t - t_mu)))
            # loss    = cov / (std_x * std_t)
            # Loss: 1 - abs(correlation) to ensure the loss is bounded between 0 and 1
            # loss = 1 - tf.abs(correlation)


        elif type == "reliability":
            # A       = 0.012
            # B       = 4.95
            # C       = 0.89
            # BETA    = 6.7
            # P       = 0.5

            A       = 3.24689126e-04
            B       = 6.75139693e-01
            C       = 1.02044605e+00
            BETA    = 3.5
            P       = 0.5

            # x, t        = inputData
            # exp_part    = tf.pow(-tf.math.log(1 - P), -1 / BETA)
            # HI          = C - tf.pow((t * exp_part) * A, B)
            # # Use tf.where to ensure no negative HI values
            # HI          = tf.where(HI < 0, tf.zeros_like(HI), HI)
            # loss        = tf.reduce_mean(tf.abs(x - HI))    

            z, t    = inputData
            HI      = C - ((t * (-np.log(1 - P))**(-1/BETA))*A)**B
            mask    = tf.math.less(HI,0)
            HI      = tf.where(mask,0.0,HI)

            # print(HI.shape, z.shape)
            
            loss    = tf.reduce_mean(tf.abs(z-HI))

        elif type == "expertCurve":
            x, t    = inputData
            y       = tf.exp(kRate*(t+ tMovement))
            loss    = tf.reduce_mean(tf.square(y - x))

        elif type == "normalExp":
            x, t    = inputData
            k       = 0.04986
            y       = tf.exp(-k*x)
            loss    = tf.reduce_mean(tf.square(y - x))

        elif type == "exportCurveWithEnv":
            z, t, env = inputData  # Unpacking input data

            # Ensure t is float32
            t = tf.cast(t, tf.float32)

            # Define HI functions for each environment
            hiEnv1 = (tf.cast(t > 30, tf.float32) * (64.8295 + 0.0248 * tf.exp(0.064 * (t + 20)))) + \
                    (tf.cast(t <= 30, tf.float32) * tf.ones_like(t) * 65.0)

            hiEnv3 = (tf.cast(t > 30, tf.float32) * (64.8295 - 0.0248 * tf.exp(0.064 * (t + 20)))) + \
                    (tf.cast(t <= 30, tf.float32) * tf.ones_like(t) * 65.0)

            # Create masks for each environment
            mask1 = tf.cast(tf.equal(env, 1), tf.float32)
            mask3 = tf.cast(tf.equal(env, 3), tf.float32)

            # Compute the target HI based on environment
            hiTarget = (mask1 * hiEnv1) + (mask3 * hiEnv3)

            # Compute L2 loss
            loss = tf.reduce_mean(tf.square(z - hiTarget))
        return alpha*loss, hiTarget


    # --------------------------------
    # Save weight of each block
    # --------------------------------
    def save_weights(self, modelDir, type = "subclass"):
        encoderReacWts      = self.encoder.get_weights()
        decoderReacWts      = self.decoder.get_weights()
        maskWts             = self.mask.get_weights()
        hiAndRulWts         = self.hiSystemAndRul.get_weights()
        
        joblib.dump(encoderReacWts, os.path.join(modelDir, "encoderWts.h5"))
        joblib.dump(decoderReacWts, os.path.join(modelDir, "decoderWts.h5"))
        joblib.dump(maskWts, os.path.join(modelDir, "maskWts.h5"))
        joblib.dump(hiAndRulWts, os.path.join(modelDir, "hiAndRulWts.h5"))



    # --------------------------------
    # Load weight for each block
    # of the model
    # --------------------------------
    def load_weights(self, model_dir, type = "subclass"):
        aiLogger.info(f"Loading the model at the direction: {model_dir}")
        encoderReacWts      = joblib.load(os.path.join(model_dir, "encoderWts.h5"))
        decoderReacWts      = joblib.load(os.path.join(model_dir, "decoderWts.h5"))
        maskWts             = joblib.load(os.path.join(model_dir, "maskWts.h5"))
        hiAndRulWts         = joblib.load(os.path.join(model_dir, "hiAndRulWts.h5"))
      
        self.encoder.set_weights(encoderReacWts)
        self.decoder.set_weights(decoderReacWts)
        self.mask.set_weights(maskWts)
        self.hiSystemAndRul.set_weights(hiAndRulWts)
        aiLogger.info(f"Loaded the model at the direction:{model_dir} successfully!!!")

    # --------------------------------
    # This function is used for subclassing
    # saving, for function api or sequential 
    # method, change to use tensorflow saving
    # --------------------------------
    def save(self, model_dir, config = None):
        aiLogger.info(f"Saving the model at the direction:{model_dir}")
        os.makedirs(model_dir, exist_ok=True)
        self.save_weights(model_dir)

        # print(config)
        # ---------------------
        # Save parameter if possible
        # if config is not None:
        #     parameters = {  "seqLength"             : config["seqLength"],
        #                     "featureDim"            : config["inFeatureDim"],
        #                     "latentDim"             : config["latentDim"],
        #                     "hiddenVAESize"         : config["hiddenVAESize"],
        #                     }
        #     parameterFile = os.path.join(model_dir, "parameters.pkl")
        #     joblib.dump(parameters, parameterFile)
        aiLogger.info(f"Saved the model at the direction:{model_dir} successfully!!!")



# if __name__ == '__main__':
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
#     warnings.filterwarnings('ignore') 

#     np.random.seed(1)
#     random.seed(1)
#     tf.random.set_seed(1)

#     # ====================================
#     # Config
#     # ------------------------------------
#     with open('utility/modelConfig.json', 'r') as configFile:
#         config = json.load(configFile)
#     myConfig   = config["system"]

#     model       = SystemModel(myConfig)       
#     model.compile( optimizer   =   Adam())

