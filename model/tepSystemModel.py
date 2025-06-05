# ====================================
#  IMPORT LIBRARIES
# ====================================
from keras.layers                   import Conv1D,  Flatten, Dense, Conv1DTranspose, Reshape, Input, Layer, GRU, LSTM, Concatenate, Embedding, MultiHeadAttention, Lambda, Multiply, Minimum, Maximum
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
        # print(encoder.summary())
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


        # x                       = Flatten(name="dec_flatten")(x)
        # x                       = Dense(self.seq_len * self.feat_dim, name="decoder_dense_final")(x)
        # self.decoder_outputs    = Reshape(target_shape=(self.seq_len, self.feat_dim))(x)
        decoder                 = Model(decoder_inputs, self.decoder_outputs, name="decoder")
        # print(decoder.summary())
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
    def __init__(self, config):
        super(HiSystemAndRul, self).__init__()
        # ------------------------------
        # Input from latent space
        # ------------------------------
        inputSensor = Input(shape=(128,45),    name = "SensorInputForHISelection")
        inputRUL1   = Input(shape=(8, 5),     name = "lastLayerAEComponent1")
        inputRUL2   = Input(shape=(8, 5),     name = "lastLayerAEComponent2")
        inputRUL3   = Input(shape=(8, 5),     name = "lastLayerAEComponent3")

        inputHI1    = Input(shape=(1,),     name = "compHI1")
        inputHI2    = Input(shape=(1,),     name = "compHI2")
        inputHI3    = Input(shape=(1,),     name = "compHI3")

        if config["useInputAsHISelection"]:
            hiSelectionInput = inputSensor
        else:
            flatInRUL1          = Flatten()(inputRUL1)  # Shape: (32, 40)
            flatInRUL2          = Flatten()(inputRUL2)  # Shape: (32, 40)
            flatInRUL3          = Flatten()(inputRUL3)  # Shape: (32, 40)
            hiSelectionInput    = Concatenate(axis=-1)([flatInRUL1, flatInRUL2, flatInRUL3])
        
        aiLogger.debug(f"The shape of hiSelectionInput {hiSelectionInput.shape}")
        # ------------------------------
        # Create the HI selection matrix
        # ------------------------------
        x                   = Dense(24, activation="relu")(hiSelectionInput)
        x                   = Dense(16, activation="relu")(x)
        selection_logits    = Dense(3, activation=None, name="Selection_Logits")(x)
        hiMatrixOutput      = Lambda(lambda x: tf.nn.softmax(x / 1), name="Selection_Output")(selection_logits)  # Softmax with τ=0.1 for sharper selection
        aiLogger.debug(f"The shape of multiplied matrix {hiMatrixOutput.shape}")

        # ------------------------------
        # Create min, max and linear HI
        # ------------------------------
        concatenatedHiInput = Concatenate(axis=-1)([inputHI1, inputHI2, inputHI3])
        minSysHi            = Minimum()([inputHI1, inputHI2, inputHI3])
        maxSysHi            = Maximum()([inputHI1, inputHI2, inputHI3]) 
        linearSysHi         = Dense(1)(concatenatedHiInput)

        aiLogger.debug(f"The shape of min, max and linear HI {minSysHi.shape, maxSysHi.shape, linearSysHi.shape}")
        
        sysConcatenatedHi   = tf.keras.layers.Concatenate(axis=-1, name="sysConcatenatedHi")([minSysHi, maxSysHi, linearSysHi])
        aiLogger.debug(f"The shape after concatenation {sysConcatenatedHi.shape}")
        selected_HI         = Multiply(name="Selected_HI")([hiMatrixOutput, sysConcatenatedHi])

        aiLogger.debug(f"The shape after multiplying with mask {selected_HI.shape}")
        sysHi               = Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=True), name="Final_HI")(selected_HI)            # Final HI (ensuring only one HI contributes to output)
        # sysHi               = linearSysHi

        aiLogger.debug(f"The shape of HI after reducing dimension {sysHi.shape}")
        # ------------------------------
        # RUL
        # ------------------------------
        concatenatedRULInput = Concatenate(axis=-1)([inputRUL1, inputRUL2, inputRUL3])
        # concatenatedRULInput = Concatenate(axis=-1)([inputRUL1, inputRUL2])
        rulLayer1   = LSTM(32, return_sequences=True)(concatenatedRULInput)
        rulLayer2   = LSTM(16, return_sequences=True)(rulLayer1)
        rulLayer3   = LSTM(3, return_sequences=False)(rulLayer2)
        rulLayer4   = Concatenate(axis=-1)([rulLayer3, sysHi])
        rul         = tf.keras.layers.Dense(1, activation='linear')(rulLayer4)
        
        # Create the model
        self.model = Model(inputs=[inputSensor, inputRUL1, inputRUL2, inputRUL3, inputHI1,inputHI2,inputHI3], outputs=[hiMatrixOutput, sysHi,rul])


    def call(self, inputs):
        # print("----------------------------------")
        # print(inputs.shape)
        return self.model(inputs)
    
# ====================================
#  The sensor selection block
# ====================================
class SensorSelection(tf.keras.Model):
    def __init__(self, inFeature = 45, numEnvs = 3, comp = "Reac"):
        '''
        The input size should be (batchSize, timeStep, featureDim)        
        '''
        super(SensorSelection, self).__init__()
        # self.mask_logits = self.add_weight(shape        = (inFeature,),
        #                                    initializer  = 'zeros',
        #                                    trainable    = True,
        #                                    name         = 'mask_logits')
        # self.numComponents  = numComponents

        # Mode 2 matrices
        self.physicMask = tf.Variable(tf.zeros((3, 45), dtype=tf.float32))
        if comp == "Reac":
            reac_2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
            reac_3 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0])
            reac_4 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1])
            self.physicMask = tf.constant(np.array([reac_2, reac_3, reac_4]), dtype=tf.float32)
        elif comp == "Sep":
            sep_2 = np.array([1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0])
            sep_3 = np.array([1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0])
            sep_4 = np.array([1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0])
            self.physicMask = tf.constant(np.array([sep_2, sep_3, sep_4]), dtype=tf.float32)

        else:
            strip_2 = np.array([1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0])
            strip_3 = np.array([1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0])
            strip_4 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
            self.physicMask = tf.constant(np.array([strip_2, strip_3, strip_4]), dtype=tf.float32)

        # Mode 3 matrices

        # Mode 4 matrices
        self.numModes       = numEnvs         # 3 modes in total
        self.mask_logits    = self.add_weight(  shape       =   (self.numModes, inFeature),  # Each mode has different sensor probabilities
                                                initializer =   "random_normal",
                                                trainable   =   True,
                                                name        =   "sensorMask"
                                            )
        # print(self.mask_logits.shape)

    def call(self,rawData,env):
        # ------------------------
        # Select the environment
        # ------------------------
        # print("-------------------------------", env.shape)
        env                 = tf.cast(env-2, tf.int32)                                # Ensure integer type for indexing, and env from 1 to three, so we need index from 0 to 2
        env_mask_logits     = tf.gather(self.mask_logits, env, axis=0)              # (batch_size, num_sensors) each sequence (after split) will have a different env, so env should be a batch (include differnt env to access to the mask)                    
        # withPhysicMask      = tf.gather(self.physicMask, env, axis=0)
        
        probs               = tf.sigmoid(env_mask_logits)
        random_uniform      = tf.random.uniform(tf.shape(probs))
        binary_mask         = tf.cast(tf.less(random_uniform, probs), tf.float32)
        # print("==================================",env_mask_logits.shape)
        # mask                = binary_mask + probs - tf.stop_gradient(probs)         # Apply the mask using the reparameterization trick
        # mask                = tf.expand_dims(mask, axis=0)
        # maskedData      = rawData * mask
        print("-------------------------------")
        print(self.mask_logits.shape)
        print(env_mask_logits.shape)
        print(rawData.shape)

        maskedData          = rawData * probs
        print(maskedData.shape)
        print("-------------------------------")
        return maskedData, probs, binary_mask
    
    def getMask(self,):
        probs               = tf.sigmoid(self.mask_logits)   
        # random_uniform      = tf.random.uniform(tf.shape(probs))
        binary_mask         = tf.cast(tf.less(0.5, probs), tf.float32) 
        return binary_mask, probs


# ==================================================
#                   SYSTEM MODEL
# ==================================================
class SystemModel(tf.keras.Model):
    def __init__(self,config = None):
        super(SystemModel, self).__init__()
        # ----------------------------
        # For model
        # ----------------------------
        self.config             = config
        
        vaeCr1                     = VAEConvInterpretable(  seq_len             = config["seqLength"],  
                                                            feat_dim            = config["in" + config["allComp"][0]], 
                                                            latent_dim          = config["latentDim"], 
                                                            hidden_layer_sizes  = config["hiddenVAESize"], # [50,100,200]      #[80, 200, 250] 
                                                            config              = config
                                                        )
        vaeCr2                     = VAEConvInterpretable(  seq_len             = config["seqLength"],  
                                                            feat_dim            = config["in" + config["allComp"][1]], 
                                                            latent_dim          = config["latentDim"], 
                                                            hidden_layer_sizes  = config["hiddenVAESize"], # [50,100,200]      #[80, 200, 250] 
                                                            config              = config
                                                            )
        vaeCv2                     = VAEConvInterpretable(  seq_len             = config["seqLength"],  
                                                            feat_dim            = config["in" + config["allComp"][2]], 
                                                            latent_dim          = config["latentDim"], 
                                                            hidden_layer_sizes  = config["hiddenVAESize"], # [50,100,200]      #[80, 200, 250] 
                                                            config              = config
                                                            )
        

        self.maskCr1           = SensorSelection(inFeature= config["in" + config["allComp"][0]], comp= "Reac")
        self.maskCr2           = SensorSelection(inFeature= config["in" + config["allComp"][1]], comp= "Sep")
        self.maskCv2           = SensorSelection(inFeature= config["in" + config["allComp"][2]], comp= "Strip")

        # self.encoder        = vae.getAttentionEncoder(kernelSize = 3, stride = 2)
        self.encoderCr1        = vaeCr1.get_encoder(kernelSize = 3, stride = 2)
        self.decoderCr1        = vaeCr1.decoderNoDense(kernelSize = 3, stride = 2)

        self.encoderCr2        = vaeCr2.get_encoder(kernelSize = 3, stride = 2)
        self.decoderCr2        = vaeCr2.decoderNoDense(kernelSize = 3, stride = 2)

        self.encoderCv2        = vaeCv2.get_encoder(kernelSize = 3, stride = 2)
        self.decoderCv2        = vaeCv2.decoderNoDense(kernelSize = 3, stride = 2)

        dataDir                = os.path.expanduser(config["dataDir"])
        modelName1             = "bestModel34Reac"
        modelName2             = "bestModel35Sep"
        modelName3             = "bestModel36Strip"
        if config["frezeComponents"]:
            self.loadWeightAndFreezePretrainedHI(modelDirComp1= dataDir + "checkpoint/" + modelName1,
                                                 modelDirComp2= dataDir + "checkpoint/" + modelName2,
                                                 modelDirComp3= dataDir + "checkpoint/" + modelName3)

        # self.encoder     = vae.get_encoder(kernelSize = 85, stride = 2)
        # self.decoder     = vae.getUnsymetricDecoder(kernelSize = 85, stride = 1)
        self.hiSystemAndRul  = HiSystemAndRul(self.config)

        # ----------------------------
        # For loss tracker
        # ----------------------------
        self.totalLossTracker               = Mean(name="totalLoss")
        self.rulLossTracker                 = Mean(name= "rulLoss")
        self.hiSelectionLossTracker         = Mean(name = "hiSelectionLoss")

        self.bestLoss                      = tf.Variable(float('inf'), trainable=False)


    # ==========================================
    # Call function
    # ------------------------------------------
    def call(self,input):

        seqData,seqCycle, env                   = input
        # ------------------------------------
        # Encoder and Decoder for 3 components
        # ------------------------------------
        # print("Encoder ")
        # print(input.shape)
        binaryMaskCr1 = None
        binaryMaskCr2 = None
        binaryMaskCv2 = None
        probsMaskCr1 = None
        probsMaskCr2 = None
        probsMaskCv2 = None
        if self.config["useMask"]:
            encInCr1, probsMaskCr1, binaryMaskCr1   = self.maskCr1(seqData, env)
            encInCr2, probsMaskCr2, binaryMaskCr2   = self.maskCr2(seqData, env)
            encInCv2, probsMaskCv2, binaryMaskCv2   = self.maskCv2(seqData, env)

            zCr1,  sysHILayerCr1                    = self.encoderCr1(encInCr1)
            zCr2,  sysHILayerCr2                    = self.encoderCr2(encInCr2)
            zCv2,  sysHILayerCv2                    = self.encoderCv2(encInCv2)
        else:
            zCr1,  sysHILayerCr1                    = self.encoderCr1(seqData)
            zCr2,  sysHILayerCr2                    = self.encoderCr2(seqData)
            zCv2,  sysHILayerCv2                    = self.encoderCv2(seqData)
        
        hiMatrixOutput, sysHi, predictedRul = self.hiSystemAndRul([seqData, sysHILayerCr1, sysHILayerCr2, sysHILayerCv2, zCr1, zCr2, zCv2])
        # hiSystem, predictedRul = self.hiSystemAndRul([sysHILayerCr1, sysHILayerCr2, zCr1, zCr2])
     
        return zCr1, zCr2,  zCv2, binaryMaskCr1, binaryMaskCr2, binaryMaskCv2, probsMaskCr1, probsMaskCr2, probsMaskCv2, sysHi, predictedRul

    # ==========================================
    # Get custom layer
    # ------------------------------------------
    def getHIandRUL(self,input):
        seqData,seqCycle, env                   = input
        # ------------------------------------
        # Encoder and Decoder for 3 components
        # ------------------------------------
        # print("Encoder ")
        # print(input.shape)
        binaryMaskCr1 = None
        binaryMaskCr2 = None
        binaryMaskCv2 = None
        probsMaskCr1 = None
        probsMaskCr2 = None
        probsMaskCv2 = None
        if self.config["useMask"]:
            encInCr1, probsMaskCr1, binaryMaskCr1   = self.maskCr1(seqData, env)
            encInCr2, probsMaskCr2, binaryMaskCr2   = self.maskCr2(seqData, env)
            encInCv2, probsMaskCv2, binaryMaskCv2   = self.maskCv2(seqData, env)

            zCr1,  sysHILayerCr1                    = self.encoderCr1(encInCr1)
            zCr2,  sysHILayerCr2                    = self.encoderCr2(encInCr2)
            zCv2,  sysHILayerCv2                    = self.encoderCv2(encInCv2)
        else:
            zCr1,  sysHILayerCr1                    = self.encoderCr1(seqData)
            zCr2,  sysHILayerCr2                    = self.encoderCr2(seqData)
            zCv2,  sysHILayerCv2                    = self.encoderCv2(seqData)
        
        hiMatrixOutput, sysHi, predictedRul = self.hiSystemAndRul([seqData,sysHILayerCr1, sysHILayerCr2, sysHILayerCv2, zCr1, zCr2, zCv2])
        
        # hiMatrixOutput, sysHi, predictedRul = self.hiSystemAndRul([seqData,sysHILayerCr1, sysHILayerCr2, sysHILayerCv2, zCr1, zCr2, zCv2])
        # hiSystem, predictedRul = self.hiSystemAndRul([sysHILayerCr1, sysHILayerCr2, zCr1, zCr2])

     
        return zCr1, zCr2,  zCv2, sysHi, predictedRul, hiMatrixOutput
    
    def getRUL(self,input):
        seqDataCr1, seqDataCr2, seqDataCv2 = input
        # ------------------------------------
        # Encoder and Decoder for 3 components
        # ------------------------------------
        # print("Encoder ")
        # print(input.shape)
        binaryMaskCr1 = None
        binaryMaskCr2 = None
        binaryMaskCv2 = None
        probsMaskCr1 = None
        probsMaskCr2 = None
        probsMaskCv2 = None
        if self.config["useMask"]:
            encInCr1, probsMaskCr1, binaryMaskCr1   = self.maskCr1(seqDataCr1)
            encInCr2, probsMaskCr2, binaryMaskCr2   = self.maskCr2(seqDataCr2)
            encInCv2, probsMaskCv2, binaryMaskCv2   = self.maskCv2(seqDataCv2)

            zCr1,  sysHILayerCr1                    = self.encoderCr1(encInCr1)
            zCr2,  sysHILayerCr2                    = self.encoderCr2(encInCr2)
            zCv2,  sysHILayerCv2                    = self.encoderCv2(encInCv2)
        else:
            zCr1,  sysHILayerCr1                    = self.encoderCr1(seqDataCr1)
            zCr2,  sysHILayerCr2                    = self.encoderCr2(seqDataCr2)
            zCv2,  sysHILayerCv2                    = self.encoderCv2(seqDataCv2)
        
        hiSystem, predictedRul = self.hiSystemAndRul([sysHILayerCr1, sysHILayerCr2, sysHILayerCv2, zCr1, zCr2, zCv2])
        # hiSystem, predictedRul = self.hiSystemAndRul([sysHILayerCr1, sysHILayerCr2, zCr1, zCr2])

     
        return predictedRul

    # ==========================================
    # Training (replace the default)
    # ------------------------------------------
    def train_step(self, input):
        with tf.GradientTape() as tape:

            # ====================================
            #           MODEL FLOW
            # ------------------------------------
            seqData,seqCycle, env                   = input[0]
            rulY                                    = input[1]
            # ------------------------------------
            # Encoder and Decoder for 3 components
            # ------------------------------------
            if self.config["useMask"]:
                encInCr1, probsMaskCr1, binaryMaskCr1   = self.maskCr1(seqData, env)
                encInCr2, probsMaskCr2, binaryMaskCr2   = self.maskCr2(seqData, env)
                encInCv2, probsMaskCv2, binaryMaskCv2   = self.maskCv2(seqData, env)


                zCr1,  sysHILayerCr1                    = self.encoderCr1(encInCr1)
                zCr2,  sysHILayerCr2                    = self.encoderCr2(encInCr2)
                zCv2,  sysHILayerCv2                    = self.encoderCv2(encInCv2)
                
            else:
                encInCr1                                = seqData
                encInCr2                                = seqData
                encInCv2                                = seqData

                zCr1,  sysHILayerCr1                    = self.encoderCr1(encInCr1)
                zCr2,  sysHILayerCr2                    = self.encoderCr2(encInCr2)
                zCv2,  sysHILayerCv2                    = self.encoderCv2(encInCv2)


            hiMatrixOutput, hiSystem, predictedRul = self.hiSystemAndRul([seqData, sysHILayerCr1, sysHILayerCr2, sysHILayerCv2, zCr1, zCr2, zCv2])

            # ====================================
            #               LOSS
            # ------------------------------------
            
            # ------------------------------------
            # Loss for RUL
            # ------------------------------------
            rulLoss     = tf.keras.losses.mae(predictedRul,rulY)
            rulLoss     = tf.reduce_mean(rulLoss)
            totalLoss   = rulLoss

            if self.config["hiSelection"]:
                hiSelectionLoss  = -tf.reduce_sum(hiMatrixOutput * tf.math.log(hiMatrixOutput + 1e-10), axis=-1)    #  Entropy-like loss to select only 1 value of 1
                totalLoss       += hiSelectionLoss
            else:
                hiSelectionLoss = 0


                        
        # --------------------------------
        # Update gradient
        # --------------------------------
        grads = tape.gradient(totalLoss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    
        # --------------------------------
        # Update losses
        # --------------------------------
        self.totalLossTracker.update_state(totalLoss)
        self.rulLossTracker.update_state(rulLoss)
        self.hiSelectionLossTracker.update_state(hiSelectionLoss)



        return {
            "totalLoss"                 : self.totalLossTracker.result(),
            "rulLoss"                   : self.rulLossTracker.result(), 
            "hiSelectionLoss"           : self.hiSelectionLossTracker.result()
        }

    # -----------------------------
    # Testing (replace the default)
    # -----------------------------
    def test_step(self, input): 


        # ====================================
        #           MODEL FLOW
        # ------------------------------------
        seqData,seqCycle, env                   = input[0]
        rulY                                    = input[1]
        # ------------------------------------
        # Encoder and Decoder for 3 components
        # ------------------------------------
        if self.config["useMask"]:
            encInCr1, probsMaskCr1, binaryMaskCr1   = self.maskCr1(seqData, env)
            encInCr2, probsMaskCr2, binaryMaskCr2   = self.maskCr2(seqData, env)
            encInCv2, probsMaskCv2, binaryMaskCv2   = self.maskCv2(seqData, env)


            zCr1,  sysHILayerCr1                    = self.encoderCr1(encInCr1)
            zCr2,  sysHILayerCr2                    = self.encoderCr2(encInCr2)
            zCv2,  sysHILayerCv2                    = self.encoderCv2(encInCv2)
            
        else:
            encInCr1                                = seqData
            encInCr2                                = seqData
            encInCv2                                = seqData

            zCr1,  sysHILayerCr1                    = self.encoderCr1(encInCr1)
            zCr2,  sysHILayerCr2                    = self.encoderCr2(encInCr2)
            zCv2,  sysHILayerCv2                    = self.encoderCv2(encInCv2)


        hiMatrixOutput, hiSystem, predictedRul = self.hiSystemAndRul([seqData, sysHILayerCr1, sysHILayerCr2, sysHILayerCv2, zCr1, zCr2, zCv2])

        # ====================================
        #               LOSS
        # ------------------------------------
        
        # ------------------------------------
        # Loss for RUL
        # ------------------------------------
        rulLoss     = tf.keras.losses.mae(predictedRul,rulY)
        rulLoss     = tf.reduce_mean(rulLoss)
        totalLoss   = rulLoss

        if self.config["hiSelection"]:
            hiSelectionLoss  = -tf.reduce_sum(hiMatrixOutput * tf.math.log(hiMatrixOutput + 1e-10), axis=-1)    #  Entropy-like loss to select only 1 value of 1
            totalLoss       += hiSelectionLoss
        else:
            hiSelectionLoss = 0


                        
    
        # --------------------------------
        # Update losses
        # --------------------------------
        self.totalLossTracker.update_state(totalLoss)
        self.rulLossTracker.update_state(rulLoss)
        self.hiSelectionLossTracker.update_state(hiSelectionLoss)



        return {
            "totalLoss"                 : self.totalLossTracker.result(),
            "rulLoss"                   : self.rulLossTracker.result(), 
            "hiSelectionLoss"           : self.hiSelectionLossTracker.result()
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
        return alpha*loss


    # --------------------------------
    # Save weight of each block
    # --------------------------------
    def save_weights(self, modelDir, type = "subclass"):
        # encoderCr1Wts       = self.encoderCr1.get_weights()
        # decoderCr1Wts       = self.decoderCr1.get_weights()
        # maskCr1Wts          = self.maskCr1.get_weights()

        # encoderCr2Wts       = self.encoderCr2.get_weights()
        # decoderCr2Wts       = self.decoderCr2.get_weights()
        # maskCr2Wts          = self.maskCr2.get_weights()

        # encoderCv2Wts       = self.encoderCv2.get_weights()
        # decoderCv2Wts       = self.decoderCv2.get_weights()
        # maskCv2Wts          = self.maskCv2.get_weights()


        hiAndRulWts         = self.hiSystemAndRul.get_weights()
        
        # joblib.dump(encoderCr1Wts, os.path.join(modelDir, "encoderCr1Wts.h5"))
        # joblib.dump(decoderCr1Wts, os.path.join(modelDir, "decoderCr1Wts.h5"))
        # joblib.dump(maskCr1Wts, os.path.join(modelDir, "maskCr1Wts.h5"))

        # joblib.dump(encoderCr2Wts, os.path.join(modelDir, "encoderCr2Wts.h5"))
        # joblib.dump(decoderCr2Wts, os.path.join(modelDir, "decoderCr2Wts.h5"))
        # joblib.dump(maskCr2Wts, os.path.join(modelDir, "maskCr2Wts.h5"))

        # joblib.dump(encoderCv2Wts, os.path.join(modelDir, "encoderCv2Wts.h5"))
        # joblib.dump(decoderCv2Wts, os.path.join(modelDir, "decoderCv2Wts.h5"))
        # joblib.dump(maskCv2Wts, os.path.join(modelDir, "maskCv2Wts.h5"))

        joblib.dump(hiAndRulWts, os.path.join(modelDir, "hiAndRulWts.h5"))



    # --------------------------------
    # Load weight for each block
    # of the model
    # --------------------------------
    def load_weights(self, modelDirComp1, modelDirComp2, modelDirComp3, modelDirSys,type = "subclass"):
        aiLogger.info(f"Loading the model at the direction: {modelDirComp1}, {modelDirComp2}, {modelDirComp3}, {modelDirSys}")
        encoderCr1Wts      = joblib.load(os.path.join(modelDirComp1, "encoderWts.h5"))
        decoderCr1Wts      = joblib.load(os.path.join(modelDirComp1, "decoderWts.h5"))
        maskCr1Wts         = joblib.load(os.path.join(modelDirComp1, "maskWts.h5"))

        encoderCr2Wts      = joblib.load(os.path.join(modelDirComp2, "encoderWts.h5"))
        decoderCr2Wts      = joblib.load(os.path.join(modelDirComp2, "decoderWts.h5"))
        maskCr2Wts         = joblib.load(os.path.join(modelDirComp2, "maskWts.h5"))

        encoderCv2Wts      = joblib.load(os.path.join(modelDirComp3, "encoderWts.h5"))
        decoderCv2Wts      = joblib.load(os.path.join(modelDirComp3, "decoderWts.h5"))
        maskCv2Wts         = joblib.load(os.path.join(modelDirComp3, "maskWts.h5"))

        hiAndRulWts         = joblib.load(os.path.join(modelDirSys, "hiAndRulWts.h5"))

        self.encoderCr1.set_weights(encoderCr1Wts)
        self.decoderCr1.set_weights(decoderCr1Wts)
        self.maskCr1.set_weights(maskCr1Wts)

        self.encoderCr2.set_weights(encoderCr2Wts)
        self.decoderCr2.set_weights(decoderCr2Wts)
        self.maskCr2.set_weights(maskCr2Wts)

        self.encoderCv2.set_weights(encoderCv2Wts)
        self.decoderCv2.set_weights(decoderCv2Wts)
        self.maskCv2.set_weights(maskCv2Wts)


        # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        # for layer in hiAndRulWts:
        #     print(layer)
        self.hiSystemAndRul.set_weights(hiAndRulWts)
        aiLogger.info(f"Loaded the model at the direction:{modelDirComp1}, {modelDirComp2}, {modelDirComp3}, {modelDirSys} successfully!!!")
    
    def loadWeightAndFreezePretrainedHI(self, modelDirComp1, modelDirComp2, modelDirComp3, type = "subclass"):
        aiLogger.info(f"Loading the model at the direction: {modelDirComp1}, {modelDirComp2}, {modelDirComp3}")
        encoderCr1Wts      = joblib.load(os.path.join(modelDirComp1, 'encoderWts.h5'))
        decoderCr1Wts      = joblib.load(os.path.join(modelDirComp1, "decoderWts.h5"))
        maskCr1Wts         = joblib.load(os.path.join(modelDirComp1, "maskWts.h5"))

        encoderCr2Wts      = joblib.load(os.path.join(modelDirComp2, "encoderWts.h5"))
        decoderCr2Wts      = joblib.load(os.path.join(modelDirComp2, "decoderWts.h5"))
        maskCr2Wts         = joblib.load(os.path.join(modelDirComp2, "maskWts.h5"))

        encoderCv2Wts      = joblib.load(os.path.join(modelDirComp3, "encoderWts.h5"))
        decoderCv2Wts      = joblib.load(os.path.join(modelDirComp3, "decoderWts.h5"))
        maskCv2Wts         = joblib.load(os.path.join(modelDirComp3, "maskWts.h5"))


        self.encoderCr1.set_weights(encoderCr1Wts)
        self.decoderCr1.set_weights(decoderCr1Wts)
        self.maskCr1.set_weights(maskCr1Wts)

        self.encoderCr2.set_weights(encoderCr2Wts)
        self.decoderCr2.set_weights(decoderCr2Wts)
        self.maskCr2.set_weights(maskCr2Wts)

        self.encoderCv2.set_weights(encoderCv2Wts)
        self.decoderCv2.set_weights(decoderCv2Wts)
        self.maskCv2.set_weights(maskCv2Wts)


        self.encoderCr1.trainable   = False
        self.decoderCr1.trainable   = False
        self.maskCr1.trainable      = False

        self.encoderCr2.trainable   = False
        self.decoderCr2.trainable   = False
        self.maskCr2.trainable      = False

        self.encoderCv2.trainable   = False
        self.decoderCv2.trainable   = False
        self.maskCv2.trainable      = False

        aiLogger.info(f"Loaded the model at the direction:{modelDirComp1, modelDirComp2, modelDirComp3} and freezed it successfully!!!")

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

