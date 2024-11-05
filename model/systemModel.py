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
import  os, warnings, sys
import  joblib


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
# VAE Time (VariationalAutoencoderConv)
# ====================================
class VAEConvInterpretable(BaseVariationalAutoencoder):    
    def __init__(   self,  
                    hidden_layer_sizes, 
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

    # ================================
    # Create module Encoder separately
    # --------------------------------
    def get_encoder(self):
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
                        kernel_size = 3, 
                        strides     = 2, 
                        activation  = 'relu', 
                        padding     = 'same',
                        name        = f'enc_conv_{i}')(x)

        sysHILayer = Conv1D(    filters     = 5, 
                                kernel_size = 3, 
                                strides     = 2, 
                                activation  = 'relu', 
                                padding     = 'same',
                                name        = 'extractedSysHiLayer')(x)
        # print("------------------------------------------------")
        # print(sysHILayer.get_shape())
        # --------------------
        # Flat and create latent
        # --------------------
        x                       = Flatten(name='enc_flatten')(x)
        self.encLastDenseDim    = x.get_shape()[-1]                                 # save the dimensionality of this last dense layer before the hidden state layer. We need it in the decoder.
        
        
        # ----------------------
        # For VAE
        # ----------------------
        zMean                   = Dense(self.latent_dim, name="zMean")(x)           # Collect mean and logVar of the model
        zLogVar                 = Dense(self.latent_dim, name="zLogVar")(x)
        
        # encoder_output          = Sampling()([zMean, zLogVar])     
        encoder_output          = Dense(self.latent_dim, name="encoder_output")(x)
        self.encoder_output     = encoder_output

        # encoder                 = Model(encoder_inputs, [zMean, zLogVar, encoder_output], name="encoder")
        encoder                 = Model(encoder_inputs, [zMean, zLogVar, encoder_output, sysHILayer], name="encoder")
        return encoder


    def get_decoder(self):
        decoder_inputs = Input(shape=(self.latent_dim,), name="decoder_input")

        x = decoder_inputs
        x = Dense(self.encLastDenseDim, name="dec_dense", activation="relu")(x)
        x = Reshape(target_shape=(-1, self.hidden_layer_sizes[-1]), name="dec_reshape")(x)
            

        for i, num_filters in enumerate(reversed(self.hidden_layer_sizes[:-1])):
            x = Conv1DTranspose(    filters     = num_filters,
                                    kernel_size = 3,
                                    strides     = 2,
                                    padding     = "same",
                                    activation  = "relu",
                                    name        = f"dec_deconv_{i}",
                                )(x)

        # last de-convolution
        x = Conv1DTranspose(        filters     = self.feat_dim,
                                    kernel_size = 3,
                                    strides     = 2,
                                    padding     = "same",
                                    activation  = "relu",
                                    name        = f"dec_deconv__{i+1}",
                                )(x)

        x                       = Flatten(name="dec_flatten")(x)
        x                       = Dense(self.seq_len * self.feat_dim, name="decoder_dense_final")(x)
        self.decoder_outputs    = Reshape(target_shape=(self.seq_len, self.feat_dim))(x)
        decoder                 = Model(decoder_inputs, self.decoder_outputs, name="decoder")
        return decoder
    
    
# ====================================
#  Hi system block and RUL
# ====================================
class HiSystemAndRul(tf.keras.Model):
    def __init__(self):
        super(HiSystemAndRul, self).__init__()

        # ------------------------------
        # Define variables
        # ------------------------------
        self.lastLoss = self.add_weight(name='last_loss', shape=(), initializer='zeros', trainable=False)  # Define lastLoss as a trainable weight

        # ------------------------------
        # Input from latent space
        # ------------------------------
        inputRUL1 = Input(shape=(8, 5))
        inputRUL2 = Input(shape=(8, 5))
        inputRUL3 = Input(shape=(8, 5))

        inputHISys1 = Input(shape=(1,))
        inputHISys2 = Input(shape=(1,))
        inputHISys3 = Input(shape=(1,))


        concatenatedHiInput = Concatenate(axis=-1)([inputHISys1, inputHISys2, inputHISys3])
        hiSystem            = tf.keras.layers.Dense(1)(concatenatedHiInput)
        
        # print(hiSystem.shape)
        
        # ------------------------------
        # RUL
        # ------------------------------
        concatenatedRULInput = Concatenate(axis=-1)([inputRUL1, inputRUL2, inputRUL3])
        rulLayer1   = GRU(32, return_sequences=True)(concatenatedRULInput)
        rulLayer2   = GRU(16, return_sequences=True)(rulLayer1)
        rulLayer3   = GRU(3, return_sequences=False)(rulLayer2)
        rulLayer4   = Concatenate(axis=-1)([rulLayer3, hiSystem])
        rul         = tf.keras.layers.Dense(1, activation='linear')(rulLayer4)

        # print(rulLayer4.shape)
        # print(rulLayer3.shape)
        # print(hiSystem.get_shape())
        # # # ------------------------------
        # # # rul
        
        # Create the model
        self.model = Model(inputs=[inputRUL1, inputRUL2, inputRUL3,inputHISys1,inputHISys2,inputHISys3], outputs=[hiSystem,rul])
        


    def call(self, inputs):
        # print("----------------------------------")
        # print(inputs.shape)
        return self.model(inputs)

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
        
        vae                     = VAEConvInterpretable( seq_len             = config["seqLength"],  
                                                        feat_dim            = config["inFeatureDim"], 
                                                        latent_dim          = config["latentDim"], 
                                                        hidden_layer_sizes  = config["hiddenVAESize"], # [50,100,200]      #[80, 200, 250] 
                                                        )
        

        self.encoder     = vae.get_encoder()
        self.decoder     = vae.get_decoder()

        self.hiSystemAndRul  = HiSystemAndRul()

        # ----------------------------
        # For loss tracker
        # ----------------------------
        self.totalLossTracker               = Mean(name="totalLoss")
        self.reacConstructionTracker        = Mean( name="cstrReLoss" )
        self.reacKlLossTracker              = Mean(name="klReLoss")
        self.reacZLossTracker               = Mean(name = "zReLoss")
        self.rulLossTracker                 = Mean(name= "rulLoss")


        self.bestLoss                       = tf.Variable(float('inf'), trainable=False)


    # ==========================================
    # Call function
    # ------------------------------------------
    def call(self,input):

        seqEff,seqMani,seqCycle    = input
        # ------------------------------------
        # Sensor selection
        # ------------------------------------
        selectionMap,compIn          = self.sensorSelection(seqEff)
        
        selectedReac    = compIn[:,:,:self.config["featureDim"]]
        selectedSep     = compIn[:,:,self.config["featureDim"]:self.config["featureDim"]*2]
        selectedStrip   = compIn[:,:,self.config["featureDim"]*2:self.config["featureDim"]*3]

        
        # ------------------------------------
        # Encoder and Decoder for 3 components
        # ------------------------------------
        print("Encoder ")
        zMeanReac,  zLogReac,   zReac,  sysHILayerReac    = self.encoderReac(selectedReac)
        zMeanSep,   zLogSep,    zSep,   sysHILayerSep     = self.encoderSep(selectedSep)
        zMeanStrip, zLogStrip,  zStrip, sysHILayerStrip   = self.encoderStrip(selectedStrip)

        print(zReac.shape)

        # print("Latent size",z_mean.shape)

        print("Decoder")
        reconsReac  = self.decoderReac(zReac)
        reconsSep   = self.decoderSep(zSep)
        reconsStrip = self.decoderStrip(zStrip)

        # ------------------------------------
        # Create RUL and HI System
        # ------------------------------------

        print("HI and system rul")
        # hiSystem, predictedRul = self.hiSystemAndRul([zReac,zSep,zStrip])
        hiSystem, predictedRul = self.hiSystemAndRul([sysHILayerReac, sysHILayerSep, sysHILayerStrip,zReac, zSep, zStrip])
        return [reconsReac,reconsSep,reconsStrip,hiSystem,predictedRul]

    # ==========================================
    # Get custom layer
    # ------------------------------------------
    def getHIandRUL(self, input):
        seqXTrainEnv,seqXAllSensorTrain,seqTimeTrain  = input
        # ------------------------------------
        # Sensor selection
        # ------------------------------------
        # allSensor   = Input(shape=(128,47))(seqXAllSensorTrain)
        selectionMap,compIn = self.sensorSelection(seqXAllSensorTrain)
        
        selectedReac    = compIn[:,:,:self.config["featureDim"]]
        selectedSep     = compIn[:,:,self.config["featureDim"]:self.config["featureDim"]*2]
        selectedStrip   = compIn[:,:,self.config["featureDim"]*2:self.config["featureDim"]*3]

        # ------------------------------------
        # Encoder and Decoder for 3 components
        # ------------------------------------
        zMeanReac,  zLogReac,   zReac,  sysHILayerReac    = self.encoderReac(selectedReac)
        zMeanSep,   zLogSep,    zSep,   sysHILayerSep     = self.encoderSep(selectedSep)
        zMeanStrip, zLogStrip,  zStrip, sysHILayerStrip   = self.encoderStrip(selectedStrip)
        
        # ------------------------------------
        # Create RUL and HI System
        # ------------------------------------
        hiSystem, predictedRul = self.hiSystemAndRul([sysHILayerReac, sysHILayerSep, sysHILayerStrip,zReac, zSep, zStrip])

        return [zReac,zSep,zStrip,hiSystem,predictedRul]

    # ==========================================
    # Training (replace the default)
    # ------------------------------------------
    def train_step(self, input):
        with tf.GradientTape() as tape:

            # ====================================
            #           MODEL FLOW
            # ------------------------------------
            seqEff,seqMani,seqCycle     = input[0]
            rulY                        = input[1]
         
            # ------------------------------------
            # Encoder and Decoder for 3 components
            # ------------------------------------
            # print("Encoder ")
            zCr1,  sysHILayerCr1      = self.encoderReac(seqEff)
            reconsReac                = self.decoderReac(zCr1)

            # ------------------------------------
            # Create RUL and HI System
            # ------------------------------------

            # print("HI and system rul")
            # hiSystem, predictedRul = self.hiSystemAndRul([sysHILayerCr1,zCr1])

            # print("Loss funcion")
            # ====================================
            #               LOSS
            # ------------------------------------
            # ------------------------------------
            # Loss for reconstruction
            # ------------------------------------
            reconLossReac   = self._get_reconstruction_loss(seqEff, reconsReac)

            # ------------------------------------
            # Loss for RUL
            # ------------------------------------
            # rulLoss = tf.keras.losses.mse(predictedRul,rulY)
            # rulLoss = tf.reduce_mean(rulLoss)

            # totalLoss = reconLossReac +    rulLoss         
                        
            # ------------------------------------
            # Add the constrain to the latent space
            # ------------------------------------
            if self.config["latentConstrains"] :
                zReacLoss     = self._getConstrain(inputData = [zCr1,    seqCycle], type = "expertCurve",initCurve=64.8295,kRate=0.044, tMovement = 20)
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
        # self.rulLossTracker.update_state(rulLoss)

        # Construction loss 
        self.reacConstructionTracker.update_state(reconLossReac)
        
        # Latent loss
        self.reacZLossTracker.update_state(zReacLoss)
       
        # KL Loss
        # self.reacKlLossTracker.update_state(klLossReac)
        # self.sepKlLossTracker.update_state(klLossSep)
        # self.stripKlLossTracker.update_state(klLossStrip)

        return {
            "totalLoss"             : self.totalLossTracker.result(),
            # "rulLoss"               : self.rulLossTracker.result(),        
            "cstrReLoss"            : self.reacConstructionTracker.result(),
            "zReLoss"               : self.reacZLossTracker.result(),
        }

    # -----------------------------
    # Testing (replace the default)
    # -----------------------------
    def test_step(self, input): 

        # with tf.GradientTape() as tape:

        # ====================================
        #           MODEL FLOW
        # ------------------------------------
        seqXTrainEnv,seqXAllSensorTrain             = input[0]
        reacSensorY,sepSensorY,stripSensorY,rulY    = input[1]

        # ------------------------------------
        # Sensor selection
        # ------------------------------------
        # allSensor   = Input(shape=(128,47))(seqXAllSensorTrain)
        compIn          = self.sensorSelection(seqXAllSensorTrain)
        
        selectedReac    = compIn[:,:,:self.config["featureDim"]]
        selectedSep     = compIn[:,:,self.config["featureDim"]:self.config["featureDim"]*2]
        selectedStrip   = compIn[:,:,self.config["featureDim"]*2:self.config["featureDim"]*3]


        # print(selectedReac.shape, selectedSep.shape, selectedStrip.shape)
        
        # ------------------------------------
        # Encoder and Decoder for 3 components
        # ------------------------------------
        # print("Encoder ")
        zMeanReac,  zLogReac,   zReac    = self.encoderReac(selectedReac)
        zMeanSep,   zLogSep,    zSep     = self.encoderSep(selectedSep)
        zMeanStrip, zLogStrip,  zStrip   = self.encoderStrip(selectedStrip)

        # print("Latent size",z_mean.shape)

        # print("decoder")
        reconsReac  = self.decoderReac(zReac)
        reconsSep   = self.decoderSep(zSep)
        reconsStrip = self.decoderStrip(zStrip)

        # ------------------------------------
        # Create RUL and HI System
        # ------------------------------------

        # print("HI and system rul")
        hiSystem, predictedRul = self.hiSystemAndRul([zReac,zSep,zStrip])

        # print("Loss funcion")
        # ====================================
        #               LOSS
        # ------------------------------------
        # Loss function for sensor selection
        # ------------------------------------
        sensorReacLoss  = tf.keras.losses.mse(selectedReac, reacSensorY)
        # sensorReacLoss *= self.config["latentDim"]
        sensorReacLoss   = tf.reduce_mean(sensorReacLoss)

        sensorSepLoss   = tf.keras.losses.mse(selectedSep,sepSensorY)
        # sensorSepLoss *= self.config["latentDim"]
        sensorSepLoss   = tf.reduce_mean(sensorSepLoss)

        sensorStripLoss  = tf.keras.losses.mse(selectedStrip,stripSensorY)
        # sensorStripLoss *= self.config["latentDim"]
        sensorStripLoss  = tf.reduce_mean(sensorStripLoss)


        # ------------------------------------
        # Loss for reconstruction
        # ------------------------------------

        reconLossReac   = tf.keras.losses.mse(reconsReac,   reacSensorY)
        reconLossReac   = tf.reduce_mean(reconLossReac)
        reconLossSep    = tf.keras.losses.mse(reconsSep,    sepSensorY)
        reconLossSep    = tf.reduce_mean(reconLossSep)
        reconLossStrip  = tf.keras.losses.mse(reconsStrip,  stripSensorY)
        reconLossStrip  = tf.reduce_mean(reconLossStrip)
        # ------------------------------------
        # Loss for RUL
        # ------------------------------------
        rulLoss = tf.keras.losses.mse(predictedRul,rulY)
        rulLoss = tf.reduce_mean(rulLoss)

        totalLoss = (sensorReacLoss + sensorSepLoss + sensorStripLoss) +            \
                    (reconLossReac + reconLossSep + reconLossStrip) +               \
                    2*rulLoss 
            
        # --------------------------------
        # Update losses
        # --------------------------------
        self.totalLossTracker.update_state(totalLoss)
        self.rulLossTracker.update_state(rulLoss)

        # Sensor loss
        self.reacSensorTracker.update_state(sensorReacLoss)
        self.sepSensorTracker.update_state(sensorSepLoss)
        self.stripSensorTracker.update_state(sensorStripLoss)

        # Construction loss 
        self.reacConstructionTracker.update_state(reconLossReac)
        self.sepConstructionTracker.update_state(reconLossSep)
        self.stripConstructionTracker.update_state(reconLossStrip)

        # KL Loss
        # self.reacKlLossTracker.update_state(klLossReac)
        # self.sepKlLossTracker.update_state(klLossSep)
        # self.stripKlLossTracker.update_state(klLossStrip)
      

        return {
            "totalLoss"             : self.totalLossTracker.result(),
            "rulLoss"               : self.rulLossTracker.result(),
            "ssReLoss"              : self.reacSensorTracker.result(),
            "ssSeLoss"              : self.sepSensorTracker.result(),
            "ssStLoss"              : self.stripSensorTracker.result(),
            "cstrReLoss"            : self.reacConstructionTracker.result(),
            "cstrSeLoss"            : self.sepConstructionTracker.result(),
            "cstrStLoss"            : self.stripConstructionTracker.result(),
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
    def _getConstrain(self,inputData, type="expertCurve",alpha = 1, initCurve = 64.8295, kRate = 0.044, tMovement = 20):
        if type == "trendability":
            x, t = inputData
            x_mu = tf.reduce_mean(x)        # Calculate means
            t_mu = tf.reduce_mean(t)        # Calculate means
            
            # -------------------------
            # Calculate covariance and 
            # standard deviation
            # -------------------------
            cov = tf.reduce_sum((x - x_mu) * (t - t_mu))
            std = tf.math.sqrt(tf.reduce_sum(tf.square(x - x_mu))) * tf.math.sqrt(tf.reduce_sum(tf.square(t - t_mu)))

            # -------------------------
            # Check if std is finite and 
            # greater than zero
            # -------------------------
            loss = tf.cond(tf.math.is_finite(std) & (std > 0), lambda: cov / (std), lambda: self.bestLoss)  # Use last loss if std is not finite
            self.add_udpate(self.bestLoss.assign(loss))         # Update lastLoss if std is valid 
            loss = tf.abs(1/loss)                               # Making positive values and increase

        elif type == "reliability":
            # A, B, C, BETA, P    = 0.012, 4.95, 0.89, 6.7, 0.5
            A = tf.constant(0.012, dtype=tf.float32)
            B = tf.constant(4.95, dtype=tf.float32)
            C = tf.constant(0.89, dtype=tf.float32)
            BETA = tf.constant(6.7, dtype=tf.float32)
            P = tf.constant(0.5, dtype=tf.float32)

            x, t        = inputData
            exp_part    = tf.pow(-tf.math.log(1 - P), -1 / BETA)
            HI          = C - tf.pow((t * exp_part) * A, B)
            # Use tf.where to ensure no negative HI values
            HI          = tf.where(HI < 0, tf.zeros_like(HI), HI)
            loss        = tf.reduce_mean(tf.abs(x - HI))    



            # x, t                = inputData
            # HI                  = C - ((t * (-np.log(1 - P))**(-1/BETA))*A)**B
            # mask                = tf.math.less(HI,0)
            # HI                  = tf.where(mask,0.0,HI)
            # loss                = tf.reduce_mean(tf.abs(x-HI))

        elif type == "expertCurve":
            x, t    = inputData
            y       = initCurve + 0.0248*tf.exp(kRate*(t+ tMovement))
            loss    = tf.reduce_mean(tf.square(y - x))
            
        return alpha*loss


    # --------------------------------
    # Save weight of each block
    # --------------------------------
    def save_weights(self, modelDir, type = "subclass"):

        encoderReacWts      = self.encoder.get_weights()
        decoderReacWts      = self.decoder.get_weights()
        hiAndRulWts         = self.hiSystemAndRul.get_weights()
        
        joblib.dump(encoderReacWts, os.path.join(modelDir, "encoderReacWts.h5"))
        joblib.dump(decoderReacWts, os.path.join(modelDir, "decoderReacWts.h5"))
        joblib.dump(hiAndRulWts, os.path.join(modelDir, "hiAndRulWts.h5"))



    # --------------------------------
    # Load weight for each block
    # of the model
    # --------------------------------
    def load_weights(self, model_dir, type = "subclass"):
        aiLogger.info(f"Loading the model at the direction: {model_dir}")
        encoderReacWts      = joblib.load(os.path.join(model_dir, "encoderReacWts.h5"))
        decoderReacWts      = joblib.load(os.path.join(model_dir, "decoderReacWts.h5"))
        hiAndRulWts         = joblib.load(os.path.join(model_dir, "hiAndRulWts.h5"))
            
      
        self.encoderReac.set_weights(encoderReacWts)
        self.decoderReac.set_weights(decoderReacWts)
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

        # ---------------------
        # Save parameter if possible
        if config is not None:
            parameters = {  "seqLength"             : config["seqLength"],
                            "featureDim"            : config["featureDim"],
                            "latentDim"             : config["latentDim"],
                            "hiddenVAESize"         : config["hiddenVAESize"],
                            }
            parameterFile = os.path.join(model_dir, "parameters.pkl")
            joblib.dump(parameters, parameterFile)
        aiLogger.info(f"Saved the model at the direction:{model_dir} successfully!!!")

