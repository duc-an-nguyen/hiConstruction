import tensorflow as tf
import joblib 
import os
from keras.models               import Sequential
from keras.layers               import LSTM, Dense, Input, Flatten, GRU, RepeatVector,TimeDistributed
from tensorflow.keras.layers    import Layer
from tensorflow.keras.models    import Model
from tensorflow.keras.metrics   import Mean
from tensorflow.keras.backend   import random_normal
from tkan                       import TKAN
from abc                        import ABC



class Sampling(Layer,):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim   = z_mean.shape[1]
        # dim = tf.shape(z_mean)[1]
        epsilon = random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
# ================================
# This class build HI from RUL for 
# TEPs dataset
# ================================
class vaeTKAN(Model):
    def __init__(self,config = None,**kwargs):
        super(vaeTKAN, self).__init__(**kwargs)
        self.seq_len            = config['seqLength']
        self.feat_dim           = config['inFeatureDim']
        self.latent_dim         = config['latentDim']
        self.reconstruction_wt  = 3


        self.total_loss_tracker             = Mean(name="total_loss")
        self.reconstruction_loss_tracker    = Mean(name="reconstruction_loss" )
        self.kl_loss_tracker                = Mean(name="kl_loss")

        self.encoder    = self.getEncoder()
        self.decoder    = self.getDecoder()
        print(self.encoder.summary())
        print(self.decoder.summary())
    
    # -----------------------
    # Call function
    # -----------------------
    def call(self,input):
        z_mean, z_log_var, _  = self.encoder(input)
        # decoderOut      = self.decoder(encoderOut)
        return z_mean, z_log_var
    
    # -----------------------
    # Encoder
    # -----------------------
    def getEncoder(self,):

        encodeInput = Input(shape=(self.seq_len, self.feat_dim),name = 'encoder_input')
        x           = TKAN(40, return_sequences=True)(encodeInput)
        # x           = TKAN(20, return_sequences=True)(x)
        # x       = TKAN(50,  sub_kan_output_dim=20, sub_kan_input_dim=20, return_sequences=False)(x)  # Final layer with return_sequences=False
        x           = TKAN(10,  sub_kan_output_dim=128, return_sequences=False)(x)

        # -----------------------
        # Create the latent space
        # and sampling
        # -----------------------
        z_mean          = Dense(self.latent_dim, name="z_mean")(x)        # Connect output of TKAN to z_mean
        z_log_var       = Dense(self.latent_dim, name="z_log_var")(x)  # Connect output of TKAN to z_log_var
        encoderOut      = Sampling()([z_mean, z_log_var])  
       
        # -----------------------
        # Create the encoder model
        # -----------------------
        encoder = Model(inputs=encodeInput, outputs=[z_mean, z_log_var,encoderOut], name='encoder')
        # print(encoder.summary())
            
        return encoder
    
    # -----------------------
    # Decoder
    # -----------------------
    def getDecoder(self,):
        latentInputs    = Input(shape=(self.latent_dim,),name = 'decoder_input')
        repeatedLatent  = RepeatVector(self.seq_len)(latentInputs)

        x           = TKAN(10, return_sequences=True)(repeatedLatent)
        # x           = TKAN(20, return_sequences=True)(x)
        x           = TKAN(40, return_sequences=True)(x)

        decoderOut  = TimeDistributed(Dense(self.feat_dim))(x)

        decoder = Model(inputs=latentInputs, outputs=decoderOut, name='decoder')

        return decoder

    # -----------------------
    # Overide the train and 
    # test function
    # -----------------------
    def train_step(self, X):
        with tf.GradientTape() as tape:
            # print("Input shape",X.shape)
            inData = X[0]
            label  = X[1]
            # print(inData.shape)
            z_mean, z_log_var, z    = self.encoder(inData)
            reconstruction          = self.decoder(z)

            # -----------------------+
            # Caculate the loss function
            # -----------------------
            # reconstruction_loss     = self._get_reconstruction_loss(label, reconstruction)
            reconstruction_loss     = tf.keras.losses.mse(inData, reconstruction)
            reconstruction_loss    *= self.latent_dim                               # Scale by size of input

            kl_loss                 = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss                 = tf.reduce_mean(kl_loss)
            kl_loss                 *= -0.5
            # kl_loss = kl_loss / self.latent_dim
            total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss


        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # -----------------------
        # Update the loss
        # -----------------------
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        # self.latent_loss_tracker.update_state(penalty_loss)

        return {
            "loss":                 self.total_loss_tracker.result(),
            "reconstruction_loss":  self.reconstruction_loss_tracker.result(),
            "kl_loss":              self.kl_loss_tracker.result(),
            # "latent_loss": self.latent_loss_tracker.result(),
        }

    
    def test_step(self, X): 
        inData = X[0]
        label  = X[1]
        # print(inData.shape)
        z_mean, z_log_var, z    = self.encoder(inData)
        reconstruction          = self.decoder(z)

        # -----------------------+
        # Caculate the loss function
        # -----------------------
        # reconstruction_loss     = self._get_reconstruction_loss(label, reconstruction)
        reconstruction_loss     = tf.keras.losses.mse(inData, reconstruction)
        reconstruction_loss    *= self.latent_dim                               # Scale by size of input

        kl_loss                 = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss                 = tf.reduce_mean(kl_loss)
        kl_loss                 *= -0.5
        # kl_loss = kl_loss / self.latent_dim

        total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            # "latent_loss": self.latent_loss_tracker.result(),
        }
    

    #======================================
    # Save checkpoint
    #--------------------------------------
    # def save_weights(self, model_dir, file_pref): 
    #     encoder_wts = self.encoder.get_weights()
    #     decoder_wts = self.decoder.get_weights()
    #     joblib.dump(encoder_wts, os.path.join(model_dir, f'{file_pref}encoder_wts.h5'))
    #     joblib.dump(decoder_wts, os.path.join(model_dir, f'{file_pref}decoder_wts.h5'))
    
    # def save(self, model_dir, file_pref): 
    #     self.save_weights(model_dir, file_pref)
    #     dict_params = {
    #         'seq_len': self.seq_len,
    #         'feat_dim': self.feat_dim,
    #         'latent_dim': self.latent_dim,
    #         'reconstruction_wt': self.reconstruction_wt,
    #     }
    #     params_file = os.path.join(model_dir, f'{file_pref}parameters.pkl') 
    #     joblib.dump(dict_params, params_file)
    

    
    # def load_weights(self, model_dir, file_pref):
    #     encoder_wts = joblib.load(os.path.join(model_dir, f'{file_pref}encoder_wts.h5'))
    #     decoder_wts = joblib.load(os.path.join(model_dir, f'{file_pref}decoder_wts.h5'))

    #     print(len(decoder_wts))
    #     for value in decoder_wts:
    #         print(value.shape)
    #     # print(encoder_wts)

    #     # self.encoder.set_weights(encoder_wts)
    #     # self.decoder.set_weights(decoder_wts)

if __name__ == '__main__':
    model = vaeTKAN()


