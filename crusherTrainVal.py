# =============================================
#                   IMPORT
# =============================================
from keras.optimizers                   import Adam
from keras.callbacks                    import EarlyStopping, ReduceLROnPlateau
from preProcessingData                  import PreProcessData
from model.systemModel                  import SystemModel
from model.systemModel                  import SubclassModelCheckpoint

from tqdm                               import tqdm
from datetime                           import datetime
import tensorflow                       as tf
import numpy                            as np 
import matplotlib.pyplot                as plt
import os, warnings
import random
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
np.random.seed(1)
random.seed(1)

# ===========================
# MODEL WITH CRUSHER
# ===========================
class CrusherTrainerTester():
    def __init__(self,model = None):
        self.model                  = model

        # --------------------------
        # Add callback
        # --------------------------
        self.saveCheckpoint  =   SubclassModelCheckpoint(   customSaveFunction      = self.model.save,  # Your custom save function
                                                            filepath                = './checkpoint/bestModel/',     # Where the model should be saved
                                                            config                  = config,
                                                            monitor                 = "totalLoss",                         # Monitoring validation loss
                                                            mode                    = 'min',                                 # 'min' because we want to minimize loss
                                                            save_best_only          = True,                        # Only save the best version
                                                            verbose                 = 1                                   # Show a message when saving
                                                            )

        log_dir = "checkpoint/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

        # --------------------------
        # 
        self.tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        earlyStopLoss       = 'totalLoss'
        self.earlyStop_Callback = EarlyStopping(monitor=earlyStopLoss, min_delta = 0.0001, patience=10) # change to 5
        self.reduceLR            = ReduceLROnPlateau(monitor='totalLoss', factor=0.1, patience=3)

    # ==============================
    # Train function
    # ------------------------------
    def compileModel(self,):
        model.compile( optimizer   =   Adam())
    # ==============================
    # Train function
    # ------------------------------
    def train(self,seqXTrain, seqYTrain, seqXVal, seqYVal,vaeConfig):
        
        history = self.model.fit(  x            = seqXTrain, 
                                    y           = seqYTrain,
                                    batch_size  = config["batchSize"],
                                    epochs      = config["epochs"],
                                    shuffle     = True,
                                    callbacks   = [self.earlyStop_Callback, self.reduceLR, self.saveCheckpoint, self.tensorboardCallback],
                                    verbose     = 1,
                            # validation_data = (seqXReacVal,seqXReacVal)
        )
    
        return history
    
    # ==============================
    # Evaluation function
    # ------------------------------
    def evaluate(self):
        pass


# ============================
#       MAIN FUNCTION
# ============================
if __name__ == '__main__':
    # ====================================
    # Set debug level and 
    # ------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
    warnings.filterwarnings('ignore') 

    np.random.seed(1)
    random.seed(1)
    tf.random.set_seed(1)

    # ====================================
    # Config
    # ------------------------------------
    with open('utility/modelConfig.json', 'r') as configFile:
        config = json.load(configFile)
    myConfig   = config["vae"]
   

    # ====================================
    # Preprocessing, Model, Trainer
    # ------------------------------------
    preData     = PreProcessData(config = myConfig)                    # Preprocessing data
    model       = SystemModel(myConfig)                                # Model
    trainer     = CrusherTrainerTester(model = model)                  # Trainer
    trainer.compileModel()

    seqXTrain, seqYTrain        = preData.getTrainData()
    history     = trainer.train(    seqXTrain   = seqXTrain,
                                    seqYTrain   = seqYTrain,
                                    epochs      = 10, 
                                    batch_size  = 32,
                                )
    






    
