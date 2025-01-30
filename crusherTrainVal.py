# =============================================
#                   IMPORT
# =============================================
from keras.optimizers                   import Adam
from keras.callbacks                    import EarlyStopping, ReduceLROnPlateau
from preProcessingDataTEP               import PreProcessData
# from preprocessingData2                 import PreProcessData
# from model.systemModel                  import SystemModel
# from model.systemModel               import SubclassModelCheckpoint
from model.componentModel               import mainModel
from model.componentModel               import SubclassModelCheckpoint

# from model.componentModel_RULonly       import SystemModel
# from model.componentModel_RULonly       import SubclassModelCheckpoint

from tqdm                               import tqdm
from datetime                           import datetime
import tensorflow                       as tf
import numpy                            as np 
import matplotlib.pyplot                as plt
import os, warnings
import random
import json
import package

package.include()

# ===========================
# MODEL WITH CRUSHER
# ===========================
class CrusherTrainerTester():
    def __init__(self,model = None, config = None):
        self.config                 = config
        self.model                  = model
        package.include()
        self.outputDir              = os.path.expanduser(self.config["dataDir"])
        # --------------------------
        # Add callback
        # --------------------------
        self.saveCheckpoint  =   SubclassModelCheckpoint(   customSaveFunction      = self.model.save,  # Your custom save function
                                                            filepath                = self.outputDir + 'checkpoint/bestModel/',     # Where the model should be saved
                                                            config                  = config,
                                                            monitor                 = "totalLoss",                          # Monitoring validation loss
                                                            # monitor                 = "rulLoss",
                                                            mode                    = 'min',                                # 'min' because we want to minimize loss
                                                            save_best_only          = True,                                 # Only save the best version
                                                            verbose                 = 1                                     # Show a message when saving
                                                            )

        log_dir = self.outputDir + "checkpoint/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

        # --------------------------
        self.tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        earlyStopLoss       = 'totalLoss'
        self.earlyStop_Callback = EarlyStopping(monitor=earlyStopLoss, min_delta = 0.01, patience=10) # change to 5
        self.reduceLR            = ReduceLROnPlateau(monitor='totalLoss', factor=0.1, patience=3)

    # ==============================
    # Train function
    # ------------------------------
    def compileModel(self,):
        model.compile( optimizer   =   Adam(learning_rate= self.config["learningRate"]))
    # ==============================
    # Train function
    # ------------------------------
    def train(self,seqXTrain, seqYTrain, seqXVal =None , seqYVal =None,):
        history = self.model.fit(   x           = seqXTrain, 
                                    y           = seqYTrain,
                                    batch_size  = self.config["batchSize"],
                                    epochs      = self.config["nEpoch"],
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
    myConfig   = config["tep"]
   
    # ====================================
    # Preprocessing, Model, Trainer
    # ------------------------------------
    preData     = PreProcessData(config = myConfig)                                         # Preprocessing data
    model       = mainModel(myConfig)                                                       # Model
    if myConfig["component"] == "system":
        model.build(input_shape=[(None, myConfig["seqLength"], myConfig["inCr1"]),(None, myConfig["seqLength"], myConfig["inCr2"]), (None, myConfig["seqLength"], myConfig["inCv2"])])
    else:
        model.build(input_shape=(None, myConfig["seqLength"], myConfig["in" + myConfig["component"]]))
    model.summary()

    trainer     = CrusherTrainerTester(model = model, config = myConfig)                    # Trainer
    trainer.compileModel()

    seqXTrain,seqYTrain     = preData.getTrainData()
    history                 = trainer.train(seqXTrain   = seqXTrain, seqYTrain = seqYTrain)
                                                                   
    
# 41,30,31




    
