# =============================================
# IMPORT
# =============================================
from utility.logger         import aiLogger
from keras.callbacks        import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
from model.systemModel      import SystemModel
from model.systemModel      import SubclassModelCheckpoint
from preProcessingData      import PreProcessData
from   keras.optimizers     import Adam
from datetime               import datetime
import tensorflow           as tf
import pandas               as pd
import matplotlib.pyplot    as plt
import numpy                as np
import os, warnings
import random
import json
import time


# =============================================
# MODEL WITH TEP
# =============================================
class TepTrainerTester(object):
    def __init__(self):

        # ---------------------------
        # Loading config file
        # ---------------------------
        with open('utility/modelConfig.json', 'r') as configFile:
            configData = json.load(configFile)
        config = configData["HiSystem"]  

        # ---------------------------
        # Set debug level and 
        # ---------------------------
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
        warnings.filterwarnings('ignore') 

        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # disabling gpu usage because my cuda is corrupted, needs to be fixed. 
        np.random.seed(1)
        random.seed(1)
        tf.random.set_seed(1)


        # ==============================
        # Config and data preprocessing
        # ==============================
        preData             = PreProcessData()
        rawData             = preData.loadDataFromH5Format(case="all")
        withRULData         = preData.addRULAndEnvironment(rawData)
        withoutInitRawData  = preData.removeInitState(inData= withRULData)

        # ---------------------------
        # For data
        cleanData           = preData.cleanData(rawSignal           =   withoutInitRawData,
                                                columnName          =   preData.allWithoutLabel,
                                                getMovingAverage    =   False,
                                                flipToDown          =   False)
        # ---------------------------
        # For label, flip only
        cleanData           = preData.cleanData(rawSignal           =   cleanData,
                                                columnName          =   preData.degraLiq,
                                                getMovingAverage    =   False,
                                                flipToDown          =   False)
        norData             = preData.normalizeDataEachModes(rawData    =   cleanData,
                                                             headline   =   preData.headline["allWithoutLabelNoBias"])   # Normalize the data
  

        # nan_count_per_column = norData[2].isna().sum()
        # print("NaN count per column:\n", nan_count_per_column)


        # ---------------------------
        # Extract the data in each Case
        data                = norData[config["case"]]
        trajectID           = data["Id"]
        reactor             = data[preData.headline["reactor"]]
        separator           = data[preData.headline["separator"]]
        stripper            = data[preData.headline["stripper"]]
        allSensor           = data[preData.headline["allWithoutLabelNoBias"]]
        rul                 = data["rul"]
        timeTraj            = data["Time"]
        # print(rul)
        env                 = data["environmentID"]
        # print(rul)
        # print(data.shape,reactor.shape, rul.shape)
        
        # ----------------------------
        # Split train and test according 
        # to IDs
        # ----------------------------
        # reactor, separator,stripper, allSensor, rul = preData.splitTrajectoryTrainVal(inData        = [reactor, separator,stripper, allSensor, rul], 
        #                                                                               trainPercent  = 0.8,
        #                                                                               shuffle       = True )  
        
        
        # trainReac,valReac,           = preData.splitTrajectoryTrainVal(inData = norData, case = 'all', trainPercent = 0.8, returnComponentSensor = True)

        # ---------------------------
        # For training
        seqXReacTrain,          \
        seqXSepTrain,           \
        seqXStripTrain,         \
        seqXAllSensorTrain,     \
        seqXRulTrain,           \
        seqXTrainEnv,seqTime        = preData.getTensorData(dataList    = [reactor,separator,stripper,allSensor,rul,env,timeTraj],
                                                            trajectId   = trajectID,  
                                                            shuffle     = True,      
                                                            seqLength   = config["seqLength"],
                                                            overlap     = config["overLap"]         
                                                            )
        seqXRulTrain        = seqXRulTrain[:,-1,:]      # get the last element in each window from rul
        seqTimeTrain        = seqTime[:,-1,:]
        
        print(seqXRulTrain)
        # print(seqXAllSensorTrain.shape)                                                                          
        # For validation
        # if config["validation"]:
        #     seqXReacVal,seqYReacVal           = preData.getTensorData(  pandasData  = reactor,
        #                                                                 component   = config["component"],  
        #                                                                 case        = config["case"],
        #                                                                 includeEnvId= False
        #                                                                 )

        # ==============================
        # Declare model
        # ==============================
        vae = SystemModel(config = config)   
        vae.build(input_shape=[(None,config["seqLength"], 1),(None,config["seqLength"],config["allSensors"]),(None,config["seqLength"], 1)])  # Batch size is None, feature size is 32
        vae.summary()

        # ==============================
        # Complite and train
        # ==============================
        vae.compile(optimizer=Adam(learning_rate = 0.01))

        # ==============================
        # Define callbacks with loss in the filename
        # ==============================
        saveCkpt            = ModelCheckpoint(  filepath            = 'model_epoch{epoch:02d}_totalLoss{totalLoss:.4f}.h5',  # Save filename with epoch and validation loss
                                                monitor             = 'totalLoss', 
                                                save_best_only      = True, 
                                                save_weights_only   = True,                                            # save weight only for subclass method      
                                                mode                = 'min',                                           # Minimize the validation loss
                                                verbose             = 1)                                               # Print messages when saving

        # --------------------------
        # filepath = './checkpoint/bestModel_{epoch}_totalLoss{totalLoss:.4f}'
        # --------------------------
        saveCheckpoint  =   SubclassModelCheckpoint(    customSaveFunction      = vae.save,  # Your custom save function
                                                        filepath                = './checkpoint/bestModel/',     # Where the model should be saved
                                                        config                  = config,
                                                        monitor                 = "totalLoss",                         # Monitoring validation loss
                                                        mode                    = 'min',                                 # 'min' because we want to minimize loss
                                                        save_best_only          = True,                        # Only save the best version
                                                        verbose                 = 1                                   # Show a message when saving
                                                        )

        log_dir = "checkpoint/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        early_stop_loss     = 'totalLoss'
        early_stop_callback = EarlyStopping(monitor=early_stop_loss, min_delta = 0.0001, patience=10) # change to 5
        reduceLR            = ReduceLROnPlateau(monitor='totalLoss', factor=0.1, patience=3)

        history = vae.fit(  x           = [seqXTrainEnv,seqXAllSensorTrain,seqTimeTrain], 
                            y           = [seqXRulTrain],
                            batch_size  = config["batchSize"],
                            epochs      = config["epochs"],
                            shuffle     = True,
                            callbacks   = [early_stop_callback, reduceLR, saveCheckpoint, tensorboardCallback],
                            verbose     = 1,
                            # validation_data = (seqXReacVal,seqXReacVal)
        )

        # ==============================
        # For saving the model
        # ==============================
        train_loss  = history.history["totalLoss"][-1]             #Plot the training loss
        dir         = config["ckptPath"] + "/" + "loss" + f"{train_loss:.2f}" + "/"


        plt.plot(history.history["totalLoss"])
        plt.plot(history.history["rulLoss"])
        plt.plot(history.history["ssReLoss"])
        plt.plot(history.history["ssSeLoss"])
        plt.plot(history.history["ssStLoss"])
        plt.plot(history.history["cstrReLoss"])
        plt.plot(history.history["cstrSeLoss"])
        plt.plot(history.history["cstrStLoss"])

        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["totalLoss","rulLoss","ssReLoss","ssSeLoss", "ssStLoss", "cstrReLoss", "cstrSeLoss", "cstrStLoss"],loc ="upper right")

        os.makedirs(dir, exist_ok=True)
        plt.savefig(dir + 'loss' + str(train_loss) + ".png")
        vae.save(dir)                                           # Save the model
        

#=====================================
#      MAIN FUNCTION
#=====================================    
def main():
    start       = time.time()
    trainer     = TepTrainerTester()
    end         = time.time()
    aiLogger.info(f"The training time is {(end- start)/60}min")

if __name__ == '__main__':
    main()

