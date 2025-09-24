# =============================================
#                   IMPORT
# =============================================
from keras.optimizers                   import Adam
from keras.callbacks                    import EarlyStopping, ReduceLROnPlateau
from preProcessingDataTEP               import PreProcessData
from model.componentModel               import mainModel
from model.tepSystemModel               import SystemModel
from model.componentModel               import SubclassModelCheckpoint
import pandas                           as pd
from tqdm                               import tqdm
from datetime                           import datetime
import time
import tensorflow                       as tf
import numpy                            as np 
import matplotlib.pyplot                as plt
import os, warnings
import random
import json
import package
import  threading
from utility.logger import aiLogger

package.include()

# ===========================
# MODEL WITH CRUSHER
# ===========================
class AiProcessor():
    def __init__(self,model = None, config = None):
        self.config                 = config
        self.model                  = model

        # --------------------------------
        # For thread
        # --------------------------------
        self._ai_thread             = threading.Thread(target=self.__ai_processing,daemon=False)   # Declare backbone_thread with __ai_process function
    

        # --------------------------------
        # For AI
        # --------------------------------
        # ---------------------------
        # Loading config file
        # ---------------------------
        with open('utility/modelConfig.json', 'r') as configFile:
            configData = json.load(configFile)
        config = configData["tep"]  

        # ---------------------------
        # Set debug level and 
        # ---------------------------
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
        warnings.filterwarnings('ignore') 

        os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
        np.random.seed(1)
        random.seed(1)
        tf.random.set_seed(1)

        outputDir               = os.path.expanduser(config["dataDir"])
        totalTraject            = 100
        normalizeHI             = True
        showRedundantShutdown   = False

        preData             = PreProcessData(config=config)
        rawData             = preData.loadDataFromH5Format(dir              = outputDir, 
                                                        name             = "data/teps.h5",
                                                        case             = "all", 
                                                        reduceSampleRate = True, 
                                                        removeHigherThr  = False)

        withRULData         = preData.addRULAndEnvironment(rawData)
        # withoutInitRawData  = preData.removeInitState(inData= withRULData)



        # ---------------------------
        # For data
        cleanData           = preData.cleanData(rawSignal           =   withRULData,
                                                columnName          =   preData.allWithoutLabel,
                                                getMovingAverage    =   False,
                                                flipToDown          =   False)
        # ---------------------------
        # For label, flip only
        cleanData           = preData.cleanData(rawSignal           =   cleanData,
                                                columnName          =   preData.degraLiq,
                                                getMovingAverage    =   False,
                                                flipToDown          =   False)
        norData             = preData.normalizeDataEachModes(rawData    = cleanData,
                                                     headline   =   preData.headline["allWithoutLabelNoBias"] + ["rul"])    # Normalize the data

        config["component"] = "Sys"
        vaeSys = SystemModel(config = config)
        vaeSys.build(input_shape=[(None, config["seqLength"], config["inReac"]), (None, 1, config["inReac"]), (None, 1)])
        vaeSys.summary()


        modelDirComp1 = outputDir + "checkpoint/" + "bestModel34Reac"
        modelDirComp2 = outputDir + "checkpoint/" + "bestModel35Sep"
        modelDirComp3 = outputDir + "checkpoint/" + "bestModel36Strip"
        modelDirSys   = outputDir + "checkpoint/" + "bestModel38Sys"
        vaeSys.load_weights(modelDirComp1, modelDirComp2, modelDirComp3, modelDirSys)




        dfHIAll = pd.DataFrame(columns=[
                                'oriReacHI',
                                'oriSepHI',
                                'oriStripHI',
                                'oriSysHI',
                                'time'
                            ])

        environmentIdx      = 4
        data                = norData[environmentIdx]
        trajectID           = data["Id"]
        reactor             = data[preData.headline["reactor"]]
        separator           = data[preData.headline["separator"]]
        stripper            = data[preData.headline["stripper"]]
        allSensor           = data[preData.headline["allWithoutLabelNoBias"]]
        rul                 = data["rul"]
        env                 = data["environmentID"]
        timeTraj            = data["Time"]

        hiEnv1              = {}
        rulEnv1             = {}
        hiEnv1["Reac"]      = []
        rulEnv1["Reac"]     = []
        hiEnv1["Sep"]       = []
        rulEnv1["Sep"]      = []
        hiEnv1["Strip"]     = []
        rulEnv1["Strip"]    = []
        hiEnv1["Sys"]       = []
        rulEnv1["Sys"]     = []
        hiEnv1["multipliedMatrix"]       = []
        timeDraw            = []
        hiEnv1["timeDraw"] = []

    
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # This function is used to handle the 
    # thread
    # ------------------------------------
    def __ai_processing(self,):
        while True:
            try:
                

                        
                #--------------------------------------
                if not self._running:
                    break
            except Exception as e:
                aiLogger.exception("__ai_processing Exception" + str(e))
    # ==============================
    # Evaluation function
    # ------------------------------
    def predict(self):

        

        # timeEachCase        = withoutInitRawData[environmentIdx].groupby('Id',sort=False)['Time']
        # trajecTime          = [eachTrajectTime[1] for eachTrajectTime in timeEachCase]                  #Extract time from the tuple and groupby
        # for eachTrajectTime in timeEachCase:
        #     print(len(eachTrajectTime)) 

        for idx in range(totalTraject):
            getAtIdx = idx + 1
            print(f"Getting the HI at the idx {getAtIdx}")
            # seqXTrainEnv, seqXAllSensorTrain,seqTime     = preData.getTensorData(   dataList    = [env,allSensor,timeTraj],
            #                                                                         trajectId   = trajectID,
            #                                                                         getAtIdx    = getAtIdx,   
            #                                                                         seqLength   = config["seqLength"],
            #                                                                         overlap     = config["overLap"]
            #                                                                     )
            seqXAllSensorTrain,     \
            seqXRulTrain,           \
            seqXTrainEnv,seqTime                = preData.getTensorData(dataList    = [allSensor,rul,env,timeTraj],
                                                                        trajectId   = trajectID,  
                                                                        shuffle     = False,      
                                                                        getAtIdx    = getAtIdx,
                                                                        seqLength   = preData.config["seqLength"],
                                                                        overlap     = preData.config["overLap"]         
                                                                        )
            seqTimeTrain        = seqTime[:,-1,:]

            xPredict            = [seqXAllSensorTrain, seqTime[:,-1,:], seqXTrainEnv[:,-1,:]]  

            hiReac, hiSep,  hiStrip, hiSystem, predictedRul, hiMatrixOutput    = vaeSys.getHIandRUL(input = xPredict)
            # squeezedHiSystem                    = tf.reduce_mean(hiSystem,axis=-1)
            # print(squeezedHiSystem.shape)
            squeezedHiReac                                 = tf.squeeze(hiReac, axis=-1)
            squeezedHiSep                                  = tf.squeeze(hiSep, axis=-1)
            squeezedHiStrip                                = tf.squeeze(hiStrip, axis=-1)
            squeezedHiSys                                  = tf.squeeze(hiSystem, axis=-1)
            timeDraw.append(seqTimeTrain)


            dfTemp      = pd.DataFrame({'id'           : np.ones(squeezedHiReac.shape)*idx,
                                        'oriReacHI'    : squeezedHiReac.numpy(), 
                                        'oriSepHI'     : squeezedHiSep.numpy(),
                                        'oriStripHI'   : squeezedHiStrip.numpy(),
                                        'oriSysHI'     : squeezedHiSys.numpy(),
                                        'time'      : np.squeeze(seqTimeTrain)})
            

            dfHIAll     = pd.concat([dfHIAll,dfTemp],ignore_index=True)

        dfHIAll["reacHI"]   = dfHIAll["oriReacHI"].copy()
        dfHIAll["sepHI"]    = dfHIAll["oriSepHI"].copy()
        dfHIAll["stripHI"]  = dfHIAll["oriStripHI"].copy()

    def start(self,):
        self._running = True
        self._ai_thread.start()
        
    def stop(self):
        self._ai_thread.stop()
        time.sleep(1.5) 
        del self._ai_thread
        # self._running = False   






    
