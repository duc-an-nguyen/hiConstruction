from tensorflow.keras.optimizers        import Adam
from tensorflow.keras.callbacks         import EarlyStopping, ReduceLROnPlateau
from preProcessingData                  import PreProcessData
from utility.config                     import rulVAECNNConfig
from model.vaeConv1DCustom              import vaeConv1DCustom 
from tqdm                               import tqdm

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
# This class to train the VAE
# model with the application
# ===========================
class appVAE():
    def __init__(self):
        pass
    # ====================
    # Preprocessing data
    # --------------------
    @staticmethod
    def dataPreprocessing(forTraining = True):
        # ------------------------
        # Init and load data
        # ------------------------
        preData     = PreProcessData()
        rawData     = preData.loadCriticalData()
        norData     = preData.normalizeBuilinFunction(rawData[preData.crusher1Value])     # Normalize the data
        
        # ------------------------
        # Get healthy data only or
        # entire dataset
        # ------------------------
        if forTraining:
            healthyStateData,_      = preData.getHealthyState(norData)                          # Extract the healthy state of machine
            inputX                  = healthyStateData[preData.manipulatedCr1]
            # outputY                 = healthyStateData[preData.efficiencyCr1]
            trainXY,valXY           = preData.splitTrainVal(inputX  = inputX,
                                                            outputY = inputX)                   # Split input and efficiency variables into train and validation dataset
            seqTrainXY              = preData.splitAndPadSequences(trainXY,seqLength=vaeConfig['seqLength'])                     # Split them into sequences
            seqValXY                = preData.splitAndPadSequences(valXY, seqLength=vaeConfig['seqLength'])                       # Split them into sequences
        else:
            inputX                  = norData[preData.manipulatedCr1]
            # outputY                 = norData[preData.efficiencyCr1]
            seqValXY                = preData.splitAndPadSequences([inputX,inputX])                       # Split them into sequences
            seqTrainXY              = None
        

        return seqTrainXY,seqValXY
    
    # ====================
    # Application trainer
    # --------------------
    @staticmethod
    def trainer(seqTrainXY,seqValXY,vaeConfig):
        model                   =   vaeConv1DCustom(vaeConfig)
        model.compile(optimizer =   Adam(),
                    loss        =   'mean_squared_error',
                    metrics     =   ['mean_absolute_error'])

        early_stop_loss     = 'loss'
        early_stop_callback = EarlyStopping(monitor=early_stop_loss, min_delta = 1e-1, patience=10) 
        reduceLR            = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10)


        history = model.fit(    x           =   seqTrainXY[0],
                                y           =   seqTrainXY[1],
                                batch_size  =   vaeConfig["batchSize"],
                                epochs      =   vaeConfig["nEpoch"],
                                shuffle     =   False,
                                callbacks   =   [early_stop_callback, reduceLR],
                                verbose     =   1,
                                validation_data = (seqValXY[0],seqValXY[1])               
                            )
    
        return history,model

    # ====================    
    # Save model and image 
    # of training
    # --------------------
    @staticmethod
    def saveModelAndProgress(history,model,nameOfModel = "default"):
        # ------------------------------
        # Create model directory
        # ------------------------------
        typeOfModel = nameOfModel
        modelDir   = "ckpt/" + typeOfModel + str(vaeConfig["encLayer"]) + "_" + str(vaeConfig["decLayer"]) +  \
                    "latent" + "_" + str(vaeConfig["latentDim"])

        if not os.path.exists(modelDir + "/"):
            os.makedirs(modelDir + "/")

        # ------------------------------
        # Plot and save
        # ------------------------------
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train","val"],loc ="upper left")

        val_loss = history.history["val_loss"][-1]
        plt.savefig(modelDir + "/" +  "loss" + str(val_loss) + ".png")

        # ==============================    
        # save model 
        # ==============================
        model.save(model_dir = modelDir, file_pref = "crusher1")

    # ====================
    # Prediction
    # --------------------
    @staticmethod
    def predict(seqData,vaeConfig):
        
        print(len(seqData[0]))
        batchData      = vaeConv1DCustom.batchGenerator(seqData,batchSize=1)
        # print(type(batchData))
        model          = vaeConv1DCustom(vaeConfig)
        model.load_weights("ckpt/VAECNN1D_ManupulatedAndEfficiencyInputOutput[200, 100, 50]_[50, 100, 200]latent_2","crusher1")
        
        lossResults     = []
        latentResults   = []
        for batch in tqdm(batchData,total=len(seqData[0])):
            batchResult  = model(batch,
                                 returnLatent               = True,             
                                 returnReconstructionError  = True)
            lossResults.append(batchResult["reconstructionLoss"])
            latentResults.append(batchResult["latent"])

        return lossResults,latentResults
    



# ============================
#       MAIN FUNCTION
# ============================
if __name__ == '__main__':
    trainIsTrue = True
    # ----------------
    # Config
    # ----------------
    with open('model/modelConfig.json', 'r') as configFile:
        config = json.load(configFile)
    vaeConfig = config["vae"]
    
    if trainIsTrue:
        seqTrainXY,seqValXY     = appVAE.dataPreprocessing()
        history,model           = appVAE.trainer(seqTrainXY,seqValXY,vaeConfig)
        appVAE.saveModelAndProgress(history,model,nameOfModel= "manipulatedReconstruction")

    else:
        _,seqData               = appVAE.dataPreprocessing(forTraining = False)
        result                  = appVAE.predict(seqData,vaeConfig)

    
