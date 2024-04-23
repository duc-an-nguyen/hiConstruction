from tensorflow.keras.optimizers        import Adam
from tensorflow.keras.callbacks         import EarlyStopping, ReduceLROnPlateau
from preProcessingData                  import PreProcessData
from utility.config                     import rulVAECNNConfig
from model.vae_conv_model               import VariationalAutoencoderConv

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


# ==============================
# Config and data preprocessing
# ==============================
with open('model/modelConfig.json', 'r') as configFile:
    config = json.load(configFile)
 
preData             = PreProcessData()
rawData             = preData.dataLoading()

# cleanData           = preData.cleanData(    rawData,
#                                             getMovingAverage    = True, 
#                                             flipToDown          = True)

norData                 = preData.normalizeData(rawData[preData.crusher1Value])     # Normalize the data
healthyStateData        = preData.getHealthyState(norData)                          # Extract the healthy state of machine

inputX                  = healthyStateData[preData.manipulatedCr1]
outputY                 = healthyStateData[preData.efficiencyCr1]

trainXY,valXY           = preData.splitTrainVal(inputX,outputY)                     # Split input and efficiency variables into train and validation dataset

seqTrainXY              = preData.splitAndPadSequences(trainXY)                # Split them into sequences
seqValXY                = preData.splitAndPadSequences(valXY)                  # Split them into sequences

model                   = VariationalAutoencoderConv(config)

# model.summary()

model.compile(optimizer =   Adam(),
              loss      =   'mean_squared_error',
              metrics   =   ['mean_absolute_error'])

early_stop_loss     = 'loss'
early_stop_callback = EarlyStopping(monitor=early_stop_loss, min_delta = 1e-1, patience=5) 
reduceLR            = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5)



history = model.fit(
        x           =   seqTrainXY[0],
        y           =   seqTrainXY[1],
        batch_size  =   config.batch_size,
        epochs      =   config.n_epochs,
        shuffle     =   True,
        callbacks   =   [early_stop_callback, reduceLR],
        verbose     =   1,
        validation_data = ([reactorVal,sepVal,stripVal],seqValXY[1])
)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train","val"],loc ="upper left")

val_loss = history.history["val_loss"][-1]
plt.savefig(config.modelDir + "/" +  "loss" + str(val_loss) + ".png")

# ==============================    
# save model 
# ==============================
model.save(model_dir = config.modelDir, file_pref = config.savedDir)