from tensorflow.keras.optimizers        import Adam
from tensorflow.keras.callbacks         import EarlyStopping, ReduceLROnPlateau
from preProcessingData                  import PreProcessData
from utility.config                     import rulVAECNNConfig
from model.systemHiAndRulModel          import systemHiAndRul

import numpy                            as np 
import matplotlib.pyplot                as plt

import os, warnings
import random


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # disabling gpu usage because my cuda is corrupted, needs to be fixed. 
np.random.seed(1)
random.seed(1)


# ==============================
# Config and data preprocessing
# ==============================
config              = rulVAECNNConfig()


preData             = PreProcessData(seqLength  = config.seqLength,
                                     overLap    = config.overLap)

rawData,rulLabel,_  = preData.getDataset(   component           = "all", 
                                            removeInitState     = True,
                                            theCase             = config.theCase)    
cleanData           = preData.cleanData(    rawData,
                                            getMovingAverage    = True, 
                                            flipToDown          = True)
norData             = preData.normalizeData(cleanData)                # Normalize the data


trainXY,valXY       = preData.splitTrainVal(norData,rulLabel)       # Split data and label into train and validation dataeset
seqTrainXY          = preData.splitAllDataIntoSequences(trainXY)
seqValXY            = preData.splitAllDataIntoSequences(valXY)  


model = systemHiAndRul(config)

# model.summary()

model.compile(optimizer =   Adam(),
              loss      =   'mean_squared_error',
              metrics   =   ['mean_absolute_error'])

early_stop_loss     = 'loss'
early_stop_callback = EarlyStopping(monitor=early_stop_loss, min_delta = 1e-1, patience=5) 
reduceLR            = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5)

reactorInput        = seqTrainXY[0][:,:,:7]
sepInput            = seqTrainXY[0][:,:,7:12]
stripInput          = seqTrainXY[0][:,:,12:17]

reactorVal          = seqValXY[0][:,:,:7]
sepVal              = seqValXY[0][:,:,7:12]
stripVal            = seqValXY[0][:,:,12:17]




history = model.fit(
        x           =   (reactorInput,sepInput,stripInput),
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