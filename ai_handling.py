# =============================================
# IMPORT
# =============================================
import  threading
import  queue
import  time
import tensorflow           as tf

from    typing          import List
from    utility.config  import BufferConfig
from    utility.logger  import aiLogger
from    threading       import Lock
from    preProcessingDataTEP   import PreProcessData
from    model.tepSystemModel   import SystemModel
import  os, warnings
import  random
import  json


class AiHandling():
    def __init__(self,config = None): 
        
        # ---------------------------
        # Loading config file
        # ---------------------------
        buffConfig              = BufferConfig()
        outputDir               = os.path.expanduser(config["dataDir"])
        
        # ----------------------------
        # Declaring AI model
        # ----------------------------
        componentModelPath      = "checkpoint/bestModel200Epoch/"
        config["component"]     = "Sys"
        self.vaeSys             = SystemModel(config = config, componentModelPath= componentModelPath)
        self.vaeSys.build(input_shape =[(None, config["seqLength"], config["inReac"]), 
                                        (None, 1, config["inReac"]), 
                                        (None, 1)])
        # vaeSys.summary()

        modelDirComp1 = outputDir + componentModelPath + "bestModelReac"
        modelDirComp2 = outputDir + componentModelPath + "bestModelSep"
        modelDirComp3 = outputDir + componentModelPath + "bestModelStrip"
        modelDirSys   = outputDir + componentModelPath + "bestModelSys"
        self.vaeSys.load_weights(modelDirComp1, modelDirComp2, modelDirComp3, modelDirSys)

        #----------------------------- 
        # Create thread and others
        #-----------------------------
        self.ai_handling            = threading.Thread(target=self.__ai_handling, daemon=False)
        
        self.__running              = False
        self.base_info              = {}
        self.utility_info           = {}

        #----------------------------
        # Condition object
        #----------------------------
        self.c_raw_ai_handling              = None
        self.c_processed_ai_handling        = threading.Condition()
        
        #----------------------------
        # Buffer
        #----------------------------
        self.raw_ai_handling_buffer         = None
        self.processed_ai_handling_buffer   = buffConfig.processedBuffer
        
        
    #==========================================
    # This function is used to start the thread
    #------------------------------------------         
    def start(self):
        self.__running = True
        self.ai_handling.start()
        
        
    #==========================================
    # This function is used to stop the thread
    #------------------------------------------   
    def stop(self):
        self.__running = False 
        
    #==========================================
    # This function expands layers buffer to 
    # multiple buffers connected to Ai Features
    #------------------------------------------     
    def __ai_handling(self):
        while True:
            try:
                # -------------------------------
                # wait till get new frame 
                # -------------------------------
                if self.c_raw_ai_handling is not None:
                    with self.c_raw_ai_handling:
                        self.c_raw_ai_handling.wait()

                # -------------------------------
                # Get data from buffer
                # -------------------------------
                if not self.raw_ai_handling_buffer.empty():
                    startTime = time.time()
                    base_info = self.raw_ai_handling_buffer.get()

                    signal      = base_info["signal"]
                    timestamp   = base_info["timestamp"]
                    env         = base_info["env"]
                    xPredict    = [signal, timestamp, env]  

                    hiReac, hiSep,  hiStrip, hiSystem, predictedRul, hiMatrixOutput    = self.vaeSys.getHIandRUL(input = xPredict)
                    
                    squeezedHiReac                                 = tf.squeeze(hiReac, axis=-1)
                    squeezedHiSep                                  = tf.squeeze(hiSep, axis=-1)
                    squeezedHiStrip                                = tf.squeeze(hiStrip, axis=-1)
                    squeezedHiSys                                  = tf.squeeze(hiSystem, axis=-1)
                    squeezedRUL                                    = tf.squeeze(predictedRul,axis=-1)
                    aiLogger.info(f"The time of ai_handling is {time.time() - startTime}")
                    aiLogger.debug(f"The value of HI is {squeezedHiReac}")
                # -------------------------------
                # get frame from full buffer and 
                # notify
                # -------------------------------
                # if self.processed_fps_control_buffer.full():
                #     self.processed_fps_control_buffer.get()  
                    
                if not self.__running:
                    break  


            except Exception as e:
                aiLogger.exception("__input_signal2ai Exception" + str(e))
    
    
 
        


        
        
        
            

                
                
                
                    
                    
                    
                    
                
                
            
        
        
        
        
        