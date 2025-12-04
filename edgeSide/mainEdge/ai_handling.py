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
        
        aiLogger.warning(f"We are using the tensorflow with the GPU of {tf.config.list_physical_devices('GPU')}")

        # ---------------------------
        # Loading config file
        # ---------------------------
        buffConfig              = BufferConfig()
        # outputDir               = os.path.expanduser(config["dataDir"])
        
        # ----------------------------
        # Declaring AI model
        # ----------------------------
        self.upperBound         = 0.1783
        self.thrHI              = -214
        self.checkHealthiedIdx  = 1

        componentModelPath      = "checkpoint/bestModel200Epoch/"
        config["component"]     = "Sys"
        self.vaeSys             = SystemModel(config = config, componentModelPath= componentModelPath)
        self.vaeSys.build(input_shape =[(None, config["seqLength"], config["inReac"]), 
                                        (None, 1, config["inReac"]), 
                                        (None, 1)])
        # vaeSys.summary()

        # modelDirComp1 = outputDir + componentModelPath + "bestModelReac"
        # modelDirComp2 = outputDir + componentModelPath + "bestModelSep"
        # modelDirComp3 = outputDir + componentModelPath + "bestModelStrip"
        # modelDirSys   = outputDir + componentModelPath + "bestModelSys"

        modelDirComp1 =  componentModelPath + "bestModelReac"
        modelDirComp2 =  componentModelPath + "bestModelSep"
        modelDirComp3 =  componentModelPath + "bestModelStrip"
        modelDirSys   =  componentModelPath + "bestModelSys"
        self.vaeSys.load_weights(modelDirComp1, modelDirComp2, modelDirComp3, modelDirSys)

        #----------------------------- 
        # Create thread and others
        #-----------------------------
        self.ai_handling            = threading.Thread(target=self.__ai_handling, daemon=False)
        
        self.__running              = False


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

                # ----------------------------------------
                # wait till get new frame 
                # ----------------------------------------
                with self.c_raw_ai_handling:
                    if self.raw_ai_handling_buffer.empty():
                        self.c_raw_ai_handling.wait()
        
                        

                # ----------------------------------------
                # Get data from buffer
                # ----------------------------------------
                if not self.raw_ai_handling_buffer.empty():
                    startTime = time.time()

                    # ---------------------------------
                    # Get data from previous thread
                    base_info, utility_info = self.raw_ai_handling_buffer.get()

                    curIdxHeath             = base_info["frameIdx"]
                    signal                  = base_info["signal"]
                    timestamp               = base_info["timestamp"]
                    env                     = base_info["env"]
                    xPredict                = [signal, timestamp, env]  


                    # ---------------------------------
                    # AI handling
                    hiReac, hiSep,  hiStrip, hiSystem, predictedRul, hiMatrixOutput    = self.vaeSys.getHIandRUL(input = xPredict)
                    
                    squeezedHiReac                                 = tf.squeeze(hiReac, axis=-1)
                    squeezedHiSep                                  = tf.squeeze(hiSep, axis=-1)
                    squeezedHiStrip                                = tf.squeeze(hiStrip, axis=-1)
                    squeezedHiSys                                  = tf.squeeze(hiSystem, axis=-1)
                    squeezedRUL                                    = tf.squeeze(predictedRul,axis=-1)

                    # -----------------------------------
                    # When the healthy position of each HI 
                    # is different, we move to make sure they 
                    # are nearly to be 1
                    # -----------------------------------
                    if curIdxHeath == 0:
                        shiftValues             = {}
                        highestHI               = max(squeezedHiReac,squeezedHiSep,squeezedHiStrip)
                        shiftValues["reacHI"]   = highestHI - squeezedHiReac
                        shiftValues["sepHI"]    = highestHI - squeezedHiSep
                        shiftValues["stripHI"]  = highestHI - squeezedHiStrip

                    reacHI, sepHI, stripHI, sysHI    = self.hiNormalization(squeezedHiReac,
                                                                            squeezedHiSep,
                                                                            squeezedHiStrip,
                                                                            hiMatrixOutput,
                                                                            shiftValues)
                    

                    # ---------------------------------
                    # Write to buffer to send
                    # ---------------------------------
                    utility_info["reacHI"]  = reacHI
                    utility_info["sepHI"]   = sepHI
                    utility_info["stripHI"] = stripHI
                    utility_info["sysHI"]   = sysHI
                    utility_info["rul"]     = squeezedRUL.numpy()
                    
                    self.processed_ai_handling_buffer.put((base_info, utility_info))
                    
                    with self.c_processed_ai_handling:
                        if (self.processed_ai_handling_buffer.qsize() >= 1):
                            self.c_processed_ai_handling.notify_all()   

                    aiLogger.info(f"The time of ai_handling is {time.time() - startTime}")
                    aiLogger.debug(f"The value of HI is {reacHI, sepHI, stripHI, sysHI}")
                # -------------------------------
                # get frame from full buffer and 
                # notify
                # -------------------------------
                    if self.processed_ai_handling_buffer.full():
                        self.processed_ai_handling_buffer.get()  
                        aiLogger.warning("The buffer <processed_ai_handling_buffer> is full, please check the size and time again to handle!!!")
                    
                if not self.__running:
                    break  

            except Exception as e:
                aiLogger.exception("__input_signal2ai Exception" + str(e))
    
    
    #==========================================
    # This funcion is used to normalize HI
    # -----------------------------------------
    def hiNormalization(self, reacHI, sepHI, stripHI, hiMatrixOutput, shiftValues):
        
        shiftedReac     = reacHI    + shiftValues["reacHI"]
        shiftedSep      = sepHI     + shiftValues["sepHI"]
        shiftedStrip    = stripHI   + shiftValues["stripHI"]

        reacHI      = (shiftedReac  - self.thrHI)/ (self.upperBound - self.thrHI)
        sepHI       = (shiftedSep   - self.thrHI)/ (self.upperBound - self.thrHI)
        stripHI     = (shiftedStrip - self.thrHI)/ (self.upperBound - self.thrHI)


        selectedFunc = hiMatrixOutput[0]
        if selectedFunc[0] > 0.5:
            sysHI        = min(reacHI,sepHI,stripHI)
        elif selectedFunc[1] > 0.5:
            sysHI        = max(reacHI,sepHI,stripHI)
        else:
            sysHI        = 0.3* reacHI + 0.3* sepHI +0.3*stripHI

        return reacHI.numpy(), sepHI.numpy(), stripHI.numpy(), sysHI.numpy()
    

    def updateFlWeight(self,reacWeight, sepWeight, stripWeight, sysWeight):
        pass
        # modelDirComp1 =  componentModelPath + "bestModelReac"
        # modelDirComp2 =  componentModelPath + "bestModelSep"
        # modelDirComp3 =  componentModelPath + "bestModelStrip"
        # modelDirSys   =  componentModelPath + "bestModelSys"
        # self.vaeSys.load_weights(modelDirComp1, modelDirComp2, modelDirComp3, modelDirSys)


        
        
        
            

                
                
                
                    
                    
                    
                    
                
                
            
        
        
        
        
        