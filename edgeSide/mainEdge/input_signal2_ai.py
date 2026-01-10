# =============================================
# IMPORT
# =============================================
from typing                  import List
from utility.config          import BufferConfig
from utility.logger          import aiLogger
from threading               import Lock
from preProcessingDataTEP    import PreProcessData

import threading
import queue
import time

import os, warnings
import random
import json


class InputSignal2Ai():
    def __init__(self, config): 
        
        # ---------------------------
        # Loading config file
        # ---------------------------
        # outputDir               = os.path.expanduser(config["dataDir"])
        
        # ----------------------------
        # Handle the data
        # ----------------------------
        environmentIdx      = 4
        self.preData        = PreProcessData(config=config)
        rawData             = self.preData.loadDataFromH5Format(dir              = "./", 
                                                                name             = "data/teps.h5",
                                                                case             = "all", 
                                                                reduceSampleRate = True, 
                                                                removeHigherThr  = False)
        withRULData         = self.preData.addRULAndEnvironment(rawData)
        originalSensor      = withRULData.copy()
        norData             = self.preData.normalizeDataEachModes(  rawData        = withRULData,
                                                                    headline       = self.preData.headline["allWithoutLabelNoBias"] + ["rul"],
                                                                    )                       # Normalize the data
        
        # ----------------------------
        # Get data for a specific case
        # ----------------------------
        caseData            = norData[environmentIdx]                       
        self.allSensor      = caseData[self.preData.headline["allWithoutLabelNoBias"]]
        self.originalSensors= originalSensor[environmentIdx][self.preData.headline["allWithoutLabelNoBias"]]
        self.timeTraj       = caseData["Time"]
        self.env            = caseData["environmentID"]
        self.trajectID      = caseData["Id"]


        aiLogger.debug(f"The shape of all sensors is {self.allSensor.shape, self.env.shape, self.trajectID.shape, self.timeTraj.shape}")

        self.getAtIdx                   = 1                     # Initilizing the idx only from 1 to 100
        self.totalTraject               = 100
        self.afterMaintenaceSubtraction = 10
        self.initPoint                  = 0

        
        #----------------------------- 
        # Create thread and others
        #-----------------------------
        self.input_input_signal2ai      = threading.Thread(target = self.__input_signal2ai, daemon=False)
        self.mutex                      = Lock()
        self.__running                  = False
        
        self.base_info                  = {}
        self.utility_info               = {}

        #----------------------------
        # Condition object
        #----------------------------
        # self.c_signal2_ai                     = None
        self.c_processed_signal2_ai             = threading.Condition()
        
        
        
        #----------------------------
        # Buffer
        #----------------------------
        buffConfig                                  = BufferConfig()
        self.processed_signal2_ai_buffer            = buffConfig.processedBuffer
        self.processed_trigger_maitenance_buffer    = buffConfig.rawBuffer
        
    #==========================================
    # This function is used to start the thread
    #------------------------------------------         
    def start(self):
        self.__running = True
        self.input_input_signal2ai.start()
        
        
    #==========================================
    # This function is used to stop the thread
    #------------------------------------------   
    def stop(self):
        self.__running = False 
        
    #==========================================
    # This function expands layers buffer to 
    # multiple buffers connected to Ai Features
    #------------------------------------------     
    def __input_signal2ai(self):
        while True:
            try:
                #-------------------------------
                # wait till get new frame 
                #-------------------------------
                # if self.c_raw_fps_control is not None:
                #     with self.c_raw_fps_control:
                #         self.c_raw_fps_control.wait()


                # -------------------------------
                # Getting data in each trajectory 
                # -------------------------------
                aiLogger.info(f"Getting the HI at the idx {self.getAtIdx}")
            
                seqXAllSensorTrain, \
                originalSensor, \
                seqXTrainEnv,       \
                seqTime                 = self.preData.getTensorData(   dataList    = [self.allSensor, self.originalSensors, self.env,self.timeTraj],
                                                                        trajectId   = self.trajectID,  
                                                                        shuffle     = False,      
                                                                        getAtIdx    = self.getAtIdx,
                                                                        seqLength   = self.preData.config["seqLength"],
                                                                        overlap     = self.preData.config["overLap"]         
                                                                        )

                seqTimeTrain        = seqTime[:,-1,:]
                curEnv              = seqXTrainEnv[:,-1,:]
                aiLogger.debug(f"The length of data is {len(seqXAllSensorTrain)}")

                # -------------------------------
                # Getting data in each specific
                # time from each trajectory
                # -------------------------------
                idxPoint        = self.initPoint    
                self.initPoint  = self.initPoint + self.afterMaintenaceSubtraction          # Moving the trajectory a little but after maintenance
                        
    
                while idxPoint < seqXAllSensorTrain.shape[0]:

                    # ------------------------------
                    # Change idx if new maintenance
                    # ------------------------------
                    if not self.processed_trigger_maitenance_buffer.empty():
                        maintenance_info = self.processed_trigger_maitenance_buffer.get()
                
                        if maintenance_info == 1:
                            aiLogger.warning("Maintenance is on going!!!!!!!!!!!!!!!!!!!!=================================================")
                            break
                        

                    curTime                         = time.time()
                    self.base_info["curTraject"]    = self.getAtIdx
                    self.base_info["normSignal"]    = seqXAllSensorTrain[idxPoint:idxPoint+1,:,:]
                    self.base_info["oriSignal"]     = originalSensor[idxPoint:idxPoint+1,:,:]
                    self.base_info["timestamp"]     = seqTimeTrain[idxPoint:idxPoint+1]
                    self.base_info["env"]           = curEnv[idxPoint:idxPoint+1]
                    self.base_info["frameIdx"]      = idxPoint
                    self.base_info["startFrameTime"]= curTime
                    self.base_info["simTime"]       = seqTimeTrain[idxPoint:idxPoint+1]

                    self.processed_signal2_ai_buffer.put((self.base_info.copy(), self.utility_info.copy()))

                
                    with self.c_processed_signal2_ai:
                        if (self.processed_signal2_ai_buffer.qsize() >= 1):
                            self.c_processed_signal2_ai.notify_all()   


                    time.sleep(0.2)
                    #-------------------------------
                    # get frame from full buffer and 
                    # notify
                    #-------------------------------
                    if self.processed_signal2_ai_buffer.full():
                        self.processed_signal2_ai_buffer.get()  
                        aiLogger.warning("The buffer is full, please check the size and time again to handle!!!")
                    

                    idxPoint += 1

                self.getAtIdx +=1 if self.getAtIdx <= self.totalTraject else 1
                

                    
                               
                        
                if not self.__running:
                    break  


            except Exception as e:
                aiLogger.exception("__input_signal2ai Exception" + str(e))
    
    
 
        


        
        
        
            

                
                
                
                    
                    
                    
                    
                
                
            
        
        
        
        
        