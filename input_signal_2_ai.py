# =============================================
# IMPORT
# =============================================
import  threading
import  queue
import  time
from    typing          import List
from    utility.config  import BufferConfig
from    utility.logger  import aiLogger
from    threading       import Lock



from keras.callbacks        import EarlyStopping, ReduceLROnPlateau
from model.componentModel   import mainModel
from model.tepSystemModel   import SystemModel
from preProcessingDataTEP   import PreProcessData
from prettytable            import PrettyTable

from keras.optimizers       import Adam
from utility.matpilotDraw   import MatPlot
from sklearn.preprocessing  import MinMaxScaler

import tensorflow           as tf
import pandas               as pd
import matplotlib.pyplot    as plt
import numpy                as np
import os, warnings
import random
import json



# ---------------------------
# Loading config file
# ---------------------------
with open('utility/modelConfig.json', 'r') as configFile:
    configData = json.load(configFile)
config = configData["tep"]  

outputDir               = os.path.expanduser(config["dataDir"])
totalTraject            = 100
normalizeHI             = True
showRedundantShutdown   = False




class InputSignal2Ai():
    def __init__(self): 
        
        # ----------------------------
        # Handle the data
        # ----------------------------
        environmentIdx      = 4

        self.preData        = PreProcessData(config=config)
        rawData             = self.preData.loadDataFromH5Format(dir              = outputDir, 
                                                                name             = "data/teps.h5",
                                                                case             = "all", 
                                                                reduceSampleRate = True, 
                                                                removeHigherThr  = False)

        withRULData         = self.preData.addRULAndEnvironment(rawData)
        norData             = self.preData.normalizeDataEachModes(  rawData        = withRULData,
                                                                    headline       = self.preData.headline["allWithoutLabelNoBias"] + ["rul"],
                                                                    )    # Normalize the data
        caseData            = norData[environmentIdx]


        #-------------------------------
        # Get the data for the 
        #-------------------------------
        getAtIdx = 80
        print(f"Getting the HI at the idx {getAtIdx}")

        
        data                = norData[environmentIdx]
        allSensor           = data[self.preData.headline["allWithoutLabelNoBias"]]


        seqXAllSensorTrain,     \
        seqXRulTrain,           \
        seqXTrainEnv,seqTime                = self.preData.getTensorData(dataList    = [allSensor,rul,env,timeTraj],
                                                                        trajectId   = trajectID,  
                                                                        shuffle     = False,      
                                                                        getAtIdx    = getAtIdx,
                                                                        seqLength   = preData.config["seqLength"],
                                                                        overlap     = preData.config["overLap"]         
                                                                        )

        seqTimeTrain        = seqTime[:,-1,:]
        curEnv              = seqXTrainEnv[:,-1,:]
        
        

        #----------------------------- 
        # Create thread and others
        #-----------------------------
        self.input_input_signal2ai      = threading.Thread(target=self.__input_signal2ai, daemon=False)
        self.mutex                      = Lock()
        
        self.__running                  = False
        self.base_info                  = {}
        self.utility_info               = {}


        #----------------------------
        # Condition object
        #----------------------------
        self.c_raw_fps_control                  = None
        self.c_processed_input_signal_2ai       = threading.Condition()
        
        
        #----------------------------
        # Buffer
        #----------------------------
        buffConfig                              = BufferConfig()
        self.processed_input_signal_2ai_buffer  = buffConfig.processedBuffer
        
        
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
                if self.c_raw_fps_control is not None:
                    with self.c_raw_fps_control:
                        self.c_raw_fps_control.wait()


                


                
                #-------------------------------
                # get the frame from buffers
                #------------------------------- 
                for camera in self.cameras:
                    if not self.raw_fps_control_buffer[camera.id].empty():
                        self.begin_fps_time[camera.id] = time.time()
                        
                        input_data   = self.raw_fps_control_buffer[camera.id].get()
                        # print(input_data)
                        stream_frame = input_data["frame"]
                        uuid         = input_data["frame_id"]
                        ts           = input_data["ts"]
                        # print(stream_frame)
                        #-------------------------------
                        # get frames base on processRate
                        #------------------------------- 
                        
                        if self.control_fps[camera.id].get_frame_now():
                            #----------------------
                            # Compress info
                            #----------------------
                            self.base_info[camera.id]['batchsize']  = self.batch_size 
                            self.base_info[camera.id]['camera']     = camera
                            self.base_info[camera.id]['frame'].append(stream_frame)
                            self.base_info[camera.id]['frame_time'].append(ts)
                            self.base_info[camera.id]['uuid'].append(uuid)
                            
                            curLen = len(self.base_info[camera.id]['frame'])
                                
                            if curLen == self.batch_size:
                                
                                #-------------------------------
                                # Motion detection 
                                #-------------------------------
                                if self.runMotionDetection:
                                    motion = self.motion_detection[camera.id].batch_detect(self.base_info[camera.id]['frame'])
                                    self.utility_info[camera.id]['motion'] = motion
                                    # aiLogger.debug("motion detection {}".format(motion))
                                else:
                                    self.utility_info[camera.id]['motion'] = True
                                
                                # if camera.id == 'cam6':
                                #     print("motion detection",motion, camera.id)
                                
                                
                                self.processed_fps_control_buffer.put((self.base_info[camera.id].copy(),self.utility_info[camera.id].copy()))
                                self.base_info[camera.id]['frame']      = []
                                self.base_info[camera.id]['frame_time'] = []
                                self.base_info[camera.id]['uuid']       = []
                                # self.utility_info[camera.id]['motion']  = False
                                
                            
                                #-------------------------------
                                # get frame from full buffer and 
                                # notify
                                #-------------------------------
                                if self.processed_fps_control_buffer.full():
                                    self.processed_fps_control_buffer.get()  
                                    
                                with self.c_processed_fps_control:
                                    if (self.processed_fps_control_buffer.qsize() >= 1):
                                        self.c_processed_fps_control.notifyAll()  
    
                            #-------------------------------
                            # Get process time and frame count
                            # to calculate FPS of AI input
                            #-------------------------------
                            self.__fps_log.measure_input_frame2ai[camera.id]['time']    += time.time()- self.begin_fps_time[camera.id]
                            self.__fps_log.measure_input_frame2ai[camera.id]['fps']     +=  1
                            self.__fps_log.measure_input_frame2ai[camera.id]['nb_skip_frame'] = self.control_fps[camera.id].skip_frames
                    
                        #-------------------------------
                        # Get process time and frame count
                        # to calculate FPS from VMS input
                        #-------------------------------
                        self.__fps_log.measure_vms_fps[camera.id]['time']   += time.time()- self.begin_fps_time[camera.id]
                        self.__fps_log.measure_vms_fps[camera.id]['fps']    += 1
                            
                        
                    if not self.__running:
                        break   


            except Exception as e:
                aiLogger.exception("__input_frame2ai Exception" + str(e))
                # print("------- __input_frame2ai Exception " + str(e))    
    
 
        


        
        
        
            

                
                
                
                    
                    
                    
                    
                
                
            
        
        
        
        
        