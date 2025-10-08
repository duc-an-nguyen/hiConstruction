#=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=--=
#                       PLEASE DECLARE YOUR LIBRARY BELOW
#=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=--=
from utility.logger import aiLogger
import time
import os
import json
# import yaml

# --------------------------
# IMPORT BASE SYSTEM
# --------------------------
from input_signal2_ai                     import InputSignal2Ai
from ai_handling                          import AiHandling


#=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=--=

class TepEdgeDevice:
    def __init__(self):
        
        #=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        #           DECLARE SOME VARIABLES FOR AI 
        #
        #=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

        with open('utility/modelConfig.json', 'r') as configFile:
            configData = json.load(configFile)
        config      = configData["tep"]

        #----------------------------------
        # Init general variable
        #----------------------------------
        # self.__batch_size                   = config.batch_size
        # self.__gpu_for_model                = config.gpu_for_model
        # gpu_idx,self.is_server              = get_GPU_available()
        
        # self.gpu_is_available               = True if gpu_idx >=0 else False
        # device                              = "cuda:" + str(gpu_idx) if self.gpu_is_available  else "cpu"
        # self.__device                       = torch.device(device)
        # self.__tensorrt                     = isTensorrt


        #=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        #           DECLARE SOME CLASSES FOR AI 
        #
        #=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                            
        #----------------------------------
        # Initialize InputSignal2Ai
        # and AiHandling
        #----------------------------------
        self.__inputSignal2AiThread              = InputSignal2Ai(config)        
        self.__aiThread                          = AiHandling(config)
                                                                                                       
        #----------------------------------
        # Connect all block in the system
        # Notice: Each block is isolated and 
        # need this one to connect
        #----------------------------------
        self.__all_connection(  self.__inputSignal2AiThread,
                                self.__aiThread,
                                )  

        # aiLogger.info("The ai model at the process {} was loaded successfully!!!".format(os.getpid()))   
    #==========================================
    # This function used to connect 4 blocks 
    # consisting of InputSignal2Ai,
    # AiHandling
    #------------------------------------------
    def __all_connection(   self,                            
                            inputFrame2AiThread,
                            aiThread,
                            ):
        #--------------------------------
        # Connect InputSignal2Ai and 
        # AiHandling 
        #--------------------------------
        aiThread.c_raw_ai_handling          = inputFrame2AiThread.c_processed_signal2_ai
        aiThread.raw_ai_handling_buffer     = inputFrame2AiThread.processed_signal2_ai_buffer
        
        #--------------------------------
        # Connect BackboneThread and 
        # ExpandBackbone2Features 
        # #--------------------------------
        # backbone2FeaturesThread.raw_rl_bottleneck_buffer     = backboneThread.processed_backbone_buffer 
        # backbone2FeaturesThread.c_raw_rl_bottleneck          = backboneThread.c_processed_backbone
        

                 
                    
    #==========================================
    # This function is used to start the Ai 
    # system
    #------------------------------------------                
    def start(self,):
        try:            
            #----------------------------------
            # Start __inputSignal2AiThread,  
            # aiThread
            # and fps_log
            #----------------------------------
            self.__inputSignal2AiThread.start()
            self.__aiThread.start()  
            
            aiLogger.warning("Ai core was started successfully!!!")
            
        except Exception as e:
            aiLogger.exception("------- ai_processor start() attribute Exception " + str(e))  

    #==========================================
    # This function is used to stop the Ai 
    # system
    #------------------------------------------
    # def stop(self,):
    #     aiLogger.warning("The Ai core was stopped!!!!")
        
        
        
    #==========================================
    # This function is used to release the Ai 
    # system
    #------------------------------------------
    # def release(self,):
    #     try:
    #         aiLogger.warning("The Ai core will be released soon !!! Please wait...")

    #         #----------------------------------
    #         # Stop backboneThread,  
    #         # backbone2FeaturesThread
    #         # and __inputFrame2AiThread
    #         #----------------------------------
    #         self.__inputFrame2AiThread.stop()
    #         time.sleep(1.5)                       # Wait to other threads complete their processing
    #         self.__backboneThread.stop() 
    #         self.__backbone2FeaturesThread.stop()

    #         #----------------------------------
    #         # Stop Yolo5
    #         #----------------------------------
    #         if 'yolo5' in self.__available_features:
    #             self.__yolo5NeckHeadThread.stop()
    #             self.__licensePlateThread.stop()
    #             self.__expandYoloNeckHead2ApplicationThread.stop()
    #             self.__yoloApp2OutThread.stop()
                
    #             for camera in self.__inputFrame2AiThread.cameras:
    #                 self.__tracking[camera.id].stop() 
    #                 self.__statistic[camera.id].stop()

    #         #----------------------------------
    #         # Stop retinaface
    #         #----------------------------------
    #         if 'retinaface' in self.__available_features:
    #             self.__retinaFaceNeckHeadThread.stop()

    #         #----------------------------------
    #         # Stop yolo_phuc
    #         #----------------------------------
    #         if 'deeplabv3p' in self.__available_features:
    #             self.__deeplabV3pNeckHeadThread.stop()
                
    #         #----------------------------------
    #         # Stop without_ai thread
    #         #----------------------------------
    #         if 'without_ai' in self.__available_features:
    #             self.__withoutAiNeckHeadThread.stop()
                
            
    #         if 'fire_classification' in self.__available_features:
    #             self.__fireClassificationNeckHeadThread.stop()

    #         #----------------------------------
    #         # Stop this one if display option
    #         # was turned on
    #         #----------------------------------
    #         self.__outputAiInfoSync.stop()
                
    #         #----------------------------------
    #         # Stop timer to measure threads, and
    #         # notify all threads that still in the
    #         # wait command
    #         #----------------------------------
    #         self.__fps_log.calculation_thread.cancel()
    #         self.__notify_to_stop()                 
    #         time.sleep(1.5)
            
    #         #----------------------------------
    #         # Check alive thread
    #         #----------------------------------
    #         # print(self.__backboneThread._ai_thread.is_alive())
    #         # print(self.__backbone2FeaturesThread.release_bottleneck.is_alive())
    #         # print(self.__yolo5NeckHeadThread._ai_feature_thread.is_alive())
    #         # print(self.__expandYoloNeckHead2ApplicationThread.feature2application.is_alive())
    #         # print(self.__yoloApp2OutThread.application2output.is_alive())
    #         # print(self.__retinaFaceNeckHeadThread._ai_feature_thread.is_alive())
    #         # print(self.__withoutAiNeckHeadThread._ai_feature_thread.is_alive())
    #         # print(self.__fireClassificationNeckHeadThread._ai_feature_thread.is_alive())
    #         # print(self.__outputAiInfoSync.ai_information.is_alive())
    #         # for camera in self.__inputFrame2AiThread.cameras:
    #         #     print("tracking:",self.__tracking[camera.id]._ai_tracking_thread.is_alive())
    #         #     print(self.__statistic[camera.id]._ai_application_thread.is_alive())
    #         #----------------------------------
    #         # Delete thread
    #         #----------------------------------
    #         del self.__inputFrame2AiThread
    #         del self.__backboneThread
    #         del self.__yolo5NeckHeadThread
    #         del self.__licensePlateThread
    #         del self.__expandYoloNeckHead2ApplicationThread
    #         del self.__yoloApp2OutThread
    #         del self.__retinaFaceNeckHeadThread
    #         del self.__deeplabV3pNeckHeadThread
    #         del self.__withoutAiNeckHeadThread
    #         del self.__fireClassificationNeckHeadThread
    #         del self.__outputAiInfoSync
    #         del self.__tracking
    #         del self.__statistic
    #         # del self.__fps_log
        
    #         aiLogger.warning("The Ai core at the process {} was released successfully!!!".format(os.getpid()))
    #     except Exception as e:
    #         aiLogger.exception("------- ai_processor release() attribute Exception " + str(e))    

    #==========================================
    # This function is used to notify to break 
    # the while loop before stop completely
    #------------------------------------------
    # def __notify_to_stop(self,):
    #     #----------------------------------
    #     # InputFrame2Ai
    #     #----------------------------------
    #     if self.__inputFrame2AiThread.input_input_frame2ai.is_alive():
    #         with self.__inputFrame2AiThread.c_raw_fps_control:
    #             self.__inputFrame2AiThread.c_raw_fps_control.notifyAll()
                
    #     #----------------------------------
    #     # BackboneThread
    #     #----------------------------------
    #     if self.__backboneThread._ai_thread.is_alive():
    #         with self.__backboneThread.c_raw_backbone:
    #             self.__backboneThread.c_raw_backbone.notifyAll()
                
    #     #----------------------------------
    #     # ExpandBackbone2Features
    #     #----------------------------------
    #     if self.__backbone2FeaturesThread.release_bottleneck.is_alive():
    #         with self.__backbone2FeaturesThread.c_raw_rl_bottleneck:
    #             self.__backbone2FeaturesThread.c_raw_rl_bottleneck.notifyAll()
            
            
    #     #----------------------------------
    #     # YOLO
    #     #----------------------------------        
    #     if 'yolo5' in self.__available_features:
    #         #----------------------------------
    #         # NeckHeadYolo5Thread
    #         #----------------------------------
    #         if self.__yolo5NeckHeadThread._ai_feature_thread.is_alive():
    #             with self.__yolo5NeckHeadThread.c_raw_neckhead:
    #                 self.__yolo5NeckHeadThread.c_raw_neckhead.notifyAll()

    #         #----------------------------------
    #         # LicensePlateThread
    #         #----------------------------------      
    #         if self.__licensePlateThread._ai_application_thread.is_alive():
    #             with self.__licensePlateThread.rawCondition:
    #                 self.__licensePlateThread.rawCondition.notifyAll()

    #         #----------------------------------
    #         # ExpandFeature2Application
    #         #----------------------------------  
    #         if self.__expandYoloNeckHead2ApplicationThread.feature2application.is_alive():
    #             with self.__expandYoloNeckHead2ApplicationThread.c_raw_feature2application:
    #                 self.__expandYoloNeckHead2ApplicationThread.c_raw_feature2application.notifyAll()
                    
    #         #----------------------------------
    #         # MergeYoloApp2Output
    #         #----------------------------------  
    #         if self.__yoloApp2OutThread.application2output.is_alive():
    #             with self.__yoloApp2OutThread.rawCondition:
    #                 self.__yoloApp2OutThread.rawCondition.notifyAll()
                    
                    
    #         #----------------------------------
    #         # TrackingThread and StatisticThread
    #         #----------------------------------
    #         for camera in self.__inputFrame2AiThread.cameras:
    #             if self.__tracking[camera.id]._ai_tracking_thread.is_alive():
    #                 with self.__tracking[camera.id].rawCondition:
    #                     self.__tracking[camera.id].rawCondition.notifyAll()
                        
    #             if self.__statistic[camera.id]._ai_application_thread.is_alive():
    #                 with self.__statistic[camera.id].rawCondition:
    #                     self.__statistic[camera.id].rawCondition.notifyAll()
                        
    #     #----------------------------------
    #     # RETINAFACE
    #     #---------------------------------- 
    #     if 'retinaface' in self.__available_features: 
    #         #----------------------------------
    #         # NeckHeadRetinaFaceThread
    #         #----------------------------------  
    #         if self.__retinaFaceNeckHeadThread._ai_feature_thread.is_alive():
    #             with self.__retinaFaceNeckHeadThread.c_raw_neckhead:
    #                 self.__retinaFaceNeckHeadThread.c_raw_neckhead.notifyAll()
        
    #     #----------------------------------
    #     # WITHOUT AI
    #     #----------------------------------        
    #     if 'without_ai' in self.__available_features:  
    #         #----------------------------------
    #         # NeckHeadWithoutAiThread
    #         #----------------------------------  
    #         if self.__withoutAiNeckHeadThread._ai_feature_thread.is_alive():
    #             with self.__withoutAiNeckHeadThread.c_raw_neckhead:
    #                 self.__withoutAiNeckHeadThread.c_raw_neckhead.notifyAll()
            
    #     #----------------------------------
    #     # FIRE CLASSIFICATION
    #     #----------------------------------         
    #     if 'fire_classification' in self.__available_features:  
    #         #----------------------------------
    #         # NeckHeadFireClassificationThread
    #         #----------------------------------  
    #         if self.__fireClassificationNeckHeadThread._ai_feature_thread.is_alive():
    #             with self.__fireClassificationNeckHeadThread.c_raw_neckhead:
    #                 self.__fireClassificationNeckHeadThread.c_raw_neckhead.notifyAll()
        
    #     #----------------------------------
    #     # OutputAiInfoSync
    #     #----------------------------------
    #     if self.__outputAiInfoSync.ai_information.is_alive():
    #         with self.__outputAiInfoSync.c_raw_ai_info_output:
    #             self.__outputAiInfoSync.c_raw_ai_info_output.notifyAll()
                
        
                
            
            





                
    
    



    

