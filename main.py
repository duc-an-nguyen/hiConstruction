from logging import raiseExceptions
import threading
import sys
# import packages
import os
import time
from multiprocessing import Process
from tepEdgeDevice import TEPEdgeDevice


#==========================================
# This function is used to run the Ai in 
# each process independently
#------------------------------------------
def run_system():

    #----------------------------------
    # Init the ai system
    #----------------------------------
    aiProcessor        = TEPEdgeDevice("./data","./ai_core/model/",isTensorrt=True)
    
    if aiProcessor.gpu_is_available:
        aiProcessor.start()

    else:
        raiseExceptions("GPU is not available")


    
#==========================================
#           MAIN FUCNTION 
#------------------------------------------
if __name__ == '__main__':

    process     = Process(target=run_system)
    process.start()
    process.join() 
  
