from pickle import TRUE
import queue
# from ai_core.utils   import get_GPU_available

#=========================================
# Name_class: BufferConfig
# Purpose   : Config the size of buffer
#=========================================    
buffer_size = 25

class BufferConfig():
    def __init__(self):
        self.buffer_size            = 25
        self.rawBuffer              = queue.Queue(maxsize=self.buffer_size) 
        self.processedBuffer        = queue.Queue(maxsize=self.buffer_size) 