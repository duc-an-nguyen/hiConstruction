import os
class tepsConfig():
    def __init__(self):
        
        # =-=-=-=-=-=-=-=-=-=-=-
        # FOR TRAINING
        self.visualize          = False
        self.cuda_visible_device = '0'
        self.n_epochs           = 2000
        
        self.batch_size         = 256
        self.val_batch_size     = 256
        self.num_workers        = 1
        self.momentumValues     = [0.8, 0.9, 0.95, 0.99]
        self.weightDecayValues  = [1e-4, 1e-3, 1e-2]    



        # =-=-=-=-=-=-=-=-=-=-=-
        # FOR MODEL

        self.numComponent   = 3
        
        # ---------------------
        # Reactor parameter
        # ---------------------
        self.reacInput          = 12
        self.reacHiddenSize     = 8
        self.reacNumLayer       = 1
        self.reacDropout        = 0

        # ---------------------
        # Separator parameter
        # ---------------------
        self.sepInput           = 17
        self.sepHiddenSize      = 4
        self.sepNumLayer        = 1
        self.sepDropout         = 0

        # ---------------------
        # Stripper parameter
        # ---------------------
        self.stripInput         = 10
        self.stripHiddenSize    = 4
        self.stripNumLayer      = 1
        self.stripDropout       = 0

        # ---------------------
        # latent space and attention
        # ---------------------
        self.latentSize         = 1
        self.attentionHead      = 1

        self.useComponentAsFeature = True


class vaeLSTMConfig():
    def __init__(self):
        self.modelName          = "VAELSTM"
        self.visualize          = False
        self.cuda_visible_device = '0'
        self.n_epochs           = 2000
        
        self.batch_size         = 256
        self.val_batch_size     = 256
        self.num_workers        = 1
        self.momentumValues     = [0.8, 0.9, 0.95, 0.99]
        self.weightDecayValues  = [1e-4, 1e-3, 1e-2]


        self.componentInput  = {"Reactor":12,"Separator":17,"Stripper":10}
        self.component  = "Stripper"
        self.overLap    = 255
        self.seqLength  = 256
        self.hiddenSize = 32
        self.latentSize = 4
        self.numLayer   = 2
        self.dropout    = 0.5

        self.last_ckpt          = 'checkpoints/last_'   + self.component + '_' + self.modelName + '.pth'
        self.best_ckpt          = 'checkpoints/best'    + self.component + '_' + self.modelName + \
                                    str(self.hiddenSize) + str(self.latentSize) + str(self.numLayer) \
                                    + str(self.dropout) + '.pth'

class vaeCNNConfig():
    def __init__(self):
        
        self.component          = "Stripper"


        self.modelName          = "VAECNN"
        self.vae_type           = 'vae_conv'           # vae_dense, vae_conv, timeVAE
        

        self.visualize          = False
        self.cuda_visible_device = '0'
        self.n_epochs           = 5000
        
        self.batch_size         = 64
        self.num_workers        = 1

        
        self.latentDim          = 1
        self.hiddenSize         = [200, 100, 50] # [50,100,200]      #[80, 200, 250] 

        self.theCase            = 2
        self.componentInput  = {"Reactor":5,"Separator":5,"Stripper":5}
        
        self.overLap    = 127
        self.seqLength  = 128

        self.modelDir   = "checkpoint/"+ self.vae_type + "_" + str(self.hiddenSize) + "_" + "latent" + "_" + str(self.latentDim) + "case" +str(self.theCase)

        # self.modelDir   = "checkpoint/TimeVAEUnNoise80_200_250_latent1"
        if not os.path.exists(self.modelDir + "/"):
            os.makedirs(self.modelDir + "/")


class rulVAECNNConfig():
    def __init__(self):
        self.theCase          = 2
        # -------------------
        # General info
        # -------------------
        self.modelName          = "rul"
        self.visualize          = False
        self.cuda_visible_device = '0'
        self.n_epochs           = 5000
        
        self.batch_size         = 64
        self.num_workers        = 1
        

        # -------------------
        # Model config
        # ------------------- 
        self.latentDim          = 1
        self.hiddenSize         = [200, 100, 50] # [50,100,200]      #[80, 200, 250] 

        self.componentInput     = {"Reactor":7,"Separator":5,"Stripper":5}
        self.reacInput          = 7
        self.sepInput           = 5
        self.stripInput         = 5
        
        self.overLap    = 127
        self.seqLength  = 128

        self.modelDir   = "checkpointNew/" + str(self.hiddenSize) + "_" + "latent" + "_" + str(self.latentDim) + "case" +str(self.theCase)

        # self.modelDir   = "checkpoint/TimeVAEUnNoise80_200_250_latent1"
        if not os.path.exists(self.modelDir + "/"):
            os.makedirs(self.modelDir + "/")


        # --------------------
        # 
        # --------------------
        self.num_heads  = 3
        self.savedDir   = "tep"
        self.useComponentAsFeature  = False
        self.numComponent   = 3

        