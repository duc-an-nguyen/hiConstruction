import pandas as pd
import numpy as np
import random
# import torch
# from sklearn.linear_model import LinearRegression
from utility.logger import aiLogger

class PreProcessData(object):
    def __init__(self,  seqLength   = 128,
                        overLap     = 127,
                        theCase     = 2,
                ):
        # ------------------------------
        # Configure some variables
        # ------------------------------
        self.seqLength      = seqLength
        self.overlap        = overLap
        self.stabletime     = 30        # hours 
        self.sampleRateTime = 3         # minuites per one sample

        self.machine            = ["reactor","separator","stripper"]
        self.rawData            = {"reactor":[],"separator":[],"stripper":[],"label":[]}
        
        # self.reactorHeadline    = [ "Reactor Coolant",       "Reactor Feed Rate",       "Reactor Pressure", \
        #                             "Reactor Temperature",   "Reactor Coolant Temp",    "Recycle Flow",
        #                             'Component A to Reactor','Component B to Reactor',  'Component C to Reactor',
        #                             'Component D to Reactor','Component E to Reactor',  'Component F to Reactor']
        
        self.allHeadline     = ["D feed",                    "E Feed",                   "A Feed",                   "A and C Feed", 
                                "Recycle",                   "Purge",                    "Separator",                "Stripper",
                                "Steam",                     "Reactor Coolant",          "Condenser Coolant",        "Agitator",
                                "msv A Feed",                "msv D Feed",               "msv E Feed",               "msv A and C Feed",
                                "Recycle Flow",              "Reactor Feed Rate",        "Reactor Pressure",         
                                "Reactor Temperature",       "Purge Rate",               "Product Sep Temp",         
                                "Product Sep Pressure",      "Product Sep Underflow",    "Stripper Pressure",
                                "Stripper Underflow",        "Stripper Temp",            "Stripper Steam Flow",      "Compressor Work",
                                "Reactor Coolant Temp",      "Separator Coolant Temp",   "Component A to Reactor",   "Component B to Reactor",
                                "Component C to Reactor",    "Component D to Reactor",   "Component E to Reactor",   "Component F to Reactor",
                                "Component A to Purge",      "Component B to Purge",     "Component C to Purge",     "Component D to Purge",
                                "Component E to Purge",      "Component F to Purge",     "Component G to Purge",     "Component H to Purge",
                                "Component D to Product",    "Component E to Product",   "Component F to Product",   "Component G to Product",
                                "Component H to Product",    "Liquid Input Stripper",    "Liquid Input Separator",   "Liquid Input Reactor",
                                'Reactor Level','Product Sep Level','Stripper Level'
                                ]
        
        # self.degraLiq         = ['Liquid Input Stripper','Liquid Input Separator','Liquid Input Reactor']
        self.degraLiq         = ['Reactor Level','Product Sep Level','Stripper Level']
        
        # --------------------------
        # Case 2
        # --------------------------
        if theCase == 2:
            self.reactorHeadline    = [ "Reactor Coolant",              "Reactor Pressure",         "Purge",
                                       "Reactor Feed Rate",             "Reactor Coolant Temp"]
                                                                        
            self.separatorHeadline  = [ "Separator",                    "Separator Coolant Temp",    "Product Sep Temp",     
                                       "Product Sep Pressure",          "Product Sep Underflow",]
                                       
                                    
                                                                      
            self.stripperHeadline   = [ "Stripper",                     "Stripper Pressure",    "Stripper Underflow",    
                                       "Stripper Temp",                 "Stripper Steam Flow"]                           
                                        


        else:
            self.reactorHeadline    = [ "Reactor Coolant",              "Reactor Pressure",         "Purge",
                                       "Reactor Feed Rate",             "Reactor Coolant Temp"]
                                                                        
            self.separatorHeadline  = [ "Separator",                    "Separator Coolant Temp",    "Product Sep Temp",     
                                       "Product Sep Pressure",          "Product Sep Underflow",]
                                       
                                    
                                                                      
            self.stripperHeadline   = [ "Stripper",                     "Stripper Pressure",    "Stripper Underflow",    
                                       "Stripper Temp",                 "Stripper Steam Flow"]

        aiLogger.info("Loading data from files ...")
        self.case          = {}

            
        self.case[1]       = pd.read_csv('data/case1.csv')
        self.case[2]       = pd.read_csv('data/case2.csv')
        self.case[3]       = pd.read_csv('data/case3.csv')
        self.case[4]       = pd.read_csv('data/case4.csv')
        self.case[5]       = pd.read_csv('data/case5_1.csv')
        self.case[6]       = pd.read_csv('data/case6.csv')
        self.case[7]       = pd.read_csv('data/thr/reactor_up_mode4_high.csv')
        self.case[8]       = pd.read_csv('data/thr/seperator_up_mode4_high.csv')
        self.case[9]       = pd.read_csv('data/thr/stripper_up_mode4_high.csv')

    # ===================================
    # This function is used to 
    # extract data from .csv file
    # -----------------------------------
    def getDataset(self,
                    component       = "all",
                    removeInitState = True
                    ):
        aiLogger.info("Converting data to list of array ...")

        allData         = []
        allLabel        = []
        allTime         = []
        allDegra        = []
        dataComponent   = {}

        # if theCase == "all":
        simulationCase  = [self.case[idxCase+1] for idxCase in range(9)]
        # else:
            # simulationCase  = [self.case[theCase]]


        for idxCase,sim in enumerate(simulationCase):
            # -------------------------
            # Extract by using panda from healine
            # and add ID to mornitor later
            # -------------------------
            idxSimColumn    = sim['Id'].to_numpy()                      # Get the id column from the data
            addIdxCase      = np.full((idxSimColumn.size,1),idxCase+1)  # Create the id to get the working condition

            dataComponent["Reactor"]        = sim[self.reactorHeadline].to_numpy()
            dataComponent["Separator"]      = sim[self.separatorHeadline].to_numpy()
            dataComponent["Stripper"]       = sim[self.stripperHeadline].to_numpy()

            degration                       = sim[self.degraLiq].to_numpy()
            testAllData                     = sim[self.allHeadline].to_numpy()

            

            # withIdReactor   = np.concatenate((addIdxCase,temReactor), axis=1)       #Add case id  to the first column
            # withIdSeparator = np.concatenate((addIdxCase,temSeparator), axis=1)     #Add case id to the first column
            # withIdStripper  = np.concatenate((addIdxCase,temStripper), axis=1)      #Add case id to the first column
            # rawData         = np.concatenate((temReactor,temSeparator,temStripper), axis=1)
             
            if component == "all":
                rawData     = np.concatenate((dataComponent["Reactor"],dataComponent["Separator"],dataComponent["Stripper"]), axis=1)

                

            elif component == "testAll":
                rawData     = testAllData

            else:
                rawData     = dataComponent[component]

            # -------------------------
            # Create the RUL label
            # -------------------------
            oneDirRulLabel  = np.array([])                              # Contain the label of RUL data
            timeStamp       = sim['Time'].to_numpy()  
            for idxRul in range(max(idxSimColumn)):  
                curId           = np.where(idxSimColumn == (idxRul+1))           
                curRul          = np.squeeze(timeStamp[curId[0][-1]] - timeStamp[curId])
                oneDirRulLabel  = np.concatenate((oneDirRulLabel,curRul), axis=0)
              
            oneDirRulLabel = oneDirRulLabel.reshape((-1, 1))
            timeStamp      = timeStamp.reshape((-1, 1))

            # print(timeStamp.shape)
           
            # -------------------------
            # Convert to a matrix, and transpose 
            # to have a shape of 8xlen
            # -------------------------
            rulLabel        = [np.squeeze(oneDirRulLabel[np.where(idxSimColumn == (idxSimExtract+1)),:] ,axis =0)   for idxSimExtract in range(max(idxSimColumn))]
            collectedData   = [np.squeeze(rawData[np.where(idxSimColumn == (idxSimExtract+1)),:] ,axis =0)   for idxSimExtract in range(max(idxSimColumn))]
            time            = [np.squeeze(timeStamp[np.where(idxSimColumn == (idxSimExtract+1)),:] ,axis =0)   for idxSimExtract in range(max(idxSimColumn))]
            degra           = [np.squeeze(degration[np.where(idxSimColumn == (idxSimExtract+1)),:] ,axis =0)   for idxSimExtract in range(max(idxSimColumn))]

            if removeInitState:
                rulLabel        = self.removeInitState(rulLabel,removeValue=True,getLast = True)    
                collectedData   = self.removeInitState(collectedData)
                time            = self.removeInitState(time,removeValue=True)
                degra           = self.removeInitState(degra)
            # -------------------------
            # This function is used to 
            # store all data
            # -------------------------
            allData     = allData   + collectedData
            allLabel    = allLabel  + rulLabel
            allTime     = allTime   + time
            allDegra    = allDegra  + degra
           
        return allData,allLabel,allTime,allDegra
    


    def removeInitState(self,data,removeValue = False, getLast = False):
        initPoint       = int(self.stabletime*60/self.sampleRateTime)
        # print(data[0].shape)

        # ----------------------------
        # Some data need to get data conversely
        # ----------------------------
        if getLast:
            removedInitData = [curData[:-initPoint,:] for curData in data]
        else:
            removedInitData = [curData[initPoint:,:] for curData in data]
        # print(removedInitData[0].shape)

        # ----------------------------
        # If the data is time, we need
        # to subtract with the first 30h (stable time)
        # ----------------------------
        if removeValue:
            removedInitData = [curData - self.stabletime for curData in removedInitData]

        
        return removedInitData
    

    # ===================================
    # This function is used to clean data
    # -----------------------------------
    def cleanData(self, rawSignal,
                        getMovingAverage = False,
                        flipToDown       = False):
        
        aiLogger.info("Cleaning data ...")
        windowSize  = 127
        padWidth    = int(windowSize/2)  # Equivalent to the floor(window_size/2) for window_size = 3
        

        cleanSignal = []
        window      = np.ones(int(windowSize))/float(windowSize)
          
        for idxSig in range(len(rawSignal)):
            trajec      = rawSignal[idxSig]                # Get each trajectory
            outputRow   = trajec.shape[0] - windowSize + 1
            # newSig      = np.zeros((outputRow,trajec.shape[1]))

            # Compute the moving average using convolution
            for idxArray in range(trajec.shape[1]):
                # ---------------------------
                if  getMovingAverage:
                    padSig      = np.pad(trajec[:,idxArray], padWidth, mode='symmetric') 
                    newSig      = np.convolve(padSig, window, 'valid')
                else:
                    newSig      = trajec[:,idxArray]
                # ---------------------------
                if flipToDown:
                    trajec[:,idxArray]   = self.flip2Down(newSig)
                else:
                    trajec[:,idxArray]   = newSig


            cleanSignal.append(trajec) 

        return cleanSignal

    def flip2Down(self,data):
        aiLogger.info("Flipping data ...")
        time_points     = np.arange(len(data))
        coefficients    = np.polyfit(time_points, data, 2)
        # polynomial      = np.poly1d(coefficients)

        # fitted_curve    = polynomial(time_points)
        if coefficients[0] < 0:
            adjusted_time_series = data 
            # print("The time series was trending downwards and has been flipped.")
        else:
            adjusted_time_series = data * -1
            # print("No adjustment needed.")
        
        return adjusted_time_series

    # ===================================
    # This function is used to normalize 
    # the data
    # -----------------------------------
    def normalizeData(self,rawData):
        aiLogger.info("Normalizing the data ...")

        minAllTrajec    = []
        maxAllTrajec    = []
        # -----------------------
        # Get the lowest element
        # from each list (trajectory)
        for idxList in range(len(rawData)):     
            trajec        = rawData[idxList]                # Get each trajectory
            minEachTrajec = trajec.min(axis=0)              # Get min each trajectory
            maxEachTrajec = trajec.max(axis=0)              # Get max each trajectory
            minAllTrajec.append(minEachTrajec.tolist())     # Store all min value 
            maxAllTrajec.append(maxEachTrajec.tolist())     

        # -----------------------
        # Get min and max value 
        # from all trajectories
        # -----------------------
        minEachFeature = np.min(np.array(minAllTrajec),axis = 0)    
        maxEachFeature = np.max(np.array(maxAllTrajec),axis = 0)
        
        # print(minEachFeature.shape)
        # print(maxEachFeature.shape)
        # print(maxEachFeature[1,-1] - minEachFeature(1,-1))
        # for eachTraject,idx in enumerate(rawData):
            
        #     if (maxEachFeature[idx] - minEachFeature[idx]) != 0:
        #         norData = (eachTraject - minEachFeature[idx]) / (maxEachFeature[idx] - minEachFeature[idx])
            
        #     else:
        #         norData = eachTraject

        norData = [(eachTraject - minEachFeature) / (maxEachFeature - minEachFeature) for eachTraject in rawData]

        # print(norData[0].shape)
            
        return norData
    
    def normalLabel(self,allDegra):
        aiLogger.info("Normalizing the label ...")
        minReac,maxReac     = (65,114)
        minSep, maxSep      = (50,136)
        minStrip,maxStrip   = (50,130)

        for idxList in range(len(allDegra)):   
            allDegra[idxList][:,0] = 1 - (allDegra[idxList][:,0] - minReac)/(maxReac - minReac)
            allDegra[idxList][:,1] = 1 - (allDegra[idxList][:,1] - minSep)/(maxSep - minSep)
            allDegra[idxList][:,2] = 1 - (allDegra[idxList][:,2] - minStrip)/(maxStrip - minStrip)
        
        # norReac     = [(eachDegre[:,0] - minReac)/(maxReac - minReac)       for eachDegre in allDegra]
        # norSep      = [(eachDegre[:,1] - minSep)/(maxSep - minSep)          for eachDegre in allDegra]
        # norStrip    = [(eachDegre[:,2] - minStrip)/(maxStrip - minStrip)    for eachDegre in allDegra]
        
        return allDegra
    
    def getTheCase(self,theCase,data,time):
        idxStart = (theCase-1)*100
        idxEnd   = idxStart + 100
        caseData = data[idxStart:idxEnd]
        caseTime = time[idxStart:idxEnd]

        return caseData,caseTime

    

    # ===================================
    # This function is used to split data 
    # into sequences.
    # The input is a list of arrays with 
    # different length
    # reactor: 6sensors
    # sep: 7 sensors
    # strip: 5 sensors
    # -----------------------------------
    def splitAllDataIntoSequences(self,dataXY):
        aiLogger.info("Spliting data into sequences now ...")

        data    = dataXY[0]
        label   = dataXY[1]
        
        allSeq   = []
        allLabel = []
         
        for idx in range(len(data)):
            transposedDt    = data[idx].transpose()
            transposedLb    = label[idx].transpose()
            # print(transposedLb)
            seqData,seqLabel    = self.splitAndPadSequences(transposedDt,
                                                           transposedLb,
                                                            self.seqLength,
                                                            self.overlap)     
            
            # print(len(seqData))
            allSeq          = allSeq    + seqData
            allLabel        = allLabel  + seqLabel

    
        tensorX =   np.stack(allSeq, axis=0)         
        tensorY =   np.stack(allLabel, axis=0)     

        tensorX   = tensorX.transpose(0, 2, 1)        # resize to num_seq, lengh_seq, and num features
             
        return tensorX,tensorY
    
    # ===================================
    # This function is used to split data 
    # into sequences.
    # -----------------------------------
    def splitAndPadSequences(self,dataX,dataY, seqLength, overlap):
        sequences       = []
        labels          = []
        step_size       = seqLength - overlap
        # print(dataX)
        totalLength     = dataX.shape[1]
        # print(dataY)

        for start in range(0, totalLength-overlap, step_size):
            end = start + seqLength
            # ----------------------
            # If the sequence would 
            # go beyond the data length, 
            # pad it
            # ----------------------
            if end > totalLength:                
                padding  = end - totalLength
                # sequence = torch.nn.functional.pad(dataX[:, start:], (0, padding), 'constant', 0)
                curLalel = dataY[:,-1]
            else:
                sequence = dataX[:,start:end]           # Get sentense
                curLalel = dataY[:,end-1]               # Get the last element in the data

            sequences.append(sequence)
            labels.append(curLalel)
        
        return sequences,labels
    

    # ===================================
    # This function is used to split data
    # to train, val, test groups
    # -----------------------------------
    def splitTrainVal(self,norData,label):

        trainData = {"reactor":[],"separator":[],"stripper":[],"label":[]}
        valData   = {"reactor":[],"separator":[],"stripper":[],"label":[]}

        trainData   =   norData[0:40]       + norData[70:140]  + \
                        norData[170:240]    + norData[270:340] + \
                        norData[370:440]    + norData[470:540]  + \
                        norData[570:600]   
        
        valData     =   norData[40:70]     + norData[140:170] + \
                        norData[240:270]    + norData[340:370] + \
                        norData[440:470]    + norData[540:570]
        
        trainLabel   =  label[0:40]       + label[70:140]  + \
                        label[170:240]    + label[270:340] + \
                        label[370:440]    + label[470:540]  + \
                        label[570:600]   
        valLabel     =  label[40:70]     + label[140:170] + \
                        label[240:270]    + label[340:370] + \
                        label[440:470]    + label[540:570]
        
        return [trainData,trainLabel],[valData,valLabel]
    

    
    
    