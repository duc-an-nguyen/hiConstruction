import pandas       as pd
import numpy        as np
import tensorflow   as tf
import random
import json
# import package
import os
# import torch
# from sklearn.linear_model import LinearRegression
from utility.logger         import aiLogger
from itertools      import accumulate


class PreProcessData(object):
    def __init__(self,  config      = None
                ):
        np.random.seed(1)
        random.seed(1)
        
        # ------------------------------
        # Configure some variables
        # ------------------------------
        self.config         = config
        self.seqLength      = self.config["seqLength"]
        self.overlap        = self.config["overLap"]
        self.stabletime     = 30        # hours 
        self.sampleRateTime = 6         # minuites per one sample



        self.reactorHeadline    = []
        self.separatorHeadline  = []
        self.stripperHeadline   = []
     
        self.machine            = ["reactor","separator","stripper"]
        self.rawData            = {"reactor":[],"separator":[],"stripper":[],"label":[]}
        self.numOfCases         = self.config["numOfCases"]
        self.trajectoryPerCase  = self.config["trajectoryPerCase"]

        self.scenario           = {}
        self.scenario[1]        = "mode 6: reactor     -> separator"
        self.scenario[2]        = "mode 4: reactor     -> stripper"
        self.scenario[3]        = "mode 6: separator   -> stripper"
        self.scenario[4]        = "mode 2: reactor     -> separator    -> stripper"
        self.scenario[5]        = "mode 1: reactor     -> stripper     -> separator"
        self.scenario[6]        = "mode 5: separator   -> stripper     -> separator"
        self.scenario[7]        = "mode 2: reactor     -> separator"
        

        self.allHeadline             = ['D feed',                   'E Feed',                   'A Feed',                   'A and C Feed',         'Recycle',              'Purge', 
                                        'Separator',                'Stripper',                 'Steam',                    'Reactor Coolant',      'Condenser Coolant',    'Agitator',                 # 12 manipulated variables
                                        'msv A Feed',               'msv D Feed',               'msv E Feed',               'msv A and C Feed',     'Recycle Flow',         'Reactor Feed Rate',        # 41 measured variables
                                        'Reactor Pressure',         'Reactor Temperature',      'Purge Rate',               'Product Sep Temp', 
                                        'Product Sep Pressure',     'Product Sep Underflow',    'Stripper Pressure',        'Reactor Coolant Temp',
                                        'Stripper Underflow',       'Stripper Temp',            'Stripper Steam Flow',      'Compressor Work',      
                                        'Separator Coolant Temp',   'Component A to Reactor',   'Component B to Reactor',   'Component C to Reactor',
                                        'Component D to Reactor',   'Component E to Reactor',   'Component F to Reactor',   'Component A to Purge',
                                        'Component B to Purge',     'Component C to Purge',     'Component D to Purge',     'Component E to Purge',
                                        'Component F to Purge',     'Component G to Purge',     'Component H to Purge',     'Component D to Product',
                                        'Component E to Product',   'Component F to Product',   'Component G to Product',   'Component H to Product',
                                        'Reactor Level',            'Product Sep Level',        'Stripper Level',
                                        'Liquid Input Stripper',    'Liquid Input Separator',   'Liquid Input Reactor']     #"rul"
        
        self.allWithoutLabel         = ['D feed',                   'E Feed',                   'A Feed',                   'A and C Feed',         'Recycle',              'Purge', 
                                        'Separator',                'Stripper',                 'Steam',                    'Reactor Coolant',      'Condenser Coolant',    'Agitator',                 # 12 manipulated variables
                                        'msv A Feed',               'msv D Feed',               'msv E Feed',               'msv A and C Feed',     'Recycle Flow',         'Reactor Feed Rate',        # 41 measured variables
                                        'Reactor Pressure',         'Reactor Temperature',      'Purge Rate',               'Product Sep Temp', 
                                        'Product Sep Pressure',     'Product Sep Underflow',    'Stripper Pressure',        'Reactor Coolant Temp',
                                        'Stripper Underflow',       'Stripper Temp',            'Stripper Steam Flow',      'Compressor Work',      
                                        'Separator Coolant Temp',   'Component A to Reactor',   'Component B to Reactor',   'Component C to Reactor',
                                        'Component D to Reactor',   'Component E to Reactor',   'Component F to Reactor',   'Component A to Purge',
                                        'Component B to Purge',     'Component C to Purge',     'Component D to Purge',     'Component E to Purge',
                                        'Component F to Purge',     'Component G to Purge',     'Component H to Purge',     'Component D to Product',
                                        'Component E to Product',   'Component F to Product',   'Component G to Product',   'Component H to Product',
                                        ]     #"rul"
        
        self.allWithoutLabelNoBias  = [ 'E Feed',                    'A Feed',                   'A and C Feed',            'Purge', 
                                        'Separator',                'Stripper',                 'Reactor Coolant',          'Condenser Coolant',                        # 12 manipulated variables
                                        'msv A Feed',               'msv D Feed',               'msv E Feed',               'msv A and C Feed',     'Recycle Flow',         'Reactor Feed Rate',        # 41 measured variables
                                        'Reactor Pressure',         'Reactor Temperature',      'Purge Rate',               'Product Sep Temp', 
                                        'Product Sep Pressure',     'Product Sep Underflow',    'Stripper Pressure',        'Reactor Coolant Temp',
                                        'Stripper Underflow',       'Stripper Temp',            'Compressor Work',      
                                        'Separator Coolant Temp',   'Component A to Reactor',   'Component B to Reactor',   'Component C to Reactor',
                                        'Component D to Reactor',   'Component E to Reactor',   'Component F to Reactor',   'Component A to Purge',
                                        'Component B to Purge',     'Component C to Purge',     'Component D to Purge',     'Component E to Purge',
                                        'Component F to Purge',     'Component G to Purge',     'Component H to Purge',     'Component D to Product',
                                        'Component E to Product',   'Component F to Product',   'Component G to Product',   'Component H to Product',
                                        ]     #"rul"

                                      
        
        # --------------------------
        # The sensor for each machine
        # --------------------------
        self.headline                = {}
        self.otherFactor             = ['D feed',                   'E Feed',                   'A Feed',                   'A and C Feed',         
                                        'Recycle',                  'Purge',                    'Steam',                    'Condenser Coolant',    
                                        'Agitator',                 'msv A Feed',               'msv D Feed',               'msv E Feed',                   # 12 manipulated variables
                                        'msv A and C Feed',         'Recycle Flow',             'Reactor Feed Rate',        'Purge Rate',                   # 41 measured variables
                                        'Compressor Work',          'Component A to Reactor',   'Component B to Reactor',   'Component C to Reactor',
                                        'Component D to Reactor',   'Component E to Reactor',   'Component F to Reactor',   'Component A to Purge',
                                        'Component B to Purge',     'Component C to Purge',     'Component D to Purge',     'Component E to Purge',
                                        'Component F to Purge',     'Component G to Purge',     'Component H to Purge',     'Component D to Product',
                                        'Component E to Product',   'Component F to Product',   'Component G to Product',   'Component H to Product',]

                
    
        # --------------------------------
        # Stable
        self.reactorHeadline = ["Stripper",               'Reactor Coolant',      "Reactor Coolant Temp", 'Stripper Underflow', 
                                'Component F to Reactor', 'Component F to Purge', "Component E to Product",]
                                

        self.separatorHeadline  = ['Reactor Pressure', 'Product Sep Temp', 'Product Sep Pressure']

        # self.stripperHeadline  = ["E Feed", "msv E Feed", "Stripper",               'Reactor Coolant',      "Reactor Coolant Temp", 'Stripper Underflow', 
        #                           'Component F to Reactor', 'Component F to Purge', "Component E to Product",]


        self.stripperHeadline    = [ "E Feed",   "A Feed",     "Purge",     "Separator", 
                                    "Purge Rate", "Product Sep Underflow",
                                      "Stripper Pressure",   "Component B to Reactor", 
                                     ]                        #stable                  
        
        # --------------------------------
        # Testing
        self.reactorHeadline = ['Reactor Coolant',      'Reactor Feed Rate', 'Reactor Pressure', 'Reactor Temperature', 
                                "Reactor Coolant Temp"]    

        self.separatorHeadline  = ['Separator', 'Product Sep Temp', 'Product Sep Underflow', "Separator Coolant Temp", "Compressor Work"]


        self.stripperHeadline   = [ "Stripper",     "Stripper Temp",  "Component G to Purge", "Component E to Product", "A and C Feed"              
                                    ]    


        # self.reactorHeadline    = ['Reactor Coolant']
        # self.separatorHeadline  = ['Product Sep Temp']
        # self.stripperHeadline   = ["Stripper"]  
        

        self.headline["reactor"]                = self.reactorHeadline
        self.headline["separator"]              = self.separatorHeadline
        self.headline["stripper"]               = self.stripperHeadline
        
        self.headline["all"]                    = self.allHeadline
        self.headline["allWithoutLabel"]        = self.allWithoutLabel
        self.headline["allWithoutLabelNoBias"]  = self.allWithoutLabelNoBias
        
        # self.reactorHeadline    = [ "Reactor Coolant",          'Reactor Pressure',         'Reactor Temperature'       'Reactor Coolant Temp'
        #                             ]
                                                                    
        # self.separatorHeadline  = [ "Separator",                'Product Sep Temp',         'Product Sep Pressure',     "Product Sep Underflow",   
        #                             "Separator Coolant Temp",      ]
                                
                                                                    
    
        self.degraLiq           = ['Reactor Level','Product Sep Level','Stripper Level']
        self.degraInput         = ['Liquid Input Stripper',    'Liquid Input Separator',   'Liquid Input Reactor']
        
        
    # ====================================
    # This function is used to generate 
    # the data which are alight with the
    # input of AI
    # ------------------------------------
    def getTrainData(self,):
        # package.include()
        # outputDir           = os.path.expanduser(self.config["dataDir"])
        outputDir = "./"
        rawData             = self.loadDataFromH5Format(dir = outputDir, name="data/teps.h5", case = "all", reduceSampleRate = True)
        # print(rawData.keys())
        # ----------------------------
        # Split train and test data
        # ----------------------------
        if self.config["splitTrainTest"]:
            rawTrainData = {}
            rawTestData  = {}
            for idxKeys,idxValues in rawData.items():
                rawTrainData[idxKeys] = idxValues[idxValues["Id"] <=    self.config["trajectoryPerCase"]].copy()
                rawTestData[idxKeys]  = idxValues[idxValues["Id"] >     self.config["trajectoryPerCase"]].copy()
        else:
            rawTrainData = rawData

        withRULData         = self.addRULAndEnvironment(rawTrainData)
        # withoutInitRawData  = self.removeInitState(inData= withRULData)

        # ---------------------------
        # For data
        cleanData           = self.cleanData(   rawSignal           =   withRULData,
                                                columnName          =   self.allWithoutLabel,
                                                getMovingAverage    =   False,
                                                flipToDown          =   False)
        # ---------------------------
        # For label, flip only
        cleanData           = self.cleanData(rawSignal              =   cleanData,
                                                columnName          =   self.degraLiq,
                                                getMovingAverage    =   False,
                                                flipToDown          =   False)
        norData             = self.normalizeDataEachModes(  rawData    =   cleanData,
                                                            headline   =   self.headline["allWithoutLabelNoBias"])   # Normalize the data
        aiLogger.info(norData.keys())
        # ---------------------------
        # Extract the data in each Case
        allSeqXAllSensorTrain   = []
        allSeqXRulTrain         = []
        allSeqXEnvTrain         = []
        allSeqTimeTrain         = []

        allSeqXAllSensorVal     = []
        allSeqXRulVal           = []
        allSeqXEnvVal           = []
        allSeqTimeVal           = []
        nbOfTrainData           = 100
        for idxCase in self.config["case"]:
            data                = norData[idxCase]
            # print(data.head)
            trajectID           = data["Id"]
            # reactor             = data[preData.headline["reactor"]]
            # separator           = data[preData.headline["separator"]]
            # stripper            = data[preData.headline["stripper"]]
            allSensor           = data[self.headline["allWithoutLabelNoBias"]]
            # allSensor           = data[self.headline["reactor"]]
            rul                 = data["rul"]
            timeTraj            = data["Time"]
            env                 = data["environmentID"]


            trainTrajectID      = trajectID[trajectID <=nbOfTrainData]
            valTrajectID        = trajectID[trajectID >nbOfTrainData]
            
            trainAllSensor      = allSensor[trajectID <=nbOfTrainData]
            valAllSensor        = allSensor[trajectID > nbOfTrainData]
        

            trainRul            = rul[trajectID <=nbOfTrainData]
            valRul              = rul[trajectID > nbOfTrainData]

            trainEnv            = env[trajectID <=nbOfTrainData]
            valEnv              = env[trajectID > nbOfTrainData]

            trainTime           = timeTraj[trajectID <=nbOfTrainData]
            valTime             = timeTraj[trajectID > nbOfTrainData]



            # ---------------------------
            # For training
            seqXAllSensorTrain,     \
            seqXRulTrain,           \
            seqXTrainEnv,seqTimeTrain        = self.getTensorData(dataList    = [trainAllSensor,trainRul,trainEnv,trainTime],
                                                                trajectId   = trainTrajectID,  
                                                                shuffle     = True,      
                                                                seqLength   = self.config["seqLength"],
                                                                overlap     = self.config["overLap"]         
                                                                )
            
            # print(trajectID.shape)
            # print(seqXAllSensorTrain.shape)
            allSeqXAllSensorTrain.append(seqXAllSensorTrain)  
            allSeqXRulTrain.append(seqXRulTrain)  
            allSeqXEnvTrain.append(seqXTrainEnv) 
            allSeqTimeTrain.append(seqTimeTrain) 

            # ---------------------------
            # For validation
            seqXAllSensorVal,     \
            seqXRulVal,           \
            seqXEnvVal,seqTimeVal        = self.getTensorData(  dataList    = [valAllSensor,valRul,valEnv,valTime],
                                                                trajectId   = valTrajectID,  
                                                                shuffle     = True,      
                                                                seqLength   = self.config["seqLength"],
                                                                overlap     = self.config["overLap"]         
                                                                )
            
            allSeqXAllSensorVal.append(seqXAllSensorVal)
            allSeqXRulVal.append(seqXRulVal)
            allSeqXEnvVal.append(seqXEnvVal)
            allSeqTimeVal.append(seqTimeVal)


        allSeqXAllSensorTrain   = np.concatenate(allSeqXAllSensorTrain, axis=0)
        allSeqXRulTrain         = np.concatenate(allSeqXRulTrain, axis=0)
        allSeqXEnvTrain         = np.concatenate(allSeqXEnvTrain, axis=0)
        allSeqTimeTrain         = np.concatenate(allSeqTimeTrain, axis=0)


        allSeqXAllSensorVal   = np.concatenate(allSeqXAllSensorVal, axis=0)
        allSeqXRulVal         = np.concatenate(allSeqXRulVal, axis=0)
        allSeqXEnvVal         = np.concatenate(allSeqXEnvVal, axis=0)
        allSeqTimeVal         = np.concatenate(allSeqTimeVal, axis=0)


        xTrain              = [allSeqXAllSensorTrain, allSeqTimeTrain[:,-1,:],allSeqXEnvTrain[:,-1,:]]
        xVal                = [allSeqXAllSensorVal, allSeqTimeVal[:,-1,:],allSeqXEnvVal[:,-1,:]]
        yTrain              = [allSeqXRulTrain[:,-1,:]]
        yVal                = [allSeqXRulVal[:,-1,:]]
        print(f"The size of the train data: xTrain {allSeqXAllSensorTrain.shape} Time {allSeqTimeTrain[:,-1,:].shape} Environment {allSeqXEnvTrain[:,-1,:].shape} RUL {allSeqXRulTrain[:,-1,:].shape}")
        print(f"The size of the val data: xVal {allSeqXAllSensorVal.shape} Time {allSeqTimeVal[:,-1,:].shape} Environment {allSeqXEnvVal[:,-1,:].shape} RUL {allSeqXRulVal[:,-1,:].shape}")
        return xTrain, yTrain, xVal, yVal

    # ===================================
    # This function is used to load the 
    # .csv file
    # -----------------------------------
    def loadCSVDataFile(self,dir = "data/", removeHigherThr = False, thr = [], columnNames = []):
        '''
        removeHigherThr : if True, remove the higher position of trajectories.
        thr             : It can be min, max or values. If min max is use, use the min or max of thresholds.
        columnName      : The column name to check
        '''

        aiLogger.info("Loading data from files ...")
        self.case          = {}  
        self.case[1]       = pd.read_csv(dir + 'data/case1mode3.csv')
        self.case[2]       = pd.read_csv(dir + 'data/case2mode6.csv')
        self.case[3]       = pd.read_csv(dir + 'data/case3mode2.csv')
        if removeHigherThr:
            for idx,keyCase in enumerate(self.case.keys()):
                print("--------------------------------------", keyCase)
                clName             = columnNames[idx]
                curCase            = self.case[keyCase]
                minAllID           = self.case[keyCase].groupby("Id")[clName].min()
                maxAllID           = self.case[keyCase].groupby("Id")[clName].max()
                print(minAllID, maxAllID)
                removeTHR          = minAllID.max() if thr[idx] == "down" else maxAllID.min()
                print("Threshold",removeTHR)
                self.case[keyCase] = curCase[curCase[clName] > removeTHR] if thr[idx] == "down" else curCase[curCase[clName] < removeTHR]
            
        # self.case[4]       = pd.read_csv('data/case4.csv')
        # self.case[5]       = pd.read_csv('data/case5.csv')
        # self.case[6]       = pd.read_csv('data/case6.csv')
        return self.case
    # ===================================
    # This function is used to save the 
    # file into H5 File
    # -----------------------------------
    def save2H5Format(self,dir = "data/",data = None):
        self.case = self.case if data is None else data
        aiLogger.info("Saving data to h5 file ...")
        for idxCase,valueCase in self.case.items():

            valueCase.to_hdf(dir + 'data/newTeps.h5', key=f"case{idxCase}", mode='a')

        aiLogger.info("Finish saving data to h5 file ...")


    # ===================================
    # This function is used to load the h5 file
    # -----------------------------------
    def loadDataFromH5Format(self,dir   = "./", 
                             name       = "data/teps.h5", 
                             case       = "all", 
                             reduceSampleRate   = False, 
                             removeHigherThr    = False, 
                             columnNames        = ["Product Sep Level", "Stripper Level","Stripper Level","Stripper Level", "Product Sep Level", "Product Sep Level"], 
                             trend              = ["up", "up", "up", "up", "up", "up"]):
        aiLogger.info("Loading data from H5 files ...")
        # print(dir)
        # ---------------------
        # Reading all cases
        if case == "all":
            self.rawData = {}
            for idxCase in range(self.numOfCases):
                curCase                         = idxCase + 1                                           # index starts from 0, case starts from 1
                curRawData                      = pd.read_hdf(name, key='case' + str(curCase))
                self.rawData[curCase]           = curRawData[curRawData.index % 2 != 1] if reduceSampleRate else curRawData

                if removeHigherThr:
                    aiLogger.info(f"Remove the exceeded amplitude at the case {curCase}")
                    clName             = columnNames[idxCase]
                    curCaseRm          = self.rawData[curCase].copy()
                    minAllID           = curCaseRm.groupby("Id")[clName].min()
                    maxAllID           = curCaseRm.groupby("Id")[clName].max()
                    aiLogger.debug(f" The min and max value at the case {curCase} are {minAllID, maxAllID}")
                    removeTHR          = minAllID.max() if trend[idxCase] == "down" else maxAllID.min()
                    aiLogger.debug(f"Threshold to remove is {removeTHR}")
                    self.rawData[curCase] = curCaseRm[curCaseRm[clName] > removeTHR] if trend[idxCase] == "down" else curCaseRm[curCaseRm[clName] < removeTHR]
                    # print(self.rawData[curCase])
        # ---------------------
        # Reading 1 case only
        else:
            self.rawData = pd.read_hdf(name, key='case' + str(case))
        
        aiLogger.info(f"Finish loading data from H5 files at case {case}...")
        return self.rawData

    # ===================================
    # Print scenario
    # -----------------------------------
    def showScenario(self,):
        for key,value in self.scenario.items():
            print(f'scenario {key}: {value}')

    # ===================================
    # This function is used to get a 
    # trajectory from a specific case
    # -----------------------------------
    def getTrajectoryAt(self,   inData          = None,
                                case            = 2,
                                idxTrajectory   = 1):
        dataCase   = inData[case]
        # print(dataCase)
        trajectory = dataCase[dataCase['Id'] == idxTrajectory]

        return trajectory
    
    # ===================================
    # This function is used to add RUL label
    # into the current pandas
    # -----------------------------------
    def addRULAndEnvironment(self,inData):
        # -------------------
        # Insert the environment
        # variable
        # -------------------
        for key in inData.keys():
            inData[key].insert(2,"environmentID",key)                     # Insert the case = idx + 1
            # aiLogger.debug(key)

        # self.headline["reactor"]    = ["environmentID"]    +    self.headline["reactor"]
        # self.headline["separator"]  = ["environmentID"]    +    self.headline["separator"]    
        # self.headline["stripper"]   = ["environmentID"]    +    self.headline["stripper"]     

        # print(self.headline["reactor"])

        withRulData = inData.copy()
        # -------------------
        # Run for each case
        # -------------------
        for caseKey,values in inData.items():       
            withRulData[caseKey]["rul"]   = values.groupby('Id',sort=False)['Time'].transform(lambda x: x.max() - x)
            
        aiLogger.info("Adding rul successfully...")
        return withRulData

    # ===================================
    # Get trajectory
    # -----------------------------------
    def getTrajectoryAt(self,   inData          = None,
                                case            = 2,
                                idxTrajectory   = 1):
        dataCase   = inData[case]
        # print(dataCase)
        trajectory = dataCase[dataCase['Id'] == idxTrajectory]

        return trajectory.copy()
    
    # ===================================
    # This function is chose to select the
    # right sensor
    # -----------------------------------
    def selectSensor(self,  inData      = None,
                            case        = 2,
                            period      = 128
                            ):
        tiny    = 1e-20       
        trajec  = self.getTrajectoryAt( inData          = inData,
                                        case            = case,
                                        idxTrajectory   = 1)
        
        diffReac    = trajec['Reactor Level'].diff(periods=period).dropna()
        diffSep     = trajec['Product Sep Level'].diff(periods=period).dropna()
        diffStrip   = trajec['Stripper Level'].diff(periods=period).dropna()

        for headline in self.allHeadline:
            curSig      = trajec[headline].diff(periods=period).dropna()
            print(headline)
            # print("----------------------------")
            # print(curSig)
           
            corrReac    =  diffReac.corr(curSig + tiny)
            corrSep     =  diffSep.corr(curSig + tiny)
            corrStrip   =  diffStrip.corr(curSig + tiny)
            # corrReac    =  np.corrcoef(curSig, diffReac)[0, 1]
            # corrSep     =  np.corrcoef(curSig, diffSep)[0, 1]
            # corrStrip   =  np.corrcoef(curSig, diffStrip)[0, 1]
            # print(corrReac)
            # print("================================")
            print(corrStrip)
            print("================================")
            # print(corrStrip)
            # print(corrReac, corrSep, corrStrip)
        return None


    # ===================================
    # This function is used to remove 
    # the initial stage of the data
    # -----------------------------------
    def removeInitState(self,inData):
        initPoint  = int(self.stabletime*60/self.sampleRateTime) + 1           # The intitial point at a stable time
        print(initPoint)
        outData    = inData.copy()                            
        for caseKey,value in inData.items():
            eachCaseFiltered            = value.groupby('Id',sort=False).apply(lambda x: x.iloc[initPoint:]).reset_index(drop=True)
            eachCaseFiltered["Time"]    = eachCaseFiltered["Time"] - self.stabletime                        # Move time to the original point

            outData[caseKey]             = eachCaseFiltered

        return outData
    # ===================================
    # This function is used to clean data
    # -----------------------------------
    def cleanData(self, rawSignal,
                        getMovingAverage    = False,
                        flipToDown          = False,
                        columnName          = None,
                        windowSize          = 127):
        
        aiLogger.info("Cleaning data ...")
        # --------------------------
        # Get average moving or 
        # flip to down
        # --------------------------
        outSignal   = rawSignal.copy()                         # Copy the data to avoid changing the underline data

        if columnName is None:
            columnName  = rawSignal[1].columns[3:-4]               # Get a random case, and get the name of all column, exclude the time, ID, environment, inputs and rul
        
        for caseKey,value in outSignal.items():                 # Run each case to get a case from a dictionary, each case contains a pandas
            for col in columnName:                              # Run with each pandas column to get the feature
                if getMovingAverage:
                    aiLogger.info(f"Getting average data at case {caseKey} and column {col} ...")
                    outSignal[caseKey][col]   = value.groupby('Id',sort=False)[col].transform(lambda x: self.getMovingAverage(x, windowSize))   
                
            # if flipToDown:
            #     aiLogger.info(f"Flipping data at case {caseKey} and column {col} ...")
            #     outSignal[caseKey][col]   = value.groupby('Id',sort=False)[col].transform(lambda x: self.flip2Down(x))   
            if flipToDown:
                aiLogger.info(f"Flipping data at case {caseKey}")
                outSignal[caseKey][columnName]      = value.groupby("Id", sort=False)[columnName].apply(lambda g: self.flip2DownPandasMultipleCol(g, g.columns))

        return outSignal

    
    def getMovingAverage(self,values, windowSize):
        
        kernel      = np.ones(int(windowSize))/float(windowSize)
        padWidth    = int(windowSize/2)                         # Equivalent to the floor(window_size/2) for window_size = 3

        padSig      = np.pad(values, padWidth, mode='symmetric') 
        smoothedSig = np.convolve(padSig, kernel, 'valid')
        return smoothedSig
        

    def flip2Down(self,data):
        
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
    

    def flip2DownPandasMultipleCol(self, df_group, colNames, flipOption = "average"):
        df_copy = df_group.copy()

        for col in colNames:
            series = df_copy[col].values
            time_points = np.arange(len(series))
            coefficients = np.polyfit(time_points, series, 2)

            if coefficients[0] >= 0:
                if flipOption == "average":
                    # Mirror around the mean to flip the trend but keep values realistic
                    mean_val = series.mean()
                    df_copy[col] = 2 * mean_val - series  # Flipped around mean
                elif flipOption == "flipWithNegative":
                    df_copy[col] = series * -1

                elif flipOption == "maxmin":
                    df_copy[col] = series.max() - (series - series.min())
           
        return df_copy[colNames]

    # ===================================
    # This function is used to move all
    # all the data point to the peak at 
    # the top
    # -----------------------------------
    def movingToPeak(self,inData = None):
        aiLogger.info("Moving data coordination ...")

        maxAllCase,_ = self.getMaxMinPandas(inData)             # Get the max values of each feature
        # print(maxAllCase.shape)
        # ----------------------------
        # Move all the trajectory to the 
        # position of higest trajectory
        # ----------------------------
        outData = inData.copy()
        for caseKey,values in inData.items():
            # ------------------------
            # Get the max value of each
            # trajectory
            # ------------------------
            maxEachCase     = [self.getTrajectoryAt(inData=inData,case = caseKey, idxTrajectory = idxTraject + 1).max(axis = 0) for idxTraject in range(self.trajectoryPerCase)]
            
            # print(maxEachCase[0].shape)
            # ------------------------
            # Get the distance from the
            # max point to a certain 
            # trajectory
            # ------------------------
            distEachCase    = [(maxAllCase - maxEachCase[idx]) for idx in range(self.trajectoryPerCase)]

            tempValues = pd.DataFrame()
            for idx,dist in enumerate(distEachCase):
                # print(dist.shape)
                # print(values[values['Id'] == idx + 1].shape)
                # print(values[values['Id'] == idx + 1][self.allHeadline])
                # print(dist[self.allHeadline])
                temp = values[values['Id'] == idx + 1].copy()
                temp[self.allHeadline] = values[values['Id'] == idx + 1][self.allHeadline] + dist[self.allHeadline]
                tempValues = pd.concat([tempValues, temp], ignore_index=True)

            outData[caseKey] = tempValues  
        return outData
    
    # ===================================
    # This function is used to move all
    # all the data point to the peak at 
    # the top
    # -----------------------------------
    def movingHiToPeak(self,inData = None):
        aiLogger.info("Moving HI coordination ...")

        maxAllComp,_ = self.getMaxNestedMatrix(inData)             # Get the max values of each feature

        # print(maxAllCase.shape)
        # ----------------------------
        # Move all the trajectory to the 
        # position of higest trajectory
        # ----------------------------
        outData = inData.copy()
        for caseKey,values in inData.items():
            # ------------------------
            # Get the max value of each
            # trajectory
            # ------------------------
            maxEachComp     = [inData[caseKey][idxTraject].max() for idxTraject in range(self.trajectoryPerCase)]
            
            # ------------------------
            # Get the distance from the
            # max point to a certain 
            # trajectory
            # ------------------------
            distEachComp    = [(maxAllComp - maxEachComp[idx]) for idx in range(self.trajectoryPerCase)]

            for idx,dist in enumerate(distEachComp):
                outData[caseKey][idx]   = outData[caseKey][idx] + dist     

        return outData
    
    def normalizeHIInGeneral(self,inData = [None], indexToShutdown = 2, time = None, rul = None, multipliedMt = None, showRedundantShutdown = True):
        
        max_values  = [tf.reduce_min(tensor).numpy() for tensor in inData[indexToShutdown]]
        # print(max_values)
        threshold   = max(max_values)  # Adjust 20 if needed
        mask        = [tf.greater_equal(curHI, threshold) for curHI in inData[indexToShutdown]]        # Create a mask for redundant shutdown
        
        time        = [time[idx][mask[idx]] for idx in range(len(time))] if not showRedundantShutdown else time
        rul         = [rul[idx][mask[idx]] for idx in range(len(rul))] if not showRedundantShutdown else rul
        rul         = [rul[idx] - rul[idx][-1] for idx in range(len(rul))] if not showRedundantShutdown else rul
        multipliedMt = [multipliedMt[idx].numpy()[mask[idx].numpy().flatten()] for idx in range(len(multipliedMt))] if not showRedundantShutdown else multipliedMt

        outData = []
        for curData in inData:
            normalized_sublist = []
            for idxTensor, tensor in enumerate(curData):
                # Apply normalization to each TensorFlow tensor
                curHI = 1 - (tensor / threshold)

                # Clip negative values to 0
                if showRedundantShutdown:
                    curHI = tf.where(curHI < 0, 0.0, curHI)
                else:
                    curHI           = curHI[mask[idxTensor]]
                    

                normalized_sublist.append(curHI)

            outData.append(normalized_sublist)

        return outData, time, rul, multipliedMt

        # return outData
    

    # ==============================
    # This function is used to normalize
    # the health indicator
    # ==============================
    def normalizeHIEachTraject(self,inData = None):

        for idx in range(self.trajectoryPerCase):
            maxEachTraj = max(np.max(inData["reactor"][idx]),np.max(inData["separator"][idx]), np.max(inData["stripper"][idx]))
            minEachTraj = min(np.min(inData["reactor"][idx]),np.min(inData["separator"][idx]), np.min(inData["stripper"][idx]))
            
            inData["reactor"][idx]  = (inData["reactor"][idx] - minEachTraj)/(maxEachTraj-minEachTraj)
            inData["separator"][idx]  = (inData["separator"][idx] - minEachTraj)/(maxEachTraj-minEachTraj)
            inData["stripper"][idx]  = (inData["stripper"][idx] - minEachTraj)/(maxEachTraj-minEachTraj)

        # maxAllComp, minAllComp = self.getMaxNestedMatrix(inData)             # Get the max values of each feature

        # for comp in inData.keys():
        #     inData[comp] = [(inData[comp][idx] - minAllComp)/(maxAllComp- minAllComp) for idx in range(self.trajectoryPerCase)]

        return inData


    def getMaxNestedMatrix(self,nested_dict):
        max_value = -np.inf  # Initialize to negative infinity to ensure any number is larger
        min_value =  np.inf
        for outer_key, inner_dict in nested_dict.items():
            for matrix in inner_dict:
                max_in_matrix = np.max(matrix)  # Get the max value in the current matrix
                min_in_matrix = np.min(matrix)  # Get the min value in the current matrix

                # ------------------------
                if max_in_matrix > max_value:
                    max_value = max_in_matrix  # Update the max value if the current one is larger

                if min_in_matrix < min_value:
                    min_value = min_in_matrix  # Update the min value if the current one is smaller

        return max_value, min_value
    
    # ===================================
    # This function is used to get max 
    # and min in a dictionary of pandas
    # -----------------------------------
    def getMaxMinPandas(self,inData):
        maxEachCase = pd.DataFrame(columns=inData.keys())          # Create an empty pandas with different key from cases
        minEachCase = pd.DataFrame(columns=inData.keys())          # Create an empty pandas with different key from cases

        for caseKey,value in inData.items():                       # Run for loop in each case to get the max and min from pandas
            maxEachCase[caseKey] = value.max()
            minEachCase[caseKey] = value.min()
        
        maxAllCase = maxEachCase.max(axis=  1)                      # Get the max value in each row (each feature) from cases     
        minAllCase = minEachCase.min(axis = 1)                      # Get the min value in each row (each feature) from cases
        # print(maxAllCase)
        # print(minAllCase)
        
        return maxAllCase,minAllCase
    

    # ===================================
    # This function is used to normalize 
    # the data
    # -----------------------------------
    def normalizeDataAllModes(self,rawData):
        aiLogger.info("Normalizing the data ...")
        
        tiny        = 1e-20       
        maxEachCase = pd.DataFrame(columns=rawData.keys())          # Create an empty pandas with different key from cases
        minEachCase = pd.DataFrame(columns=rawData.keys())          # Create an empty pandas with different key from cases

        for caseKey,value in rawData.items():                       # Run for loop in each case to get the max and min from pandas
            maxEachCase[caseKey] = value.max()
            minEachCase[caseKey] = value.min()
        
            # print("max:",maxEachCase[caseKey]["Reactor Temperature"])
            # print("min:",minEachCase[caseKey]["Reactor Temperature"])
        
        maxAllCase = maxEachCase.max(axis=  1)                      # Get the max value in each row (each feature) from cases     
        minAllCase = minEachCase.min(axis = 1)                      # Get the min value in each row (each feature) from cases
        # print(maxAllCase)
        # print(minAllCase)
        # ----------------------
        # Normalize data
        # ----------------------
        # print(rawData)
        for caseKey in rawData.keys():  
            rawData[caseKey].iloc[:,3:-1]  = (rawData[caseKey].iloc[:,3:-1] - minAllCase[3:-1].T) / (maxAllCase[3:-1].T - minAllCase[3:-1].T + tiny) # exclude the time, ID, environment and rul
        
        # print("=====================================")
        # print(rawData[2]["Reactor Temperature"][100000])
        return rawData
    
    # ===================================
    # This function is used to normalize
    # the data from the input rawData
    # -----------------------------------
    def normalizeDataEachModes(self,rawData,headline = None, returnMaxMin = False):
        '''
        rawData:    the dictionary of the data with the format of {case: value}, in which case is the case of the TEP,
                    while the value are trajectories identified through IDs. The size of the value is rowxcol (row: IDxrowInEachTrajectory, col: features)
                    case = trajectories in multiple ID
        headline: the headline which you want to make the normalization
        '''
        aiLogger.info("Normalizing the data ...")

        # ----------------------
        # Convert all to float 64
        # for consistence
        # ----------------------
        for caseKey in rawData.keys():
            rawData[caseKey] = rawData[caseKey].astype('float64')

        # maxEachCase = pd.DataFrame(columns=rawData.keys())          # Create an empty pandas with different key from cases
        # minEachCase = pd.DataFrame(columns=rawData.keys())          # Create an empty pandas with different key from cases
        maxEachCase = {}
        minEachCase = {}
        for caseKey,value in rawData.items():                       # Run for loop in each case to get the max and min from pandas
            maxEachCase[caseKey] = value.max(axis = 0)
            minEachCase[caseKey] = value.min(axis = 0)

            # print("casekey", caseKey)
            # print(f"The max value of case {caseKey} is {maxEachCase}")
            # print(f"The min value of case {caseKey} is {minEachCase}")
   
            # rawData[caseKey].iloc[:,3:-1]  = (rawData[caseKey].iloc[:,3:-1] - minEachCase[3:-1].T) / (maxEachCase[3:-1].T - minEachCase[3:-1].T)
            rawData[caseKey][headline]  = (rawData[caseKey][headline] - minEachCase[caseKey][headline].T) / (maxEachCase[caseKey][headline].T - minEachCase[caseKey][headline].T)
        
        if returnMaxMin:
            return maxEachCase,minEachCase
        else:
            return rawData

    
    # ===================================
    # This function is used to split data
    # into each sequences and return the 
    # tensor data for each component
    # -----------------------------------
    def getTensorData(self, dataList        = [None], 
                            trajectId       = None, 
                            env             = None,
                            getAtIdx        = None,
                            shuffle         = False,
                            getLastOnly     = False,
                            seqLength       = 128,
                            overlap         = 127):
        '''
        dataList(*)     :   a list of pandas with multiple trajectory which is identified through IDs. Each should be the length of rowxcol (row: IDxrowInEachTrajectory, col: features),
                            Each element should be the same 1st Dimention-shape(0), for example [data1, data2, data3] with the length of (221918, 60) (221918, 6) (221918,)
                            and it must be a list of numpy or pandas consistenly  
        trajectId(*)    :   the matrix to have the same row with pandasData aiming to identify the which row belong to which Id

        getAtIdx        :   Access a specific Idx
        shuffle         :   Shuffle the order of windows or not
        getLastOnly     :   Only get the last element in each window

        - Use for a specific case:
        seqXAllSensorTrain,seqXRulTrainEnv     = preData.getTensorData( dataList    = [allSensor,env],
                                                                        trajectId   = trajectId
                                                                        getAtIdx   = idx)
        - Use for training:
        seqXReacTrain,          \
        seqXSepTrain,           \
        seqXStripTrain,         \
        seqXAllSensorTrain,     \
        seqXRulTrain,           \
        seqXRulTrainEnv     = preData.getTensorData(dataList    = [reactor,separator,stripper,allSensor,rul,env],
                                                    trajectId   = trajectID,  
                                                    shuffle     = True          
                                                    )
                                                                                                                                
        (*): obligation
        '''
        
        aiLogger.info("Spliting data into sequences now ...")

        # -----------------------
        # Check if getAtIdx out
        # of range
        # -----------------------
        # if getAtIdx is not None and getAtIdx not in trajectId:
        #     print("Test again the getAtIdx value, it's out of range!!!")

        pandasDataList  = [pd.DataFrame(array) for array in dataList] if isinstance(dataList[0], np.ndarray) else dataList      # Convert to pandas if numpy         
        pandasData      = [value.to_frame() if value.ndim == 1 else value for value in pandasDataList]                          # Add new dimension

        storedDimIdx2   = [value.shape[1] for value in pandasData]                                                              # Get the size of each entry to extract later
        storedDimIdx2   = list(accumulate(storedDimIdx2))
        storedDimIdx1   = [0] + storedDimIdx2[:-1]
        # mergedAll       = np.concatenate(pandasData, axis=1)                                                                  # Merge in the same numpy to suffle and extract train and val, stack with col together because we have the same row
        mergedAll       = pd.concat(pandasData, axis=1)

        # print(len(mergedAll))
        # print(trajectId)
        # -----------------------
        # Declare sequence list
        # and tensor list for each
        # components
        # -----------------------
        allSeq      = []   

        # ---------------------------
        # Get the tensor for all idx
        # in each case
        # ---------------------------
        if getAtIdx is None:
            for idxId in range(self.trajectoryPerCase):                                                 # Get each ID
                trajectory          = mergedAll[trajectId.values == (idxId + 1)]                                 # Extract tracjectory from ID
                componentTrajectory = trajectory.reset_index(drop = True)                               # Get the only sensors for each component
            
                sequences           = self.splitAndPadSequences(data                = componentTrajectory, 
                                                                seqLength           = seqLength, 
                                                                overlap             = overlap,
                                                                getLastElementOnly  = getLastOnly)           # Split the each trajectory into each seq
                allSeq      = allSeq     + sequences                     # trajectories

        # ---------------------------
        # Get the tensor at a specific
        # index
        # ---------------------------
        else:
            trajectory          = mergedAll[trajectId == getAtIdx]                           # Extract tracjectory from ID
            # print("-----------------------",len(trajectory))

            componentTrajectory = trajectory.reset_index(drop = True)     # Get the only sensors for each component
            
            sequences           = self.splitAndPadSequences(data                = componentTrajectory, 
                                                            seqLength           = seqLength, 
                                                            overlap             = overlap,
                                                            getLastElementOnly  = getLastOnly)           # Split the each trajectory into each seq
            
            allSeq      = allSeq  + sequences                     # trajectories

        # -------------------
        # Convert the list 
        # to tensor with np
        # -------------------
        tensor =   np.stack(allSeq, axis=0)   

        # aiLogger.debug(f"The shape of the tensor is {tensor.shape}")
        if shuffle:
            np.random.shuffle(tensor)

        # print("tesnsor shape",tensor.shape)
        # for (idx1,idx2) in zip(storedDimIdx1,storedDimIdx2):
        #     print(idx1,idx2)
        #     test = tensor[:,:,idx1:idx2]
        #     print(test.shape)

        tensorList    = [tensor[:,:,idx1:idx2]   for (idx1,idx2) in zip(storedDimIdx1,storedDimIdx2)]
        return tensorList

    # -----------------------------------
    # This function is used to split data 
    # into sequences.
    # -----------------------------------
    def splitAndPadSequences(self,  data                = None, 
                                    seqLength           = 128, 
                                    overlap             = 127,
                                    getLastElementOnly  = False ):
        '''
        data        : a pandas single trajectory with multiple column or 1 column
        seqLength   : the length of window to split the data
        overlap     : the overlap between previous window and the next window
        getLastElementOnly : instead return a data from a window (start to end), return only the last point in each window (like Rul)
        '''
        sequences       = []
        step_size       = seqLength - overlap
        totalLength     = len(data)
        # print("total length",totalLength)
        # print(data.shape)
        # print(data)
        for start in range(0, totalLength-overlap, step_size):
            end = start + seqLength
            # ----------------------
            # If the sequence would 
            # go beyond the data length, 
            # pad it
            # ----------------------
            if end > totalLength:      
                aiLogger.error("The ending loop should be lower totalLength, please check again!!!")
                # raise Exception("The ending loop should be lower totalLength, please check again!!!")          
                # padding  = end - totalLength
                # # sequence = torch.nn.functional.pad(dataX[:, start:], (0, padding), 'constant', 0)
                # curLalel = dataY[:,-1]
            else:
                if getLastElementOnly:
                    sequence = data[end-1]                  # Get the last element in the sequence
                else:
                    sequence = data[start:end]              # Get sentense
                
            sequences.append(sequence)
        
        return sequences
    

    # ===================================
    # This function is used to split data
    # to train, val, test groups according to
    # the ID of trajectories. For example:
    # we have 100 signals, then we shuffle
    # these 100 signals to get a scale for
    # train (for example 70 signals) and 
    # a scale for validation (for example 
    # 30 signals) later. 
    # -----------------------------------
    def splitTrajectoryTrainVal(self,   inData                  = [None], 
                                        trainPercent            = 0.7,
                                        shuffle                 = False,
                                        ):
        '''
        inData      : the input is the list of data with the same 1st Dimention-shape(0), for example [data1, data2, data3] with the length of (221918, 60) (221918, 6) (221918,)
                      and it must be a list of numpy or pandas consistenly 
        trainPercent: the percentage need to split
        shuffle     : shuffle the data according the 1st dimention or not 
        outData     : contain the train and val values with the same list as initial under the numpy type
        '''
        outData         = {}

        # ----------------------------------
        # Add the dim if only have 1 dim
        # ----------------------------------
        if isinstance(inData[0], np.ndarray):
            reshapedList    = [value.reshape(-1, 1) if value.ndim == 1 else value for value in inData]    
        else:
            reshapedList    = [value.values.reshape(-1, 1) if value.ndim == 1 else value.values for value in inData]

        storedDimIdx2   = [value.shape[1] for value in reshapedList]                                                # Get the size of each entry to extract later
        storedDimIdx2   = list(accumulate(storedDimIdx2))
        storedDimIdx1   = [0] + storedDimIdx2[:-1]
        mergedAll       = np.concatenate(reshapedList, axis=1)                                                       # Merge in the same numpy to suffle and extract train and val, stack with col together because we have the same row

        # print(storedDimIdx1,storedDimIdx2,mergedAll.shape)

        if shuffle:
            shuffledIdx     = np.random.permutation(mergedAll.shape[0])                                             # Shuffle the data
            shuffledArray   = mergedAll[shuffledIdx]
        else:
            shuffledArray   = mergedAll

        split_index = int(len(shuffledArray) * trainPercent)                                                        # Split the shuffled array into training (2/3) and validation (1/3)
        trainData  = shuffledArray[:split_index]
        valData    = shuffledArray[split_index:]

        outData["train"]    = [trainData[:,idx1:idx2]   for (idx1,idx2) in (storedDimIdx1,storedDimIdx2)]
        outData["val"]      = [valData[:,idx1:idx2]     for (idx1,idx2) in (storedDimIdx1,storedDimIdx2)]

        return outData
    

    def mergeCompToSystemHI(self,compHi):

        # geometric_system_hi = [np.exp(np.mean(np.log(np.array([compHi["reactor"][idxTraj], compHi["separator"][idxTraj], compHi["stripper"][idxTraj]])), axis=0)) for idxTraj in range(self.trajectoryPerCase)]

        # Calculate minimum HI
        systemHI = [np.minimum(np.minimum(compHi["reactor"][idxTraj], compHi["separator"][idxTraj]), compHi["stripper"][idxTraj]) for idxTraj in range(self.trajectoryPerCase)]

        return systemHI
    
    def getTHRLiquidLabel(self,rawData):
        reacTHR     = {}
        sepTHR      = {}
        stripTHR    = {}
        # ['Reactor Level','Product Sep Level','Stripper Level']

        for idxCase,valueCase in rawData.items():
            minPerTraject       = valueCase.groupby('Id',sort=False)["Reactor Level"].min()
            reacTHR[idxCase]    = minPerTraject.min()
        

            minPerTraject       = valueCase.groupby('Id',sort=False)["Product Sep Level"].min()
            sepTHR[idxCase]     = minPerTraject.min()

            minPerTraject       = valueCase.groupby('Id',sort=False)["Stripper Level"].min()
            stripTHR[idxCase]   = minPerTraject.min()

            maxPerTraject       = valueCase.groupby('Id',sort=False)["Reactor Level"].max()
            maxPerTraject       = valueCase.groupby('Id',sort=False)["Product Sep Level"].max()
            maxPerTraject       = valueCase.groupby('Id',sort=False)["Stripper Level"].max()


        return reacTHR,sepTHR,stripTHR

    def normalizeLabelHI(self,labelHI):
        maxReac     = 128
        maxSep      = 134
        maxStrip    = 129

        minReac     = 63.66
        minSep      = 46
        minStrip    = 45

        labelHI["Reactor Level"]        = (labelHI["Reactor Level"] - minReac)/(maxReac -minReac)
        labelHI["Product Sep Level"]    = (labelHI["Product Sep Level"] - minSep)/(maxSep -minSep)
        labelHI["Stripper Level"]       = (labelHI["Stripper Level"] - minStrip)/(maxStrip -minStrip)
        return labelHI
if __name__ == '__main__':

    with open('utility/modelConfig.json', 'r') as configFile:
        config = json.load(configFile)
    myConfig   = config["tep"]

    preData     = PreProcessData(config=myConfig)
    testData = preData.getTrainData()

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=
    # HANDLE THE DATA WITH ALL CASE
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=
    # rawData             = preData.loadDataFromH5Format(case="all")
    # withRULData         = preData.addRULAndEnvironment(rawData)
    # withoutInitRawData  = preData.removeInitState(inData= withRULData)

    # reacTHR,sepTHR,stripTHR     = preData.getTHRLiquidLabel(withoutInitRawData)
    # print(reacTHR)
    # print(sepTHR)
    # print(stripTHR)
    # a=1
    # # ---------------------------
    # # For data
    # cleanData           = preData.cleanData(rawSignal           =   withoutInitRawData,
    #                                         columnName          =   preData.allWithoutLabel,
    #                                         getMovingAverage    =   False,
    #                                         flipToDown          =   True)
    # # ---------------------------
    # # For label, flip only
    # cleanData           = preData.cleanData(rawSignal           =   cleanData,
    #                                         columnName          =   preData.degraLiq,
    #                                         getMovingAverage    =   False,
    #                                         flipToDown          =   True)
    # # data                = preData.movingToPeak(inData           =   cleanData)

    # norData             = preData.normalizeDataEachModes(cleanData)                      # Normalize the data
    # # _                   = preData.selectSensor( inData          = norData,    # Select the sensor for each component in a specific mode      
    # #                                             case            = 2)

    # # # ---------------------------
    # # # Split train and val before
    # # # getting sentences
    # # train,val           = preData.splitTrajectoryTrainVal(  inData          = norData, 
    # #                                                         case            = 'all',
    # #                                                         trainPercent    = 0.7)

    # # # =-=-=-=-=-=-=-=-=-=-=-=-=-=
    # # # HANDLE THE DATA WITH A SPECIFIC 
    # # # CASE
    # # # =-=-=-=-=-=-=-=-=-=-=-=-=-=
    # # seqXReac            = preData.getTensorData(pandasData  = norData,
    # #                                             component   = "reactor",  
    # #                                             case        = 2,
    # #                                             shuffle     = True)
    
    # # seqXSep            = preData.getTensorData( pandasData  = norData,
    # #                                             component   = "separator",  
    # #                                             case        = 2)
    # # seqXStrip          = preData.getTensorData( pandasData  = norData,
    # #                                             component   = "stripper",  
    # #                                             case        = 2)
    
    
                                            
    # print(seqXReac.shape)
    





