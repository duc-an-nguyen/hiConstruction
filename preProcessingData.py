import pandas as pd
import numpy as np
import random
from utility.logger             import aiLogger
from prettytable                import PrettyTable


class PreProcessData(object):
    def __init__(self,  seqLength   = 128,
                        overLap     = 127,
                ):
        
        np.random.seed(1)
        random.seed(1)
        # ------------------------------
        # Configure some variables
        # ------------------------------
        self.seqLength      = seqLength
        self.overlap        = overLap
        self.stabletime     = 30        # hours 
        self.sampleRateTime = 3         # minuites per one sample

        self.machine            = ["crusher1","crusher2","conveyor"]
        self.rawData            = {"crusher1":[],"crusher2":[],"conveyor":[]}
                
        
        # --------------------------
        # CRUSHER 1
        # --------------------------
        self.crusher1Headline    = [    'tiempo',                           'Descripción Equipo',           'Modo de Falla', 
                                        'Comentario',                       'Especialidad',                 'Motivo', 
                                        'Inicio Reportado',                 'Fin Reportado',                'corriente(A)', 
                                        'velocidad(rpm)',                   'nivel_tolva_compensacion(%)',  'Potencia(kW)', 
                                        'posicion_poste(%)',                'rendimiento(ton)',             'presion_estanque_aire(bar)', 
                                        'Piñón_ext(°C)',                    'Piñón_int(°C)',                'Piñón_int(mm/s)', 
                                        'Piñón_ext(mm/s)',                  'Eje_piñón_aceite(%)',          'Lubricación_retorno(°C)', 
                                        'Lubricación_depósito(°C)',         'Lubricación_entrada(°C)',      'Lubricación_principal(psi)',
                                        'Lubricación_principal(A)',         'Lubricación_respaldo(psi)',    'Lubricación_respaldo(A)', 
                                        'Lubricación_post_filtros(psi)',    'Lubricación_válvula_MPS(psi)', 'Lubricación_entrada_sup(psi)',
                                        'Lubricación_entrada_inf(psi)',     'Lubricación_entrada_sup(m3/h)','Lubricación_entrada_inf(m3/h)', 
                                        'Lubricación_depósito(%)',          'Motor_Fase1(A)',               'Motor_Fase2(A)', 
                                        'Motor_Fase3(A)',                   'Motor_Fase1(V)',               'Motor_Fase2(V)', 
                                        'Motor_Fase3(V)',                   'Equipo']
        
        self.crusher1Value    = [    
                                        'corriente(A)', 
                                        'velocidad(rpm)',                   'nivel_tolva_compensacion(%)',  'Potencia(kW)', 
                                        'posicion_poste(%)',                'rendimiento(ton)',             'presion_estanque_aire(bar)', 
                                        'Piñón_ext(°C)',                    'Piñón_int(°C)',                'Piñón_int(mm/s)', 
                                        'Piñón_ext(mm/s)',                  'Eje_piñón_aceite(%)',          'Lubricación_retorno(°C)', 
                                        'Lubricación_depósito(°C)',         'Lubricación_entrada(°C)',      'Lubricación_principal(psi)',
                                        'Lubricación_principal(A)',         'Lubricación_respaldo(psi)',    'Lubricación_respaldo(A)', 
                                        'Lubricación_post_filtros(psi)',    'Lubricación_válvula_MPS(psi)', 'Lubricación_entrada_sup(psi)',
                                        'Lubricación_entrada_inf(psi)',     'Lubricación_entrada_sup(m3/h)','Lubricación_entrada_inf(m3/h)', 
                                        'Lubricación_depósito(%)',          'Motor_Fase1(A)',               'Motor_Fase2(A)', 
                                        'Motor_Fase3(A)',                   'Motor_Fase1(V)',               'Motor_Fase2(V)', 
                                        'Motor_Fase3(V)',                   ]
        
        self.manipulatedCr1 =  [    
                                        'corriente(A)', 
                                        'velocidad(rpm)',                   'Potencia(kW)', 
                                        'posicion_poste(%)',                'presion_estanque_aire(bar)',   'Eje_piñón_aceite(%)',          
                                        
                                        'Lubricación_principal(psi)',
                                        'Lubricación_principal(A)',         'Lubricación_respaldo(psi)',    'Lubricación_respaldo(A)', 
                                        'Lubricación_post_filtros(psi)',    'Lubricación_válvula_MPS(psi)', 'Lubricación_entrada_sup(psi)',
                                        'Lubricación_entrada_inf(psi)',     'Lubricación_entrada_sup(m3/h)','Lubricación_entrada_inf(m3/h)', 
                                        'Lubricación_depósito(%)',          'Motor_Fase1(A)',               'Motor_Fase2(A)', 
                                        'Motor_Fase3(A)',                   'Motor_Fase1(V)',               'Motor_Fase2(V)', 
                                        'Motor_Fase3(V)',                   ]
        
        self.efficiencyCr1 = [          'nivel_tolva_compensacion(%)',      'rendimiento(ton)',             'Piñón_ext(°C)',          
                                        'Piñón_int(°C)',                    'Piñón_int(mm/s)',              'Piñón_ext(mm/s)',
                                        'Lubricación_retorno(°C)',          'Lubricación_depósito(°C)',     'Lubricación_entrada(°C)',]
        

        # --------------------------
        # CRUSHER 2
        # --------------------------                                                 
        self.crusher2Headline  = [      'tiempo',                           'Descripción Equipo',           'Modo de Falla', 
                                        'Comentario',                       'Especialidad',                 'Motivo', 
                                        'Inicio Reportado',                 'Fin Reportado',                'Corriente(A)',
                                        'Nivel_Tolva(%)',                   'Potencia(kW)',                 'Presión_Diferencial(psi)', 
                                        'Renimiento(ton)',                  'Velocidad(rpm)',               'Eje_Piñón_Vibración_1', 
                                        'Eje_Piñón_Vibración_2',            'Eje_Piñón_Vibración_3',        'Eje_Piñón_Vibración_4',
                                        'Enfriamiento_Temp_Aire(°C)',       'Enfriamiento_Corriente_1(A)',  'Enfriamiento_Corriente_2(A)', 
                                        'Enfriamiento_Corriente_3(A)',      'Hidráulico_Corriente_Bomba(A)','Lubricación_Calefactor_Corriente(A)',
                                        'Lubricación_Calefactor_Principal_Corriente(A)',                    'Lubricación_Calefactor_Respaldo_Corriente(A)',
                                        'Lubricación_Flujo_Entrada_Sup(m3/h)',                              'Lubricación_Presión_Entrada_Inf(psi)', 
                                        'Lubricación_Temp_Retorno(°C)',                                     'Lubricación_Temp_Depósito(°C)', 
                                        'Motor_Corriente_F1(A)',            'Motor_Corriente_F2(A)',        'Motor_Corriente_F3(A)', 
                                        'Motor_Voltaje_F1(V)',              'Motor_Voltaje_F2(V)',          'Motor_Voltaje_F3(V)', 
                                        'Motor_Temp_F1(°C)',                'Motor_Temp_F2(°C)',            'Equipo']
        
        self.crusher2Value  = [         'Corriente(A)',
                                        'Nivel_Tolva(%)',                   'Potencia(kW)',                 'Presión_Diferencial(psi)', 
                                        'Renimiento(ton)',                  'Velocidad(rpm)',               'Eje_Piñón_Vibración_1', 
                                        'Eje_Piñón_Vibración_2',            'Eje_Piñón_Vibración_3',        'Eje_Piñón_Vibración_4',
                                        'Enfriamiento_Temp_Aire(°C)',       'Enfriamiento_Corriente_1(A)',  'Enfriamiento_Corriente_2(A)', 
                                        'Enfriamiento_Corriente_3(A)',      'Hidráulico_Corriente_Bomba(A)','Lubricación_Calefactor_Corriente(A)',
                                        'Lubricación_Calefactor_Principal_Corriente(A)',                    'Lubricación_Calefactor_Respaldo_Corriente(A)',
                                        'Lubricación_Flujo_Entrada_Sup(m3/h)',                              'Lubricación_Presión_Entrada_Inf(psi)', 
                                        'Lubricación_Temp_Retorno(°C)',                                     'Lubricación_Temp_Depósito(°C)', 
                                        'Motor_Corriente_F1(A)',            'Motor_Corriente_F2(A)',        'Motor_Corriente_F3(A)', 
                                        'Motor_Voltaje_F1(V)',              'Motor_Voltaje_F2(V)',          'Motor_Voltaje_F3(V)', 
                                        'Motor_Temp_F1(°C)',                'Motor_Temp_F2(°C)',            ]

        # --------------------------
        # CONVEYOR BEALT
        # --------------------------                                                 
        self.conveyorHeadline   = [     'tiempo',                           'Descripción Equipo',           'Modo de Falla', 
                                        'Comentario',                       'Especialidad',                 'Motivo', 
                                        'Inicio Reportado',                 'Fin Reportado',                'Corriente(A)', 
                                        'Pesómetro(ton/h)',                 'Equipo']        


        # --------------------------
        # MODES OF FAILURE
        # --------------------------
        self.usedCriticalMOF      = [   'Válvula defectuosa', 
                                        'Lubricación  deficiente',
                                        'Bomba hidráulica con fuga',
                                        'Fuga de aceite',
                                        'Falla sistema de hidroset-poste',                     
                                        'Mandibulas o revestimiento suelto', 
                                    ] 
        
        self.criticalMOF        = [ "Válvula defectuosa",
                                    "Mantención programada", 
                                    "Falta de aceite hidráulico",
                                    "Lubricación  deficiente",
                                    "Temperatura fuera de rango",
                                    "Bomba hidráulica con fuga",
                                    "Fuga de aceite",
                                    "Falla en sistema hidráulico",
                                    "Falla sistema de hidroset-poste",
                                    "Mandibulas o revestimiento suelto"
                                   ]
        
        self.notModeOfFailure  = [  "Stand by",
                                    "Stock de producto lleno",
                                    "Cambio de Turno",
                                    "Tronadura",
                                    "Falla o problemas aguas arriba",
                                    "Falta de equipo carguío mina",
                                    "Otra falla mecánica",
                                    "Limpieza equipo",
                                    "Detención por Enclavamiento",
                                    "Colación",
                                    "Falla o problemas aguas abajo",
                                    "Ajuste de Setting (Asociado al Plan)",
                                    "Falta de inventario",
                                    "Otra falla instrumentación",
                                    "Falta de Equipos de Apoyo",
                                    "Puesta en Marcha o sistema en vació",
                                    "Bloqueo de Equipos",
                                    "Factor Climático",
                                    "Inspección de equipo ante evento operacional (chatarra, falsa alarma etc)",
                                    "Puesta en marcha de equipos o sistemas en vacío",
                                    "Sensor nivel lectura errónea",
                                    "Corte energía interno",
                                    "Esperas de personal de reparación y/o repuestos y/o equipos de apoyo y/o movimiento de traslado (ins)",
                                    "Esperas de personal de reparación y/o repuestos y/o equipos de apoyo y/o movimiento de traslado (mec)",
                                    "Factor Externo (huelgas, manifestaciones, cortes de caminos, etc)",
                                    "Otra falla operacional",
                                    "Sensor de flujo lectura errónea",
                                    "Sobrecarga, Rebalses y atollos"
                                    ]
                                                                         
                                        
        
    
    def dataLoading(self,):
        aiLogger.info("Loading data from files ...")
    
        self.crusher1       = pd.read_excel('data/criticalMOFCrusher1.xlsx',engine='openpyxl')
        # self.crusher2       = pd.read_excel('data/3200CR002_26-3-2024_fixed.xlsx',engine='openpyxl')
        # self.conveyor       = pd.read_excel('data/3200CV002_26-3-2024_fixed.xlsx',engine='openpyxl')

        aiLogger.info("Finishing loading data ...")

        print(self.crusher1.columns)
        # aiLogger.debug(self.crusher2.columns)
        # aiLogger.debug(self.conveyor.columns)

    # ===================================
    # This func is used to remove some 
    # redundant mode of failure inside the
    # excel file
    # -----------------------------------
    def rmRedundantMOF(self,):
        df_filtered  = self.crusher1
        for idxFeature in self.notModeOfFailure:
            df_filtered = df_filtered[df_filtered['Modo de Falla'] != idxFeature]


        output_path = 'data/criticalMOFCrusher1.xlsx'  # Specify your output file path
        df_filtered.to_excel(output_path, index=False)


    # ===================================
    # This func is used to sumarry the data
    # according to time which is inside 
    # the excell file into a concise way
    # -----------------------------------
    def sumarryDataFromFile(self,):
        mofTable                = PrettyTable()
        mofTable.field_names    = ["No","MOF", "startingTime", "endingTime", "staringRow","endingRow"]

        no                      = 1
        usefulMOF               = []
        mofCol                  = self.crusher1['Modo de Falla']
        preMOF                  = None
        beginingRow             = 0
        beginingTime            = 0


        for idx in range(len(self.crusher1)):
            curIdx = idx
            preIdx = idx -1

            if (mofCol[curIdx] != preMOF) and curIdx!=0:
                # --------------------------
                # Collect the last state
                # --------------------------
                endingTime      = self.crusher1['tiempo'][preIdx] 
                endingRow       = preIdx
                
                if not pd.isna(preMOF):
                    mofTable.add_row([no,preMOF, beginingTime,endingTime,beginingRow+2,endingRow+2])
                    no +=1
                    
                    if preMOF not in usefulMOF:
                        usefulMOF.append(preMOF)
                
                # --------------------------
                # Assign new state
                # --------------------------
                preMOF          = mofCol[curIdx]
                beginingTime    = self.crusher1['tiempo'][curIdx]
                beginingRow     = curIdx
                
        
        print("All the mode of failure:",usefulMOF)
        print(mofTable)


    # ===================================
    # This function is used to sumarry all
    # the mode of failure inside the data
    # -----------------------------------
    def sumarryMOF(self,):
        allMofTable                 = PrettyTable()
        allMofTable.field_names     = ["Elimination", "Critical MOF", "Used MOF"]

        elimination     = self.notModeOfFailure.copy()
        criticalMOF     = self.criticalMOF.copy()
        usedMOF         = self.usedCriticalMOF.copy()


        max_length        = max(len(elimination), len(criticalMOF), len(usedMOF))

        # -------------------------------
        # Fill the empty ones
        # -------------------------------
        while len(elimination) < max_length:
            elimination.append("")

        while len(criticalMOF) < max_length:
            criticalMOF.append("")

        while len(usedMOF) < max_length:
            usedMOF.append("")

        # -------------------------------
        # Add row and print
        # -------------------------------
        for idx in range(max_length):
            allMofTable.add_row([elimination[idx], criticalMOF[idx], usedMOF[idx]])

        print(allMofTable)


    # ===================================
    # This function is used to print the 
    # manipulated and efficiency variable
    # -----------------------------------
    def splitInputandEfficiencyVar(self,):
        table     = PrettyTable()
        table.field_names = ["ManupulatedVar", "EfficiencyVar"]

        manipulatedCr1    = self.manipulatedCr1.copy()
        efficiencyCr1     = self.efficiencyCr1.copy()

        max_length        = max(len(self.manipulatedCr1), len(self.efficiencyCr1))

        # -------------------------------
        # Fill the empty ones
        # -------------------------------
        while len(manipulatedCr1) < max_length:
            manipulatedCr1.append("")

        while len(efficiencyCr1) < max_length:
            efficiencyCr1.append("")

        # -------------------------------
        # Add row and print
        # -------------------------------
        for idx in range(max_length):
            table.add_row([manipulatedCr1[idx], efficiencyCr1[idx]])

        print(table)

    # ===================================
    # This function is used to 
    # extract data from .csv file
    # -----------------------------------
    def getCriticalTrajectory(self,
                    ):
        aiLogger.info("Getting critical trajectories ...")
        
        
        

        

        
        # -------------------------
        # Convert to a matrix, and transpose 
        # to have a shape of 8xlen
        # -------------------------
        # rulLabel        = [np.squeeze(oneDirRulLabel[np.where(idxSimColumn == (idxSimExtract+1)),:] ,axis =0)   for idxSimExtract in range(max(idxSimColumn))]
        # collectedData   = [np.squeeze(rawData[np.where(idxSimColumn == (idxSimExtract+1)),:] ,axis =0)   for idxSimExtract in range(max(idxSimColumn))]
        # time            = [np.squeeze(timeStamp[np.where(idxSimColumn == (idxSimExtract+1)),:] ,axis =0)   for idxSimExtract in range(max(idxSimColumn))]
        # degra           = [np.squeeze(degration[np.where(idxSimColumn == (idxSimExtract+1)),:] ,axis =0)   for idxSimExtract in range(max(idxSimColumn))]



           
        # return criticalTrajectories
    


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
    

    
    
if __name__ == '__main__':
    pre             = PreProcessData()
    pre.dataLoading()
    criticalTrajec  = pre.getCriticalTrajectory()



    # Check the text data inside
    # print(preData.crusher1.head())

    # time        = preData.crusher1['tiempo']

    # for idxLabel in preData.crusher1Value:
    #     currentI    = preData.crusher1[idxLabel].to_numpy().astype(float)
    #     for idx in range(len(currentI)):
    #         # print(type(currentI[idx]))
    #         if type(currentI[idx]) is str:
    #             print(idx)
