import pandas as pd
import numpy as np
import random
from utility.logger             import aiLogger
from prettytable                import PrettyTable
from sklearn.preprocessing      import StandardScaler, MinMaxScaler
import umap
from utility.matpilotDraw       import MatPlot
import matplotlib.pyplot        as plt

class PreProcessData(object):
    def __init__(self,  
                ):
        
        np.random.seed(1)
        random.seed(1)
        # ------------------------------
        # Configure some variables
        # ------------------------------
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
        
        self.beauManipulatedCr1 = [     'corriente(A)',                     'Potencia(kW)',                 'presion_estanque_aire(bar)', 
                                        'Lubricación_post_filtros(psi)',    'Lubricación_válvula_MPS(psi)', 'Lubricación_entrada_sup(psi)',
                                        'Lubricación_entrada_sup(m3/h)',    'Lubricación_entrada_inf(m3/h)', 'Motor_Fase1(A)',
                                        'Motor_Fase2(A)',                   'Motor_Fase3(A)',                'Motor_Fase1(V)',
                                        'Motor_Fase2(V)',                   'Motor_Fase3(V)'
                                        ]
        self.beauEfficiencyCr1  = self.efficiencyCr1
    
        self.addEfficiencyCr1 = [       'Eficiencia_Rendimiento',           'Z_Score_Int',                  'Z_Score_Ext',                              
                                        'Ratio_Flujo_Aceite',               'Rendimiento_por_Revolucion',   'Temperatura_Diferencia_Media',
                                        'Potencia_Media',                   'Eficiencia_Enfriamiento']
        

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
                                                                         
 

    # ===================================
    # Loading the excell file, take more
    # time to load, so prefer to load by
    # h5 function
    # -----------------------------------
    def dataLoadingOriginalData(self,):
        aiLogger.info("Loading data from files ...")
    
        self.crusher1       = pd.read_excel('data/3200CR001_26-3-2024_fixed.xlsx',engine='openpyxl')
        # self.crusher2       = pd.read_excel('data/3200CR002_26-3-2024_fixed.xlsx',engine='openpyxl')
        # self.conveyor       = pd.read_excel('data/3200CV002_26-3-2024_fixed.xlsx',engine='openpyxl')

        aiLogger.info("Finishing loading data ...")

        print(self.crusher1.columns)
        return self.crusher1
    
    # ===================================
    # Save to H5 file with numeric values from 
    # crusher 1 only, abandon some text column
    # -----------------------------------
    def save2H5Format(self, rawData=None,
                            loadFileToSave = True):
        if loadFileToSave:
            rawData     = pd.read_excel('data/criticalMOFCrusher1.xlsx',engine='openpyxl')
        # savedData  = rawData[self.crusher1Value]
        savedData  = rawData
        aiLogger.info("Saving data to h5 file ...")
        savedData.to_hdf('data/criticalMOFCrusher1.h5', key='crusher1', mode='w')

        aiLogger.info("Finish saving data to h5 file ...")

    # ===================================
    # This function is used to load the h5 file
    # -----------------------------------
    def loadCriticalData(self,type='h5'):
        aiLogger.info("Loading critical MOF data from files ...")
        if type == 'h5':
            self.crusher1 = pd.read_hdf('data/criticalMOFCrusher1.h5', key='crusher1')
        else:
            self.crusher1 = pd.read_excel('data/criticalMOFCrusher1.xlsx',engine='openpyxl')

        aiLogger.info("Finish loading data from files ...")
        # print(self.crusher1.columns)
        return self.crusher1

    # ===================================
    # This func is used to remove some 
    # redundant mode of failure inside the
    # excel file
    # -----------------------------------
    def rmRedundantMOF(self, orginalData):
        df_filtered  = orginalData
        for idxFeature in self.notModeOfFailure:
            df_filtered = df_filtered[df_filtered['Modo de Falla'] != idxFeature]
            aiLogger.info(f"Removing {idxFeature}")


        output_path = 'data/criticalMOFCrusher1.xlsx'  # Specify your output file path
        df_filtered.to_excel(output_path, index=False)
        aiLogger.info("Finish to remove redundant data and save to a file xlsx ...")
        return df_filtered.reset_index()


    # ===================================
    # This func is used to sumarry the data
    # according to time which is inside 
    # the excell file into a concise way
    # -----------------------------------
    def sumarryDataFromFile(self,crusher1):
        mofTable                = PrettyTable()
        mofTable.field_names    = ["No","MOF", "startingTime", "endingTime", "staringRow","endingRow"]

        no                      = 1
        usefulMOF               = []
        mofCol                  = crusher1['Modo de Falla']
        preMOF                  = None
        beginingRow             = 0
        beginingTime            = 0
        markedPoint             = {}

        print(mofCol[0:160])
        lenCrusher1 = len(crusher1)
        for idx in range(lenCrusher1):
            curIdx = idx
            preIdx = idx -1
            if curIdx ==149:
                a=1
            if (mofCol[curIdx] != preMOF) and curIdx!=0:
                # --------------------------
                # Collect the last state
                # --------------------------
                endingTime      = crusher1['tiempo'][preIdx] 
                endingRow       = preIdx
                
                if not pd.isna(preMOF):
                    mofTable.add_row([no,preMOF, beginingTime,endingTime,beginingRow+2,endingRow+2])
                    no +=1
                    
                    if preMOF not in usefulMOF:
                        usefulMOF.append(preMOF)
                        markedPoint[preMOF] = [[beginingRow,endingRow]]

                    else:
                        markedPoint[preMOF].append([beginingRow,endingRow])
                # --------------------------
                # Assign new state
                # --------------------------
                preMOF          = mofCol[curIdx]
                beginingTime    = crusher1['tiempo'][curIdx]
                beginingRow     = curIdx
                
        
        print("All the mode of failure:",usefulMOF)
        print(markedPoint)
        print(mofTable)
        return markedPoint


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
        for idx in self.criticalMOF:
            print(idx)

    # ===================================
    # This function is used to calculate
    # additional efficient variables
    # -----------------------------------
    def additionEfficiencyVar(self,df, window_size=30, use_global_mean=True):
        """
        # adding new feature columns that assess various aspects of operational efficiency and stability.

        Parameters:
            - df (DataFrame): The input DataFrame that contains operational data.
            - window_size (int): The size of the rolling window used for calculating moving averages. This affects how data smoothing is applied, particularly for new features that are based on rolling computations.
            - use_global_mean (bool): A flag to determine whether to fill NaN values with the global mean of the respective columns. This can be helpful in maintaining data integrity when initial rows lack enough data due to the rolling window.

        The function adds the following features to the DataFrame:
            - Efficiency performance, based on the ratio of performance (tons) to power (kW).
            - Z-scores for internal and external vibrations to assess the stability and anomaly in vibrations.
            - Oil flow ratio, which compares upper to lower flow rates, indicative of lubrication system efficiency.
            - Performance per revolution, using a rolling average to smooth the measurement over specified window sizes.
            - Average temperature difference and average power, also smoothed with a rolling window.
            - Cooling efficiency, calculated from the average temperature difference divided by the average power.
            - Optionally fills NaN values with the global mean across columns if specified.
        """
    

        #-----------------------------
        # Add efficiency performance:
        # - Calculates efficiency by dividing performance in tons by power in kW.
        #-----------------------------
        step        = 1
        padSize     = window_size- step         # Step by 1 element so overlap is 29
        print(padSize)
        epsilon     = 1e-10  # A small number to prevent division by zero

        df['Eficiencia_Rendimiento'] = df['rendimiento(ton)'] / (df['Potencia(kW)']+ epsilon)

        #-----------------------------
        # Calculate Z-Scores for internal vibrations:
        # - Normalizes internal vibrations relative to their mean and standard deviation.
        #-----------------------------
        df['Z_Score_Int'] = (df['Piñón_int(mm/s)'] - df['Piñón_int(mm/s)'].mean()) / (df['Piñón_int(mm/s)'].std() + epsilon)

        #-----------------------------
        # Calculate Z-Scores for external vibrations:
        # - Normalizes external vibrations relative to their mean and standard deviation.
        #-----------------------------
        df['Z_Score_Ext'] = (df['Piñón_ext(mm/s)'] - df['Piñón_ext(mm/s)'].mean()) / (df['Piñón_ext(mm/s)'].std() + epsilon)

        #-----------------------------
        # Add oil flow ratio:
        # - Calculates the oil flow ratio by dividing the upper flow by the lower flow.
        #-----------------------------
        df['Ratio_Flujo_Aceite'] = df['Lubricación_entrada_sup(m3/h)'] / (df['Lubricación_entrada_inf(m3/h)'] + epsilon)

        #-----------------------------
        # Calculate performance per revolution using a sliding window:
        # - Smooths the relationship between performance and speed using a moving average.
        #-----------------------------
        df['Rendimiento_por_Revolucion']    = (df['rendimiento(ton)'] / (df['velocidad(rpm)'] + epsilon)).rolling(window=window_size).mean()
        

        #-----------------------------
        # Calculate the average temperature difference using a sliding window:
        # - Smooths the temperature differences between the lubrication input and return.
        #-----------------------------
        df['Temperatura_Diferencia_Media'] = (df['Lubricación_entrada(°C)'] - df['Lubricación_retorno(°C)']).rolling(window=window_size).mean()
        
        #-----------------------------
        # Calculate the average power using a sliding window:
        # - Smooths the consumed power using a moving average.
        #-----------------------------
        df['Potencia_Media']    = df['Potencia(kW)'].rolling(window=window_size).mean()
       

        #-----------------------------
        # Calculate the cooling efficiency:
        # - Divides the average temperature difference by the average power to assess the cooling system's efficiency.
        #-----------------------------
        df['Eficiencia_Enfriamiento'] = df['Temperatura_Diferencia_Media'] / (df['Potencia_Media'] + epsilon)

        # print(df[self.addEfficiencyCr1])
        #-----------------------------
        # Use the global mean to fill NaN if specified to use the global mean.
        #-----------------------------
        if use_global_mean:
            dfSensor = df.select_dtypes('number')
            df.loc[:, dfSensor.columns] = dfSensor.fillna(dfSensor.mean())

            # df[self.addEfficiencyCr1].fillna(df[self.addEfficiencyCr1].mean(), inplace=True)

        print("===============================")
        # print(df[self.addEfficiencyCr1])
        return df
        

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
    # get the healthy state of machine
    # -----------------------------------
    def getHealthyState(self,data):
                    
        aiLogger.info("Getting critical trajectories ...")
        
        # ----------------------------
        # Get the healthy state by 
        # accessing manually the healthy
        # index
        # ----------------------------
        self.healthyExcellPos = {"startingRow":  [18317, 28175,  43739,  64526,  87163,  107316, 115185, 133833, 225442, 280493, 310291], #277099 instead of 280493, but the current values is 0
                                 "endingRow":    [27068, 42727,  63051,  86359,  98506,  115132, 132599, 157742, 249835, 283708, 321436],
                                 "percent":      [0.5,   0.5,    0.5,    0.5,    0.5,    0.5,    0.5,    0.5,    0.3,    0.5,    0.7]}   # should subtract 2

        
        self.healthyData =  pd.DataFrame(columns=self.crusher1Value)
        for idx in range(len(self.healthyExcellPos["startingRow"])):
            lenOfHealthyRow         = int((self.healthyExcellPos["endingRow"][idx] - self.healthyExcellPos["startingRow"][idx]) * self.healthyExcellPos["percent"][idx])

            startingPos             = self.healthyExcellPos["startingRow"][idx] - 2                     # In excell file, the index is higher 2 than reality
            endingPos               = self.healthyExcellPos["startingRow"][idx] - 2 + lenOfHealthyRow

            htEachPeriod            = data[startingPos:endingPos]         
            self.healthyData        = pd.concat([self.healthyData,htEachPeriod],ignore_index=True)
        
        # curData         = rawData[preData.crusher1Value]
        # greaterThan500  = curData["corriente(A)"]== 0
        # indices         = greaterThan500[greaterThan500].index.tolist()
        # print(indices)
        return self.healthyData,self.healthyExcellPos
    
    def getTrajectory(self,data):
        trajectoryIdx = {   "Válvula defectuosa":           [[0,5035], [198148, 198761],[198761,208449]],
                            "Falta de aceite hidráulico":   [[98574,101638]],
                            "Lubricación  deficiente":      [[158678,164501-2],[256591,256651]],
                            "Temperatura fuera de rango":   [[164501-2,168035]],
                            "Bomba hidráulica con fuga":    [[168033,168856], [284796,294328]],
                            "Fuga de aceite":               [[168856,169796], [170335, 180620]],
                            "Falla sistema de hidroset-poste": [[180620,187027], [187027,188517],[188517,188581]],
                            "Mandibulas o revestimiento suelto": [[225441, 249970], [249970, 250188],[250188, 256591]],
                            }
        trajectoryList = {}
        for key,values in trajectoryIdx.items():     # Run per each MOF
            trajectoryList[key] = []
            for idxValue in values:                  # Run per each list of index
                trajectoryList[key].append(data[idxValue[0]: idxValue[1]])
                

        # print(trajectoryList)
        return trajectoryList


    def cleanData(self, data, windowSize = 239):
        output     = data.copy()
        columnName = data.columns
        for cl in columnName:
            output[cl] = self.getMovingAverage(data[cl], windowSize=windowSize)
        
        return output
    
    def getMovingAverage(self,values, windowSize = 119):
        
        kernel      = np.ones(int(windowSize))/float(windowSize)
        padWidth    = int(windowSize/2)                         # Equivalent to the floor(window_size/2) for window_size = 3

        padSig      = np.pad(values, padWidth, mode='symmetric') 
        smoothedSig = np.convolve(padSig, kernel, 'valid')
        return smoothedSig


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

        minValue        = rawData.min(axis = 0)
        maxValue        = rawData.max(axis = 0)
        
        
        norData = (rawData - minValue) / (maxValue - minValue) 

        # print(norData[0].shape)
            
        return norData
    
    # ===================================
    # This function is used to normalize
    # with build-in function
    # -----------------------------------
    def normalizeBuilinFunction(self, data, scaler = MinMaxScaler()):
        variablesToNormalize                                 = data.select_dtypes('number')
        scaledData                                           = scaler.fit_transform(variablesToNormalize)
        self.scaledData                                      = data.copy()
        self.scaledData.loc[:, variablesToNormalize.columns] = scaledData
        return self.scaledData
    

    # ===================================
    # This function is used to split data 
    # into sequences.
    # The input is a list of arrays with 
    # different length
    # reactor: 6sensors
    # sep: 7 sensors
    # strip: 5 sensors
    # -----------------------------------
    def splitAllDataIntoSequences(  self,
                                    data        = None,
                                    seqLength   = 30,
                                    overlap     = 29,):
        aiLogger.info("Spliting data into sequences now ...")

        
        allSeq   = []
        allLabel = []
         
        for idx in range(len(data)):
            transposedDt    = data[idx].transpose()
            # transposedLb    = label[idx].transpose()
            # print(transposedLb)
            seqData,seqLabel    = self.splitAndPadSequences(transposedDt,
                                                            # transposedLb,
                                                            seqLength,
                                                            overlap)     
            
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
    def splitAndPadSequences(self,  data, 
                                    seqLength = 120, 
                                    overlap   = 119):
        sequencesX      = []
        sequencesY      = []
        dataX           = data[0]
        dataY           = data[1]
        step_size       = seqLength - overlap
        totalLength     = len(dataX)
        
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
                aiLogger.error("Check again the for loop!!!")
           
            else:
                sequenceX = dataX[start:end].to_numpy()           # Get sentense
                sequenceY = dataY[start:end].to_numpy()           # Get sentense
                
            sequencesX.append(sequenceX)
            sequencesY.append(sequenceY)

        tensorX =   np.stack(sequencesX, axis=0)
        tensorY =   np.stack(sequencesY, axis=0)
           
        return [tensorX,tensorY]
    

    # ===================================
    # This function is used to split data
    # to train, val, test groups
    # -----------------------------------
    def splitTrainVal(self,inputX,outputY,scaleForTrain = 0.8):

        xLength     =   len(inputX)
        
        trainX      =   inputX[ :int(xLength*scaleForTrain)]
        trainY      =   outputY[:int(xLength*scaleForTrain)]
        
        valX        =   inputX[ int(xLength*scaleForTrain):]
        valY        =   outputY[int(xLength*scaleForTrain):]
        
        return [trainX,trainY],[valX,valY]
    

    
    
if __name__ == '__main__':

    generateH5file  = True
    summaryFile     = True

    plot                = MatPlot()
    preData             = PreProcessData()

    # --------------------------------
    # Generate the data if it lacks 
    # --------------------------------
    if generateH5file is True:
        orginalData     =   preData.dataLoadingOriginalData()
        filterData      =   preData.rmRedundantMOF(orginalData)
        preData.save2H5Format(filterData)

    # --------------------------------
    # Preprocess the data
    # --------------------------------
    rawDataH5           = preData.loadCriticalData()

    if summaryFile is True:
        preData.sumarryDataFromFile(rawDataH5)


    addedVariablesData  = preData.additionEfficiencyVar(rawDataH5)
    normalizedData      = preData.normalizeBuilinFunction(addedVariablesData,  scaler = MinMaxScaler())
    trajectory          = preData.getTrajectory(normalizedData)


    dimReductionModels  =    umap.UMAP( n_components = 2,
                                        n_neighbors  = 15, 
                                        random_state = 40,
                                        min_dist     = 0.5
                                        )
    
    # --------------------------------
    # Get the label and the trajectory
    # --------------------------------
    theTraject          = trajectory["Falla sistema de hidroset-poste"][0]           # Get the trajectory

    
    lengthData          = len(theTraject)
    # if lengthData >= 1440:
    #     sizeEachState   = 1440
    # else:
    #     sizeEachState   = 120
    sizeEachState       = 900 # 15 hours = 900min, or 60 min
    labels              = np.repeat(np.arange(1, (theTraject.shape[0] // sizeEachState) + 1), sizeEachState)
    num_full_groups     = theTraject.shape[0] // sizeEachState
    remainder           = theTraject.shape[0] % sizeEachState
    if (remainder) > 0:
        labels = np.concatenate([labels, np.full(remainder, num_full_groups + 1)])
    print(labels)

    efficiencyTraject   = theTraject[preData.efficiencyCr1 + preData.addEfficiencyCr1 ]

    print(efficiencyTraject)

    embedding           = dimReductionModels.fit_transform(efficiencyTraject)
                                                
    
    # sizeEachState       = 60
    # lengthData          = len(theTraject)
    # labels              = np.repeat(np.arange(1, (theTraject.shape[0] // sizeEachState) + 1), sizeEachState)
    # num_full_groups     = theTraject.shape[0] // sizeEachState
    # remainder           = theTraject.shape[0] % sizeEachState
    # if (remainder) > 0:
    #     labels = np.concatenate([labels, np.full(remainder, num_full_groups + 1)])

    plt.figure(figsize=(10, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c= labels, cmap='Spectral', s=10)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(4)-0.5).set_ticks(np.arange(3))
    plt.title('Supervised UMAP projection of the Iris dataset', fontsize=24)
    plt.show()


    # fig = plt.figure(figsize=(10, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # sc = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=labels, cmap='Spectral', s=50)
    # plt.colorbar(sc, label='Class Label')
    # ax.set_title('3D UMAP Projection', fontsize=24)
    # ax.set_xlabel('Component 1')
    # ax.set_ylabel('Component 2')
    # ax.set_zlabel('Component 3')
    # plt.show()




    # htData, _       = preData.getHealthyState(addedVariables)
    # scaler          = MinMaxScaler()

    # scaledData      = preData.normalizeBuilinFunction(  data   = addedVariables, 
    #                                                     scaler = scaler) 
    # scaledHealthy   = preData.normalizeBuilinFunction(data   = htData, 
    #                                                  scaler = scaler)
    

    # scaledEfficiency = scaledData[preData.efficiencyCr1 + preData.addEfficiencyCr1]
    # dimReductionModels  =    umap.UMAP( n_components = 2,
    #                                             n_neighbors  = 15, 
    #                                             random_state = 40)
    


    # dimReductionModels.fit(scaledEfficiency,scaledData)
    # print(scaledData)

    # print(scaledHealthy)


