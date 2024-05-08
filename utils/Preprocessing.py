#! "C:\Users\CAEX\Desktop\Diego\Crusher Project\2ndrealdatajournalchile\mmodal\Scripts\python.exe"

import numpy  as np
import pandas as pd
from prettytable        import PrettyTable


class PreProcessData(object):
    def __init__(self,  
                ):
        
        # ------------------------------
        # Configure some variables
        # ------------------------------

        # self.machine            = ["crusher1","crusher2","conveyor"]
        # self.rawData            = {"crusher1":[],"crusher2":[],"conveyor":[]}
                
        
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


    # ===================================
    # Loading the excell file, take more
    # time to load, so prefer to load by
    # h5 function
    # -----------------------------------
    def dataLoading(self, typeFile):
    
        if typeFile=='excel':
            self.crusher1      = pd.read_excel('data/criticalMOFCrusher1.xlsx', engine='openpyxl')
            # self.crusher2    = pd.read_excel('data/3200CR002_26-3-2024_fixed.xlsx',engine='openpyxl')
            # self.conveyor    = pd.read_excel('data/3200CV002_26-3-2024_fixed.xlsx',engine='openpyxl')
        elif typeFile=='h5':
            self.crusher1      = pd.read_hdf('data/criticalMOFCrusher1.h5')

        #print(self.crusher1.columns)
        return self.crusher1
    

    # ===================================
    # This func divides the time series into 
    # trajectories between events.
    # -----------------------------------
    def splitTimeSeries(self, filter=True, threshold=1000):
        self.crusher1Raw = self.crusher1.copy()

        # --------------------------------------------
        # 'event' column assigns an ID to all rows that 
        # belong to an event, such as maintenance and 
        # failure modes.
        # --------------------------------------------
        filteredCondition                               = self.crusher1['Modo de Falla'].isin(self.criticalMOF)
        self.crusher1.loc[filteredCondition, 'event']   = self.crusher1[filteredCondition].groupby(['Inicio Reportado', 
                                                                                                  'Fin Reportado']).ngroup().astype('str')

        # --------------------------------------------
        # With 'event' column, the 'trajectory' 
        # column is created which assigns an ID 
        # to all the rows that are between two events
        # --------------------------------------------
        mask                        = self.crusher1['event'].isnull()
        groups                      = mask.ne(mask.shift()).cumsum().astype('str') 
        self.crusher1['trajectory'] = np.where(mask, groups, np.nan)

        # ------------------------------------
        # trajectories that contain fewer 
        # rows than the threshold are excluded
        # ------------------------------------
        if filter:
            count                   = self.crusher1.groupby('trajectory').size().reset_index(name='count')
            self.crusher1           = pd.merge(self.crusher1, count, on='trajectory')
            self.crusher1           = self.crusher1[self.crusher1['count']>threshold] 
            self.crusher1['count']  = self.crusher1['count'].astype('str')

        return self.crusher1


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
        for idx in self.criticalMOF:
            print(idx)


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
    # This function is used to normalize 
    # the data
    # -----------------------------------
    def normalizeData(self, scaler):
        variablesToNormalize                                     = self.crusher1.select_dtypes('number')
        scaledData                                               = scaler.fit_transform(variablesToNormalize)
        self.scaledCrusher1                                      = self.crusher1.copy()
        self.scaledCrusher1.loc[:, variablesToNormalize.columns] = scaledData
        return self.scaledCrusher1
    

    # ===================================
    # This function is used to 
    # get the healthy state of machine
    # -----------------------------------
    def getHealthyStatePrevious(self): #, data):
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

            startingPos             = self.healthyExcellPos["startingRow"][idx] - 2
            endingPos               = self.healthyExcellPos["startingRow"][idx] - 2 + lenOfHealthyRow

            htEachPeriod            = self.crusher1[startingPos:endingPos]         
            self.healthyData        = pd.concat([self.healthyData, htEachPeriod], ignore_index=True)
        
        return self.healthyData, self.healthyExcellPos
    

    def getHealthyState(self):
        self.healthyTrajectories = { '7': 0.5,  '9': 0.5, '11': 0.5, 
                                    '15': 0.5, '23': 0.5, '25': 0.5, 
                                    '27': 0.5, '55': 0.3, '63': 0.5, 
                                    '71': 0.7}

        self.healthyData =  pd.DataFrame(columns=self.crusher1.columns)
        for trajectory in self.healthyTrajectories:
            percentage     = self.healthyTrajectories[trajectory]
            dfTrajectory   = self.crusher1[self.crusher1['trajectory']==trajectory]
            lenOfHealthy   = len(dfTrajectory)

            healthyPeriod  = dfTrajectory[:int(lenOfHealthy*percentage)]
            self.healthyData    = pd.concat([self.healthyData, healthyPeriod], ignore_index=True)
        
        self.crusher1 = self.healthyData
        
    
    # ===================================
    # This function is used to calculate
    # additional efficient variables
    # -----------------------------------
    def additionEfficiencyVar(self, window_size=30, use_global_mean=True):

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
        padSize     = window_size- step     # Step by 1 element so overlap is 29
        epsilon     = 1e-10                 # A small number to prevent division by zero

        df = self.crusher1

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

        self.newEfficiencyCr1 = ['Eficiencia_Rendimiento',      'Z_Score_Int',                  'Z_Score_Ext', 
                                 'Ratio_Flujo_Aceite',          'Rendimiento_por_Revolucion',   'Temperatura_Diferencia_Media',
                                 'Potencia_Media',              'Eficiencia_Enfriamiento'
                                 ]

        #-----------------------------
        # Use the global mean to fill NaN if specified to use the global mean.
        #-----------------------------
        
        if use_global_mean:
            dfSensor = df.select_dtypes('number')
            df.loc[:, dfSensor.columns] = dfSensor.fillna(dfSensor.mean())

        return df
        


if __name__ == '__main__':
    pre             = PreProcessData()
    pre.dataLoading()
