import pandas as pd
import numpy as np
import random
from utility.logger             import aiLogger
from prettytable                import PrettyTable
from sklearn.preprocessing      import StandardScaler, MinMaxScaler
from itertools                  import accumulate

# import umap
from utility.matpilotDraw       import MatPlot
import matplotlib.pyplot        as plt
import json
import os
import package


class PreProcessData(object):
    def __init__(self,  config = None
                ):
        
        np.random.seed(1)
        random.seed(1)
        # ------------------------------
        # Configure some variables
        # ------------------------------
        self.config         = config

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
        
        
        self.addEfficiencyCr1 = ['Eficiencia_Rendimiento',    'Vibracion_piñon_int_norm',  'Vibracion_piñon_ext_norm', 
                                'Desviacion_Vibracion_Int',  'Desviacion_Vibracion_Ext',  'Discrepancia_temperatura_lubricacion']
        self.addedManipulatedCr1 = ['Discrepancia_Corriente_Motor', 'Discrepancia_Voltaje_Motor', 'Sistema Lubricación(A)']


        
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

        self.crusher2ConveyorValue  = [value for value in self.crusher2Value if value != 'Corriente(A)']
        self.conveyorValue          = ['Corriente(A)', 'Pesómetro(ton/h)']
        


        # --------------------------
        # MODES OF FAILURE
        # --------------------------
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
                                    "Sobrecarga, Rebalses y atollos",

                                    "Otra falla eléctrica",                  # only CR2
                                    'Inchancable',
                                    'Filtros saturados',
                                    'Toma de muestra',
                                    'Otra falla control',

                                    'Inspección de equipo ante evento operacional', # only CV2
                                    'Limpiezas (fuera de plan)',
                                    'Detector metales con alarma',
                                    "Falta de Operador",
                                    'Chutes/tolva/tolvines con rotura',
                                    "Cortes de Alimentación"      
                                    ]
        
        # [ 'Sensor temperatura lectura errónea', 
        #  'Falla en sistema hidráulico', 
        #  'Fuga de aceite',  , 
        #  'Correas de transmisión dañadas y/o cortadas', 
        #  'Correas transmisión \xa0con deflexión', 
        #  'Correas\xa0 transmisión con tensión fuera de rango',  ] cr2
        
        # [ 'Polín de retorno en mal estado', 
        #  'Polín de carga en mal estado', , 
        #  'Corte correa con alarma',,
        #     'Desalineamiento con alarma' 


        self.allShutdownType    = ['Válvula defectuosa',                            'Otra falla control',               'Mandibulas o revestimiento suelto',            'Factor Climático',      
                                    'Tronadura',                                     'Stand by',                         'Corte energía interno',                        'Falla en sistema hidráulico',   
                                    'Otra falla eléctrica',                          'Sensor de flujo lectura errónea',  'Puesta en Marcha o sistema en vació',          'Filtros saturados',
                                    'Falta de equipo carguío mina',                 'Fuga de aceite',                   'Cortes de Alimentación',                       'Lubricación \xa0deficiente', 
                                    'Detector metales con alarma',                  'Otra falla instrumentación',       'Sensor temperatura lectura errónea',           'Limpiezas (fuera de plan)', 
                                    'Chutes/tolva/tolvines con rotura',             'Bomba hidráulica con fuga',        'Falta de inventario',                          'Correas\xa0 transmisión con tensión fuera de rango', 
                                    'Inspección de equipo ante evento operacional', 'Falta de Equipos de Apoyo',        'Falla sistema de hidroset-poste',              'Esperas de personal de reparación y/o repuestos y/o equipos de apoyo y/o movimiento de traslado (ins)', 
                                    'Inspección de equipo ante evento operacional (chatarra, falsa alarma etc)',        'Sobrecarga, Rebalses y atollos',               'Esperas de personal de reparación y/o repuestos y/o equipos de apoyo y/o movimiento de traslado (mec)', 
                                    'Bloqueo de Equipos',                           'Correas transmisión \xa0con deflexión', 'Toma de muestra',                         'Cambio de Turno', 
                                    'Colación',                                     'Falta de aceite hidráulico',       'Stock de producto lleno',                      'Falla o problemas aguas abajo', 
                                    'Sensor nivel lectura errónea',                 'Factor Externo (huelgas, manifestaciones, cortes de caminos, etc)',                'Correas de transmisión dañadas y/o cortadas', 'Limpieza equipo', 
                                    'Polín de carga en mal estado',                 'Desalineamiento con alarma',       'Otra falla operacional',                       'Falta de Operador', 
                                    'Polín de retorno en mal estado',               'Inchancable',                      'Puesta en marcha de equipos o sistemas en vacío', 'Corte correa con alarma', 
                                    'Detención por Enclavamiento',                  'Falla o problemas aguas arriba',   'Ajuste de Setting (Asociado al Plan)',         'Otra falla mecánica', 
                                    'Temperatura fuera de rango',                   'Mantención programada']
                                                                         


    # ====================================
    # This function is used to generate 
    # the data which are alight with the
    # input of AI
    # ------------------------------------
    def getTrainDataCrusher(self,):
        package.include()
        outputDir       = os.path.expanduser(self.config["dataDir"])
        cr1, cr2, cv2   = self.loadH5File(dir=outputDir + "data/", name = "InterpolatedWholeConsistentData", keys = ["cr1", "cr2", "cv2"])                           # Load the data from critical file

        # -------------------------------
        # Handle to make the length of 
        # each component consistently
        # cr1 = self.nanFilling(dfFiltered=cr1, validColumn=self.crusher1Value)
        # cr2 = self.nanFilling(dfFiltered=cr2, validColumn=self.crusher2Value)
        # cv2 = self.nanFilling(dfFiltered=cv2, validColumn=self.conveyorValue + self.crusher2ConveyorValue)
        # insertedCr1, insertedCr2,  insertedCv2   = self.fillForTheSameLength(data         = [withHourCr1,withHourCr2,withHourCv2], 
        #                                                                      validColumn  = [self.crusher1Value, self.crusher2Value, self.conveyorValue + self.crusher2ConveyorValue])

        if self.config["splitTrainTest"]:
            target_time = pd.Timestamp('2023-10-01 00:00:00')

            rowCr1 = cr1.index[pd.to_datetime(cr1['tiempo']) == target_time][0]

            rowCr2 = cr2.index[pd.to_datetime(cr2['tiempo']) == target_time][0]

            rowCv2 = cv2.index[pd.to_datetime(cv2['tiempo']) == target_time][0]

            # Display the row
            # print(rowCr1, rowCr2, rowCv2)

            trainCr1 = cr1.iloc[:rowCr1]
            trainCr2 = cr2.iloc[:rowCr2]
            trainCv2 = cv2.iloc[:rowCv2]
        else:
            trainCr1 = cr1
            trainCr2 = cr2
            trainCv2 = cv2

        # --------------------------------
        # Add hour column into the file
        # print(cv2.columns)
        withHourCr1             = self.addHours(trainCr1.copy())
        withHourCr2             = self.addHours(trainCr2.copy())
        withHourCv2             = self.addHours(trainCv2.copy())

        # if self.config["component"]  == "system":
        
        # else:
        #     insertedCr1 = withHourCr1
        #     insertedCr2 = withHourCr2
        #     insertedCv2 = withHourCv2
        # --------------------------------
        # Sumarry data (option)
        # _,mofCr1                = self.sumarryDataFromFile(withHourCr1, printUsefulMOF = True, printMOFTable = True,typeOfTime= "hours")
        # _                       = self.sumarryDataFromFile(withHourCr2, printUsefulMOF = True, printMOFTable = True,typeOfTime= "hours")
        # _                       = self.sumarryDataFromFile(withHourCv2, printUsefulMOF = True, printMOFTable = True,typeOfTime= "hours")

        withIDResults, mofOrder, shutdownMask = self.attachIDPerEvent(df = [withHourCr1,withHourCr2,withHourCv2])
        # withIDCr1, withIDCr2, withIDCv2, IdSys

        print(mofOrder["cr1"], len(mofOrder["cr1"]))
        print(withIDResults["cr1"])

        print("--------------------WITHOUT MAINTENANCE--------------------------")
        print(withIDResults["cr1"]['trajIDWithoutM'])
        print(withIDResults["cr2"]['trajIDWithoutM'])
        print(withIDResults["cv2"]['trajIDWithoutM'])

        print("------------------WITH MAINTENANCE----------------------------")
        print(withIDResults["cr1"]['trajIDWithM'])
        print(withIDResults["cr2"]['trajIDWithM'])
        print(withIDResults["cv2"]['trajIDWithM'])
        # print(withIDResults["sys"])

        handledValues           = self.crusher1Value + self.addEfficiencyCr1 + self.addedManipulatedCr1
        addedData               = self.addition_var(withIDResults["cr1"]) 
        # -----------------------------
        # Normalize
        normalizedCr1           = self.normalizeBuilinFunction(addedData,            headline = handledValues)  
        normalizedCr2           = self.normalizeBuilinFunction(withIDResults["cr2"], headline = self.crusher2Value)  
        normalizedCv2           = self.normalizeBuilinFunction(withIDResults["cv2"], headline = self.conveyorValue + self.crusher2Value)  

        # -----------------------------
        # Add the Cycle and RUL to 
        # each component
        withTimeDataCr1         = self.addCycleAndRul(normalizedCr1, shutdownMask["cr1"])                        # Add the time to the data
        withTimeDataCr2         = self.addCycleAndRul(normalizedCr2, shutdownMask["cr2"])                        # Add the time to the data
        withTimeDataCv2         = self.addCycleAndRul(normalizedCv2, shutdownMask["cv2"])                        # Add the time to the data
        
        # aiLogger.info("Export the file to test!!!")
        # withTimeDataCr1.to_excel("IDCrusher1.xlsx", index=False)

        # -----------------------------
        # Add the cycle and RUL of system
        systemData                      = pd.DataFrame()
        systemData['hours']             = withTimeDataCr1["hours"]
        systemData['trajIDWithoutM']    = withIDResults["sysIDWithoutM"]
        systemData['trajIDWithM']       = withIDResults["sysIDWithM"]
        withTimeDataSys                 = self.addCycleAndRul(systemData)
        
        # -----------------------------
        # Extract before splitting
        timeLabel               = ["tFailureWithMaintenance"]
        dataCr1                 = withTimeDataCr1[ timeLabel + handledValues]
        cycleCr1                = withTimeDataCr1["tFailureWithoutMaintenance"]
        rulCr1                  = withTimeDataCr1["rul"]
        trajectIDCr1            = withTimeDataCr1["trajIDWithoutM"]

        dataCr2                 = withTimeDataCr2[timeLabel + self.crusher2Value]
        cycleCr2                = withTimeDataCr2["tFailureWithoutMaintenance"]
        rulCr2                  = withTimeDataCr2["rul"]
        trajectIDCr2            = withTimeDataCr2["trajIDWithoutM"]

        dataCv2                 = withTimeDataCv2[timeLabel + self.conveyorValue + self.crusher2ConveyorValue]
        cycleCv2                = withTimeDataCv2["tFailureWithoutMaintenance"]
        rulCv2                  = withTimeDataCv2["rul"]
        trajectIDCv2            = withTimeDataCv2["trajIDWithoutM"]

        systemRUL               = withTimeDataSys["rul"]



        # for idx in range(39):
        #     print(f"The ID {idx+1}: {(systemData['trajectoryID'] == (idx+1)).sum()}, MOF: {mofCr1[idx]}")
        # -----------------------------
        # Split with the system ID
        if self.config["component"] == "system":
            seqDataCr1,seqCycleCr1, seqRulCr1,    \
            seqDataCr2,seqCycleCr2 ,seqRulCr2,    \
            seqDataCv2,seqCycleCv2, seqRulCv2,    \
                                    seqRulSys          = self.getTensorData(    dataList   = [  dataCr1, cycleCr1, rulCr1, 
                                                                                                dataCr2, cycleCr2, rulCr2,
                                                                                                dataCv2, cycleCv2, rulCv2,
                                                                                                systemRUL], 
                                                                                trajectId  = withIDResults["sys"],
                                                                                seqLength  = self.config['seqLength'], 
                                                                                overlap    = self.config['seqLength']-1,
                                                                                typeInput  = "listOfPandas")
            
        elif self.config["component"] == "Cr1":
            seqData,seqCycle, seqRul           = self.getTensorData(    dataList   = [dataCr1, cycleCr1, rulCr1 ],                                                                                          
                                                                        trajectId  = trajectIDCr1,
                                                                        seqLength  = self.config['seqLength'], 
                                                                        overlap    = self.config['seqLength']-1,
                                                                        typeInput  = "listOfPandas")
       
        
        elif self.config["component"] == "Cr2":
            seqData,seqCycle, seqRul           = self.getTensorData(    dataList   = [dataCr2, cycleCr2, rulCr2 ],                                                                                          
                                                                        trajectId  = trajectIDCr2,
                                                                        seqLength  = self.config['seqLength'], 
                                                                        overlap    = self.config['seqLength']-1,
                                                                        typeInput  = "listOfPandas")
    
        elif self.config["component"] == "Cv2":
            seqData,seqCycle, seqRul           = self.getTensorData(    dataList   = [dataCv2, cycleCv2, rulCv2],                                                                                          
                                                                        trajectId  = trajectIDCv2,
                                                                        seqLength  = self.config['seqLength'], 
                                                                        overlap    = self.config['seqLength']-1,
                                                                        typeInput  = "listOfPandas")
            
        
        # print(f"The size of the train data: {seqDataCr1.shape} {seqCycleCr1[:,-1,:].shape} {seqRulCr1[:,-1,:].shape}")
        

        if self.config["component"] == "system":
            xTrain      = [seqDataCr1,seqCycleCr1[:,-1,:], seqDataCr2,seqCycleCr2[:,-1,:], seqDataCv2, seqCycleCv2[:,-1,:]]
            yTrain      = seqRulSys[:,-1,:]
            print(f"The size of the train data: {seqDataCr1.shape} {seqCycleCr1[:,-1,:].shape} {seqDataCr2.shape} {seqCycleCr2[:,-1,:].shape} {seqDataCv2.shape} {seqCycleCv2[:,-1,:].shape} {seqRulSys[:,-1,:].shape}")
        else:
            xTrain      = [seqData,seqCycle[:,-1,:]]
            yTrain      = seqRul[:,-1,:]
            print(f"The size of the train data: {seqData.shape} {seqCycle[:,-1,:].shape} {seqRul[:,-1,:].shape}")

        return xTrain, yTrain
    
    # ==================================
    # This function is used to make the
    # values in a list having the same length
    # per time
    def fillForTheSameLength(self, data=[],validColumn = [], option = "zero"):
        """
        Ensures all DataFrames in the input list have the same length by:
        1. Aligning all DataFrames to the union of their time values.
        2. Filling missing 'MOF' values with the right previous value (forward fill).
        3. Filling other missing values with zeros.

        Args:
            data (list): A list of pandas DataFrames. Each must have a 'time', 'MOF', and other columns.

        Returns:
            list: A list of DataFrames with the same length and aligned time values.
        """
        # ---------------------------
        # Add the MOF for the data
        # ---------------------------
        all_times = sorted(set().union(*[df['hours'] for df in data]))          # Step 1: Get the union of all unique time values across all DataFrames
        
        def reindex_and_fill(df, all_times, validColumn):                       # Step 2: Reindex each DataFrame and fill missing values
            df["Modo de Falla"] = df["Modo de Falla"].fillna("good")
            df                  = df.set_index('hours')                         # Set 'time' as the index
            df                  = df.reindex(all_times)                         # Reindex to include all time values
            df['Modo de Falla'] = df['Modo de Falla'].fillna(method='ffill')    # Forward fill 'MOF' column
            df["Modo de Falla"] = df["Modo de Falla"].replace("good", float("nan"))
            # df[validColumn]     = df[validColumn].fillna(0)
            df                  = self.redundantMOFInterpolation(orginalData=df, validColumn=validColumn, mainternanceAsMOF=True)
            
            return df.reset_index()                                             # Reset index to restore 'time'

        # Apply the processing to each DataFrame in the list
        return [reindex_and_fill(df, all_times, validColumn[idx]) for idx,df in enumerate(data)]
    
    # ===================================
    # This function is used to attach the ID
    # for each trajectory between period 
    # of 2 shutdown time, or a period between
    # mainternance and shutdown. 
    def attachIDPerEvent(self, df = [None], summaryData = False):
        '''
        df                      : A list of of MOF column
        results                 : return the same order of pandas df with an addition of trajectoryID, and return with the system ID
        trajIDWithoutM          : consider Maintenance as not critical shutdown
        trajIDWithM             : consider Maintenance as critical shutdown
        '''
        aiLogger.info("Attaching ID into trajectories")
        # --------------------------
        # Assign the critical MOF
        # --------------------------
        compKeys      = ["cr1", "cr2", "cv2", "sys"]
        withIDDf      = {}
        mofOrder      = {}
        sdSystemEvent = pd.DataFrame()
        shutdownMask  = {}

        for compKey,curDf in zip(compKeys[:3], df):
            shutdownMask[compKey]           = {}   
            mofOrder[compKey]               = {}
            machineMOF                      = curDf['Modo de Falla'].dropna().unique()
            elimination                     = [x for x in machineMOF if x in self.notModeOfFailure.copy()] 
            criticalMOFWithMaintenance      = [x for x in machineMOF if x not in elimination]
            criticalMOFWithoutMaintenance   = [x for x in machineMOF if x not in (elimination + ["Mantención programada"])]

            # --------------------------
            # Assign the trajectory ID 
            # per event WITHOUT MAINTENANCE
            # --------------------------
            curDf["isShutdownNM"]   = curDf['Modo de Falla'].isin(criticalMOFWithoutMaintenance)                                                           # Create a mask, True if it's shutdown state, otherwise False         
            curDf['tempIdNM']       = ( (curDf["isShutdownNM"] == False) & (curDf["isShutdownNM"].shift(1, fill_value=False)) |
                                        (curDf["isShutdownNM"]) & (curDf["isShutdownNM"].shift(1, fill_value=False)) &
                                        (curDf['Modo de Falla'] != curDf['Modo de Falla'].shift(1, fill_value=None))
                                        ).astype(int)
                                                                         
            curDf['trajIDWithoutM'] = curDf['tempIdNM'].cumsum()                                                                                           # Using cumsum to accumulate the ID                                                                                   # Using bfill to fill entire ID
            curDf['trajIDWithoutM'] += 1

            # compMOFWithoutM1 = [curDf.iloc[idxCompMOF-1]['Modo de Falla'] for idxCompMOF in range(1, len(curDf)) if curDf.iloc[idxCompMOF]['tempIdNM'] == 1]
            
            compMOFWithoutM  = curDf['Modo de Falla'].shift(1)[curDf['tempIdNM'] == 1].dropna().tolist()

            # compMOFWithoutM         = curDf.loc[curDf['tempIdNM'] == 1, 'Modo de Falla'].shift(1).dropna().tolist()
            # aiLogger.debug(compMOFWithoutM)
            # aiLogger.debug(len(compMOFWithoutM))
            
            # --------------------------
            # Assign the trajectory ID 
            # per event INCLUDING MAINTENANCE
            # --------------------------
            curDf["isShutdownM"]    = curDf['Modo de Falla'].isin(criticalMOFWithMaintenance)                                                            # Create a mask, True if it's shutdown state, otherwise False                                                 # Step 1: Detect shutdown events                        
            curDf['tempIdM']         = ((curDf["isShutdownM"] == False) & (curDf["isShutdownM"].shift(1, fill_value=False)) |
                                        (curDf["isShutdownM"]) & (curDf["isShutdownM"].shift(1, fill_value=False)) &
                                        (curDf['Modo de Falla'] != curDf['Modo de Falla'].shift(1, fill_value=None))
                                        ).astype(int)

            curDf['trajIDWithM']    = curDf['tempIdM'].cumsum()                                                                                  # Using cumsum to accumulate the ID                                                                                   # Using bfill to fill entire ID
            curDf['trajIDWithM']    += 1
            compMOFWithM            = curDf['Modo de Falla'].shift(1)[curDf['tempIdM'] == 1].dropna().tolist()
            
            # aiLogger.debug(compMOFWithM)
            # aiLogger.debug(len(compMOFWithM))
            # --------------------------
            # Extract to handle later
            curDf["isBreak"]                        = curDf['Modo de Falla'].isin(elimination)        
            shutdownMask[compKey]["isBreak"]        = curDf["isBreak"]    
            shutdownMask[compKey]["isShutdownNM"]   = curDf["isShutdownNM"]     # Extract the shutdown mask to cover the shutdown time
            shutdownMask[compKey]["isShutdownM"]    = curDf["isShutdownM"]      # 
            withIDDf[compKey]                       = curDf                     # Extract the dictionary of data for each component (curDf will change in for loop)
            mofOrder[compKey]["compMOFWithM"]       = compMOFWithM              # Extract the order of mode of failure which is used for visualization
            mofOrder[compKey]["compMOFWithoutM"] = compMOFWithoutM
            # --------------------------
            # use OR operator to get the event
            # from each component for system
            # --------------------------
            if compKey == "cr1":    # Initialize the system-wide shutdown event DataFrame
                sdSystemEvent['isShutdownNM']   = curDf["isShutdownNM"]
                sdSystemEvent['isShutdownM']    = curDf["isShutdownM"]
            else:
                sdSystemEvent['isShutdownNM']   |= curDf["isShutdownNM"]
                sdSystemEvent['isShutdownM']    |= curDf["isShutdownM"]
        # -----------------------------
        # Handling the trajectory of system, 
        # ****Check later here about the criteria
        # of maintenance
        # -----------------------------
        sdSystemEvent['tempIdNM']           = [1 if (idx > 0 and sdSystemEvent["isShutdownNM"][idx] == False and sdSystemEvent["isShutdownNM"][idx - 1] == True)
                                                    # (idx > 0 and sdSystemEvent["isShutdownNM"][idx] == True  and sdSystemEvent["isShutdownNM"][idx - 1] == True and sdSystemEvent['Modo de Falla'][idx] != sdSystemEvent['Modo de Falla'][idx-1])
                                                else 0
                                                for idx in range(len(sdSystemEvent))]                                                                     
        sdSystemEvent['trajIDWithoutM']     = sdSystemEvent['tempIdNM'].cumsum()                                                        # Using cumsum to accumulate the ID                                                                                   # Using bfill to fill entire ID
        sdSystemEvent['trajIDWithoutM']     += 1
        withIDDf["sysIDWithoutM"]           = sdSystemEvent['trajIDWithoutM']


        sdSystemEvent['tempId']             = [1 if idx > 0 and sdSystemEvent["isShutdownM"][idx] == False and sdSystemEvent["isShutdownM"][idx - 1] == True 
                                                else 0
                                                for idx in range(len(sdSystemEvent))]
        sdSystemEvent['trajIDWithM']        = sdSystemEvent['tempId'].cumsum()                                                        # Using cumsum to accumulate the ID                                                                                   # Using bfill to fill entire ID
        sdSystemEvent['trajIDWithM']       += 1
        withIDDf["sysIDWithM"]              = sdSystemEvent['trajIDWithM']

        if summaryData: 
            self.sumarryDataFromFile(withIDDf)

        return withIDDf, mofOrder, shutdownMask
    
    # # ===================================
    # # Sumarry data from file
    # # -----------------------------------
    # def sumarryDataFromFile(self,inData, printUsefulMOF = False, printMarkedPoint = False, printMOFTable = False, typeOfTime = "timestep", usefulMOFOnly = True):
    #     '''
    #     inData: the time series data overtime with the MOF column

    #     markedPoint:
    #     mofTable: Content the concise content of the data to print
    #     '''
    #     aiLogger.info("Sumarry the data")
    #     timeCol = "tiempo" if typeOfTime == "timestep" else "hours"
        
    #     mofTable                = PrettyTable()
    #     mofTable.field_names    = ["No","MOF", "startingTime", "endingTime", "staringRow","endingRow"]




    # ===================================
    # This func divides the time series into 
    # trajectories between events.
    # -----------------------------------
    def split_time_series(self,data):
        data.reset_index(drop=True, inplace=True)

        # --------------------------------------------
        # 'event' column assigns an ID to all rows that 
        # belong to an event, such as maintenance and 
        # failure modes.
        # --------------------------------------------
        filtered_condition                      = data['Modo de Falla'].isin(self.criticalMOF)
        data.loc[filtered_condition, 'event']   = data[filtered_condition].groupby(['Inicio Reportado', 
                                                                                    'Fin Reportado']).ngroup().astype('str')

        # --------------------------------------------
        # With 'event' column, the 'trajectory' 
        # column is created which assigns an ID 
        # to all the rows that are between two events
        # --------------------------------------------
        mask                        = data['event'].isnull()
        groups                      = mask.ne(mask.shift()).cumsum().astype('str') 
        data['trajectory'] = np.where(mask, groups, np.nan)
        data['trajectory'] = pd.factorize(data['trajectory'])[0]+1
        data['trajectory'] = data['trajectory'].astype('str')
        data['trajectory'] = data['trajectory'].replace('0', np.nan)
        self.trajectories           = data.trajectory.dropna().unique()
        data.drop(columns='event', inplace=True)

        data = data.dropna(subset=['trajectory'])
        data['trajectory'].astype("int")
        return data
    
    def remove_redundant_data(self, df=None, threshold=200):
        df['tiempo']  = pd.to_datetime(df['tiempo'])


        # Many sensors have zero value during this time period
        df.drop(df[df['tiempo'].between('2023-11-14 20:01:00', '2023-11-17 04:30:00')].index, inplace=True)

        # Trajectories that have a duration less than the threshold are removed
        count = df.groupby('trajectory').size().reset_index(name='count')
        df = pd.merge(df, count, on='trajectory')
        df = df[df['count']>threshold] 
        df['period'] = df['tiempo'].dt.strftime('%d-%m-%Y %H')
        df.drop(columns='count', inplace=True)
        self.trajectories = df.trajectory.dropna().unique() 

        return df
    # ===================================
    # Loading the excell file, take more
    # time to load, so prefer to load by
    # h5 function
    # -----------------------------------
    def dataLoadingOriginalData(self, component = "crusher1"):
        aiLogger.info("Loading data from files ...")

        if component == "crusher1":
            componentData       = pd.read_excel('data/3200CR001_26-3-2024_fixed.xlsx',engine='openpyxl')
        elif component == "crusher2":
            componentData       = pd.read_excel('data/3200CR002_26-3-2024_fixed.xlsx',engine='openpyxl')
        elif component == "conveyorBelt":
            componentData       = pd.read_excel('data/3200CV002_26-3-2024_fixed.xlsx',engine='openpyxl')

        aiLogger.info("Finishing loading data ...")

        print(componentData.columns)
        return componentData
    
    # ===================================
    # This function is used to save an available
    # data into a H5 format
    # -----------------------------------
    def save2H5Format(self, rawData         =   [None],
                            name            =   "criticalMOFCrusher1",
                            dir             =   "data/",
                            keys             =   ["defaultName1"],
                            ):
        '''
        rawData:    it should be a list of the pandas data or numpy
        name:       the name of the one wanting to save
        dir:        directory to save
        key:        if many cases in a same file, use key to denote the difference
        '''
        if rawData[0] is None:
            raise KeyError("The value of rawData should not be None, please put valid values into a list")

        if not os.path.exists(dir):
            os.makedirs(dir)

        for idx, data in enumerate(rawData):
            aiLogger.info("Saving data to h5 file ...")
            writingMode = 'w' if idx==0 else 'a'
            data.to_hdf(dir + name + ".h5", key=keys[idx], mode= writingMode)
            
            aiLogger.info(f"Finish saving data {keys[idx]} to h5 file ...")
    
    # ===================================
    # This function is used to load multiple
    # file or a single file with different 
    # format
    # -----------------------------------
    def loadH5File(self,    dir     = "data/",
                            name    = "default",
                            keys    = [None]
                        ):
        '''
        rawData:    it should be a list of the pandas data or numpy
        name:       the name of the one wanting to save
        dir:        directory to save
        key:        it should be a list of keys, if many cases in a same file, use key to denote the difference
        '''
        if keys[0] is None:
            raise KeyError("The value of key should not be None, please put valid values into a list")

        aiLogger.info("Loading H5 file ...")
        
        returnedData = []
        for idxKey in keys:
            data = pd.read_hdf(dir + name + '.h5', key= idxKey)
            returnedData.append(data)
        aiLogger.info("Finish loading data from files ...")
        # print(self.crusher1.columns)
        return returnedData


    # ===================================
    # This function is used to load the h5 file
    # -----------------------------------
    def loadFile(self,          dir     = "data/",
                                name    = "default",
                                type    ='h5'):
        aiLogger.info(f"Loading the {type} file at the directory {dir + name} ...")
        if type == 'h5':
            data = pd.read_hdf(dir + name + '.' + type)
        else:
            data = pd.read_excel(dir + name + '.' + type,engine='openpyxl')

        aiLogger.info(f"Finish loading the {type} file at the directory {dir + name}.")
        print(data.columns)
        return data
    

    # ===================================
    # This function is used to convert
    # time from ns to hours for caculation
    # -----------------------------------
    def addHours(self,data):
        aiLogger.info("Add hour to data")
        df              = pd.DataFrame(data)
        reference_time  = pd.to_datetime(data['tiempo'].iloc[0])
        data['tiempo']  = pd.to_datetime(df['tiempo'])
        df['hours']     = ((data['tiempo'] - reference_time) / pd.Timedelta(hours=1)).round(3)
        return df
    # ===================================
    # This function is used to add RUL, 
    # failure time and others
    # -----------------------------------
    def addCycleAndRul(self,df, shutdownMask = None):
        df["tFailureWithMaintenance"]       = df.groupby('trajIDWithM',     sort=False)['hours'].transform(lambda x: x - x.min())
        df["tFailureWithoutMaintenance"]    = df.groupby('trajIDWithoutM',  sort=False)['hours'].transform(lambda x: x - x.min())
        # print(df["tFailureWithMaintenance"])
        if shutdownMask is not None:
            # ---------------------------
            # Shutdown mask with maintenance
            shutdownMaskWithM                                       = shutdownMask["isShutdownM"].copy()
            shutdownMaskWithM.loc[shutdownMaskWithM == True]        = np.nan
            shutdownMaskWithM.loc[shutdownMaskWithM == False]       = True

            # ---------------------------
            # Shutdown mask without maintenance
            shutdownMaskWithoutM                                    = shutdownMask["isShutdownNM"].copy()
            shutdownMaskWithoutM.loc[shutdownMaskWithoutM == True]  = np.nan
            shutdownMaskWithoutM.loc[shutdownMaskWithoutM == False] = True

            # ---------------------------
            # Break mask with maintenance
            breakMask                                               = shutdownMask["isBreak"].copy()
            breakMask.loc[breakMask == True]                        = np.nan
            breakMask.loc[breakMask == False]                       = True



            df["tFailureWithMaintenance"]       = df["tFailureWithMaintenance"]     * shutdownMaskWithM
            df["tFailureWithoutMaintenance"]    = df["tFailureWithoutMaintenance"]  * shutdownMaskWithoutM
            df["tBreak"]                        = df["tFailureWithoutMaintenance"]  * breakMask

            df["tFailureWithMaintenance"]       = df["tFailureWithMaintenance"].fillna(method='ffill')
            df["tFailureWithoutMaintenance"]    = df["tFailureWithoutMaintenance"].fillna(method='ffill')
            df["tBreak"]                        = df["tBreak"].fillna(method='ffill')

        df["rul"]                           = df.groupby('trajIDWithoutM',sort=False)['hours'].transform(lambda x: x.max() - x)
        return df
    
    def addFloatTimeOld(self,data):
        df              = pd.DataFrame(data)
        reference_time  = pd.to_datetime(data['tiempo'].iloc[0])
        data['tiempo']  = pd.to_datetime(df['tiempo'])
        df['hours']     = (data['tiempo'] - reference_time) / pd.Timedelta(hours=1)
   
        # df["rul"]       = df.groupby('trajectory',sort=False)['hours'].transform(lambda x: x - x.min())
        return df
    
    # ===================================
    # This function is used to clean the 
    # string values
    # -----------------------------------
    def cleanStringInNumeric(self, data):
        validColumn         = data.columns.difference(['tiempo',          'Descripción Equipo',   'Modo de Falla', 
                                                        'Comentario',       'Especialidad',          'Motivo',
                                                        'Inicio Reportado',  'Fin Reportado',       'Equipo'])
    
        maskNonNumeric = data[validColumn].applymap(lambda x: not isinstance(x, (int, float)))
        
        data[maskNonNumeric] = np.nan                                                           # Set non-numeric values to NaN
        data[validColumn] = data[validColumn].interpolate(method='linear')                      # Interpolate

        # -------------------------------
        # Print the non-numeric positions
        # -------------------------------
        nonNumericIndices = maskNonNumeric.stack()[maskNonNumeric.stack()].index.tolist()
        print(f"Indices with non-numeric values: {nonNumericIndices}")

        return data
    
    # =====================================
    # This function is used to remove non
    # critical MOF
    # -------------------------------------
    def removeNonCriticalMOF(self,df = [], mainternanceAsCriticalFailure = True, printRemovedCol = False, removeAllNoneCritical = False):
        '''
        df                              : it should be a list of component input, if input is multiple components, this function will remove joint non critical MOF only.
                                          If the input is only the 1 element input, it will remove all non criticall component
        removeAllNoneCritical           : true if wantting to remove all the non critical component (both joint and non joint), false if wantting to remove only joint ones
        mainternanceAsCriticalFailure   : remove mainternace or not. 
        '''
        removedColumn   = pd.DataFrame()
        removedLog      = pd.DataFrame(columns=['DataFrame_Index', 'Row_Index', 'Modo de Falla'])
        for idx,curDf in enumerate(df):
            machineMOF      = curDf['Modo de Falla'].dropna().unique()
            # --------------------------
            # Find the eliminated MOF:
            # --------------------------
            if mainternanceAsCriticalFailure:
                elimination     = [x for x in machineMOF if x in self.notModeOfFailure.copy()]
            else:
                elimination     = [x for x in machineMOF if x in self.notModeOfFailure.copy()] + ["Mantención programada"] 
            
            # --------------------------
            # Assign the ID over events
            # --------------------------
            eliminatedRow    = curDf['Modo de Falla'].isin(elimination)  
            if idx == 0:    # Initialize the system-wide shutdown event DataFrame
                removedColumn['isElimined'] = eliminatedRow
            else:
                removedColumn['isElimined'] &= removedColumn['isElimined']

        if printRemovedCol:    
            for df_idx, curDf in enumerate(df):
                removedRows = curDf[removedColumn['isElimined'].values]
                if not removedRows.empty:
                    print(f"DataFrame {df_idx}: Rows to be removed:\n{removedRows[['Modo de Falla']].reset_index()}")
        
        result          = [curDf[~removedColumn['isElimined'].values].reset_index(drop=True) for curDf in df]
        returnedResult  = result if len(result)>1 else result[0]
        return returnedResult

    # ===================================
    # This function is used to interpolate
    # the redundant MOF
    # -----------------------------------
    def redundantMOFInterpolation(self, orginalData, validColumn = None, typeOfPadding = "meanPrevious", padAtEdge = True, replaceMOFWithNan = True, saveToFile = False, typeToSave = 'h5', nameOfSaving = "default", mainternanceAsMOF = True):
        '''
        originalData    : it should be a pandas with continious running time of singal. 
        typeOfPadding   : the type of inference. 
                            "linear":  linear inference, 
                            "zero"  : replace all the sensor with the zeros values
                            "spline": spline inference
                            "meanPrevious": get the average of previous state to fill
                            "symetricPadding": using the data of before and later the event to pad
                            "orginal": do nothing
        padAtEdge       : handling the nan value at edge while the normal one can't handle
        saveToFile      : if want to save the handled data
        '''
        alwaysWorkVariable = []
        aiLogger.info("Handing the redundant Mode of failure")
        dfFiltered  = orginalData
        # for idxFeature in self.notModeOfFailure:
        
        
        if validColumn is None:
            validColumn         = dfFiltered.columns.difference(['tiempo',          'Descripción Equipo',   'Modo de Falla', 
                                                                'Comentario',       'Especialidad',          'Motivo',
                                                                'Inicio Reportado',  'Fin Reportado',       'Equipo'] 
                                                                + alwaysWorkVariable)                              
        maskNonCritial                                  = dfFiltered['Modo de Falla'].isin(self.notModeOfFailure) if mainternanceAsMOF else  dfFiltered['Modo de Falla'].isin(self.notModeOfFailure + ["Mantención programada"])
        dfFiltered.loc[maskNonCritial, validColumn]     = np.nan                                         # Assign the nonCritical to Nan
        # nan_rows_count                                  = dfFiltered[validColumn].isnull().any(axis=1).sum()
        # print("Number of rows with at least one NaN value:", nan_rows_count)

        dfFiltered                                      = self.nanFilling(dfFiltered, validColumn = validColumn, typeOfPadding=typeOfPadding, padAtEdge = padAtEdge)

        if replaceMOFWithNan:
            dfFiltered.loc[maskNonCritial, 'Modo de Falla'] = np.nan
    
        if saveToFile:      # Use for debugging
            aiLogger.info("Saving the cleaned data")
            output_path = 'data/' + nameOfSaving + '.' + typeToSave # Specify your output file path        
            dfFiltered.to_hdf(output_path, mode='w') if  typeToSave=="h5" else dfFiltered.to_excel(output_path, index=False)


        aiLogger.info("Finish to remove redundant data and fill nan values")
        return dfFiltered


    # ===================================
    # This func is used to fill the nan values
    # with options
    # -----------------------------------
    def nanFilling(self,dfFiltered, validColumn = None, typeOfPadding = "meanPrevious", padAtEdge = True):
        aiLogger.info("Filtering the nan values")
        dfFiltered[validColumn]   = dfFiltered[validColumn].apply(pd.to_numeric, errors='coerce')         # Convert all the column to numeric, the exception string will be Nan  
        
        if validColumn is None:
            validColumn = dfFiltered.columns()
        # ---------------------------------
        # Interpolate
        # ---------------------------------
        if typeOfPadding == "linear":
            dfFiltered[validColumn] = dfFiltered[validColumn].interpolate(method='linear')

        elif typeOfPadding == "zero":
            dfFiltered[validColumn] = dfFiltered[validColumn].fillna(0)
        
        elif typeOfPadding == "spline":
            dfFiltered[validColumn] = dfFiltered[validColumn].interpolate(method='spline', order=2)
        
        elif typeOfPadding == "meanPrevious":
            for col in validColumn:
                rolling_mean    = dfFiltered[col].rolling(window=256, min_periods=255, closed='left').mean()        # Create the mean value with if have 255 element -> to get the first mean for nan
                dfFiltered[col] = dfFiltered[col].fillna(rolling_mean)
            # dfFiltered[validColumn] = dfFiltered[validColumn].interpolate(method='linear')
        else:
            dfFiltered = dfFiltered
        # -----------------------------------------
        # Fill any remaining NaNs with forward-fill 
        # or backward-fill as needed (at edge)
        # -----------------------------------------
        if typeOfPadding != "remove" and padAtEdge == True:
            dfFiltered[validColumn] = dfFiltered[validColumn].fillna(method='ffill')
            dfFiltered[validColumn] = dfFiltered[validColumn].fillna(method='bfill')
        
        return dfFiltered

    # ===================================
    # This func is used to sumarry the data
    # according to time which is inside 
    # the excell file into a concise way
    # -----------------------------------
    def sumarryDataFromFile(self,inData, printUsefulMOF = False, printMarkedPoint = False, printMOFTable = False, typeOfTime = "timestep", usefulMOFOnly = True):

        '''
        inData: the time series data overtime with the MOF column

        markedPoint:
        mofTable: Content the concise content of the data to print
        '''
        aiLogger.info("Sumarry the data")
        timeCol = "tiempo" if typeOfTime == "timestep" else "hours"
        
        mofTable                = PrettyTable()
        mofTable.field_names    = ["No","MOF", "startingTime", "endingTime", "staringRow","endingRow"]

        no                      = 1
        usefulMOF               = []
        mofOrder                = []
        mofCol                  = inData['Modo de Falla']
        preMOF                  = None
        beginingRow             = 0
        beginingTime            = 0
        markedPoint             = {}

        # print(mofCol[0:160])
        lenCrusher1 = len(inData)
        for idx in range(lenCrusher1):
            curIdx = idx
            preIdx = idx -1

            # ------------------------------------------
            # If the current state differs the previous 
            # state, record the difference, otherwise,
            # Let it continue
            
            if (mofCol[curIdx] != preMOF) and curIdx!=0:  
                # if usefulMOFOnly:
                #     if mofCol[curIdx] in self.notModeOfFailure:
                #         continue
                # --------------------------
                # Collect the last state
                # --------------------------
                endingTime      = inData[timeCol][preIdx] 
                endingRow       = preIdx
                
                if not pd.isna(preMOF):                                     # If previous one is not nan value, to ensure that we are at the last point of a MOF (MOF to Nah), not the trasition from Nah to MOF                   
                    mofTable.add_row([no,preMOF, beginingTime,endingTime,beginingRow,endingRow])
                    mofOrder.append(preMOF)
                    no +=1
                    
                    if preMOF not in usefulMOF:                             # If we haven't recored the MOF in a "set - python type" of MOF, write it
                        usefulMOF.append(preMOF)    
                        markedPoint[preMOF] = [[beginingRow,endingRow]]     # record the marked point of the beginging and ending point of each MOF   

                    else:
                        markedPoint[preMOF].append([beginingRow,endingRow])
                # --------------------------
                # Assign new state
                # --------------------------
                preMOF          = mofCol[curIdx]
                beginingTime    = inData[timeCol][curIdx]
                beginingRow     = curIdx
        if printMOFTable:
            print(mofTable) 
        if printUsefulMOF:
            print("All the mode of failure:",usefulMOF)
        
        if printMarkedPoint:
            print(markedPoint)
        
        return markedPoint, mofOrder


    # ===================================
    # This function is used to sumarry all
    # the mode of failure inside the data
    # -----------------------------------
    def sumarryMOF(self, data):

        allMofTable                 = PrettyTable()
        allMofTable.field_names     = ["Elimination", "Critical MOF"]

        machineMOF      = data['Modo de Falla'].unique()
        elimination     = [x for x in machineMOF if x in self.notModeOfFailure.copy()]

        # elimination     = self.notModeOfFailure.copy()
        criticalMOF     = [x for x in machineMOF if x not in elimination]

        max_length        = max(len(elimination), len(criticalMOF))

        # -------------------------------
        # Fill the empty ones
        # -------------------------------
        while len(elimination) < max_length:
            elimination.append("")

        while len(criticalMOF) < max_length:
            criticalMOF.append("")

        # -------------------------------
        # Add row and print
        # -------------------------------
        for idx in range(max_length):
            allMofTable.add_row([elimination[idx], criticalMOF[idx]])

        print(allMofTable)
        for idx in self.criticalMOF:
            print(idx)
    
    # ===================================
    # -----------------------------------
    def fillNaNAndRemoveNoiseSpikes(self,signal,threshold=100):

        # Check for NaN values
        nan_positions = signal[signal.isna()].index

        # Print positions of NaN values
        if not nan_positions.empty:
            print("NaN values found at positions:", list(nan_positions))
        else:
            print("No NaN values found in the signal.")    

        signal              = signal.interpolate(method='linear')
        zScores             = (signal - signal.mean()) / signal.std()
        outliers            = signal[np.abs(zScores) > threshold]
        print(signal)
        print(signal.mean())
        print(signal.std())
        # signal[outliers]    = np.nan  # Set outliers to NaN
        # signal              = signal.interpolate()  # Interpolate to fill NaNs

        return signal

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
        # Efficiency performance: eficienciaRendimiento
        # Performance in tons by power in kW.
        #-----------------------------
        step        = 1
        padSize     = window_size- step         # Step by 1 element so overlap is 29
        print(padSize)
        epsilon     = 1e-10  # A small number to prevent division by zero
        
        eficienciaRendimiento = df['rendimiento(ton)'] / (df['Potencia(kW)'] + epsilon)                                  
        df['Eficiencia_Rendimiento'] = self.fillNaNAndRemoveNoiseSpikes(eficienciaRendimiento,threshold=100)


        print(df['Eficiencia_Rendimiento'])
        print(df['rendimiento(ton)'])
        print(df['Potencia(kW)'])
        


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

        df['Eficiencia_Enfriamiento'] = df['Temperatura_Diferencia_Media'] / (df['Potencia_Media'] + epsilon )

        # print(df['Eficiencia_Enfriamiento'])
        # print(df['Temperatura_Diferencia_Media'])
        # print(df['Potencia_Media'])

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
        
    def addition_var(self, df, window_size=60, use_global_mean=True):
        """
        # adding new feature columns that assess various aspects of operational efficiency and stability.

        Parameters:
            - df (DataFrame): The input DataFrame that contains operational data.
            - window_size (int): The size of the rolling window used for calculating moving averages. This affects how data smoothing is applied, particularly for new features that are based on rolling computations.
            - use_global_mean (bool): A flag to determine whether to fill NaN values with the global mean of the respective columns. This can be helpful in maintaining data integrity when initial rows lack enough data due to the rolling window.

        The function adds the following features to the DataFrame:
            - Efficiency performance, based on the ratio of performance (tons) to power (kW).
            - Z-scores for internal and external vibrations to assess the stability and anomaly in vibrations.
            - Performance per revolution, using a rolling average to smooth the measurement over specified window sizes.
            - Average power, smoothed with a rolling window.
            - Standard deviation of internal and external vibrations.
            - Discrepancy of current between motor phases.
            - Standard deviation of air tank pressure.
            - Optionally fills NaN values with the global mean across columns if specified.
        """

        epsilon = 1e-10  # A small number to prevent division by zero
        # df = self.crusher1

        # The current variables of the lubrication system are unified
        df['Sistema Lubricación(A)'] = df['Lubricación_respaldo(A)']
        df.loc[df['Lubricación_principal(A)']!=0, 'Sistema Lubricación(A)'] = df['Lubricación_principal(A)']

        # Calculate the total performance in the rolling window
        df['Total_Rendimiento'] = df['rendimiento(ton)'].rolling(window=window_size, min_periods=1).sum()
        
        # Calculate the energy in the rolling window
        df['Energía_Total'] = df['Potencia(kW)'].rolling(window=window_size, min_periods=1).sum()
        
        # Calculate the efficiency performance
        df['Eficiencia_Rendimiento'] = df['Total_Rendimiento'] / (df['Energía_Total'] + epsilon)
        df.loc[df['Eficiencia_Rendimiento'] > 30, 'Eficiencia_Rendimiento'] = 30 # upper threshold
        df.loc[df['Eficiencia_Rendimiento'] < 0,  'Eficiencia_Rendimiento'] = 0  # lower threshold

        # Normalizes internal and external vibrations of the pinion relative to their mean and standard deviation
        df['Vibracion_piñon_int_norm'] = (df['Piñón_int(mm/s)'] - df['Piñón_int(mm/s)'].mean()) / (df['Piñón_int(mm/s)'].std() + epsilon)
        df['Vibracion_piñon_ext_norm'] = (df['Piñón_ext(mm/s)'] - df['Piñón_ext(mm/s)'].mean()) / (df['Piñón_ext(mm/s)'].std() + epsilon)

        # Calculate the average power using a sliding window: Smooths the consumed power using a moving average.
        df['Potencia_Media'] = df['Potencia(kW)'].rolling(window=window_size, min_periods=1).mean()

        # Calculate the standard deviation of internal and external vibrations
        df['Desviacion_Vibracion_Int'] = df['Piñón_int(mm/s)'].rolling(window=window_size, min_periods=1).std()
        df['Desviacion_Vibracion_Ext'] = df['Piñón_ext(mm/s)'].rolling(window=window_size, min_periods=1).std()

        # Calculate the discrepancy of current and voltage between motor phases
        df['Discrepancia_Corriente_Motor'] = df[['Motor_Fase1(A)', 'Motor_Fase2(A)', 'Motor_Fase3(A)']].std(axis=1).rolling(window=window_size, min_periods=1).mean()
        df['Discrepancia_Voltaje_Motor']   = df[['Motor_Fase1(V)', 'Motor_Fase2(V)']].std(axis=1).rolling(window=window_size, min_periods=1).mean()

        # Calculate the standard deviation of air tank pressure
        df['Desviacion_Presion_Aire'] = df['presion_estanque_aire(bar)'].rolling(window=window_size, min_periods=1).std()

        # Variables Diego
        # Discrepancy between the temperature variables of the lubrication system
        df['Discrepancia_temperatura_lubricacion'] = df[['Lubricación_retorno(°C)','Lubricación_depósito(°C)','Lubricación_entrada(°C)']].std(axis=1).rolling(window=window_size, min_periods=1).mean()   
        
        # Discrepancy between the flow of the lubrication system
        df['Discrepancia_flujo_lubricacion'] = df[['Lubricación_entrada_sup(m3/h)', 'Lubricación_entrada_inf(m3/h)']].std(axis=1).rolling(window=window_size, min_periods=1).mean()

        self.addedEfficiencyCr1     = ['Eficiencia_Rendimiento',    'Vibracion_piñon_int_norm',  'Vibracion_piñon_ext_norm', 
                                   'Desviacion_Vibracion_Int',  'Desviacion_Vibracion_Ext',  'Discrepancia_temperatura_lubricacion',   
                                    #'Potencia_Media',          'Desviacion_Presion_Aire',   'Discrepancia_flujo_lubricacion'
                                    ]
        
        # New manipulable variables
        self.addedManipulatedCr1 = ['Discrepancia_Corriente_Motor', 'Discrepancia_Voltaje_Motor', 'Sistema Lubricación(A)']
        # self.variables_used = self.manipulated_cr1 + self.new_manipulable_cr1 + self.efficiency_cr1 + self.new_efficiency_cr1
        #-----------------------------
        # Use the global mean to fill NaN if specified to use the global mean.
        #-----------------------------
        if use_global_mean:
            try:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            except Exception as e:
                print(f"Error filling NaN values: {e}")
        return df


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
    
    # ===================================
    # This function is used to extract
    # the trajectory in the data with the format
    # of each trajectory is happened in real life
    # without following any format
    # -----------------------------------
    def extractTrajectory(self,data =  None, headline = [], trajecIdx = None, mergeInAList = False):
        '''
        data                    : It is the whole data of a machine per time
        headline                : It should be a list of headline group which need to extract. For example: [['headline1', 'headline2'], [headline3, 'headline4', 'headline5]],
                                    all the headline should be extract in the data column, which make the output have the same length for next steps
        trajecIdx               : It should be a dictionary of mode of failure, include the index of each trajectory
        extractTrajectPerMof    : if this flag is enabled, the data of output will return the same format as trajectIdx, 
                                and including the list of data of each trajectory. Otherwise, it will merge all the
                                trajectories into a list without seperating the trajectory names
        
        returnedData            : return a list of trajectory which is the same order of headline
        '''
        if trajecIdx is None:
            trajecIdx = {   "Válvula defectuosa":           [[0,5035], [198148, 198761],[198761,208449]],
                            "Falta de aceite hidráulico":   [[98574,101638]],
                            "Lubricación  deficiente":      [[158678,164501-2],[256591,256651]],
                            "Temperatura fuera de rango":   [[164501-2,168035]],
                            "Bomba hidráulica con fuga":    [[168033,168856], [284796,294328]],
                            "Fuga de aceite":               [[168856,169796], [170335, 180620]],
                            "Falla sistema de hidroset-poste": [[180620,187027], [187027,188517],[188517,188581]],
                            "Mandibulas o revestimiento suelto": [[225441, 249970], [249970, 250188],[250188, 256591]],
                            }
        returnedData = []
        # -----------------------------
        # Init the values
        # -----------------------------
        for idxHeadline in headline:                    # Run per list of headline
            if not mergeInAList:
                trajectoryList = {}
                for key,values in trajecIdx.items():        # Run per each MOF
                    trajectoryList[key] = []                # Make the same format as trajecIdx
                    for idxValue in values:                 # Run per each list of index
                        trajectoryList[key].append(data[idxValue[0]: idxValue[1]][idxHeadline])
            else:
                 trajectoryList = []
                 for key,values in trajecIdx.items():       # Run per each MOF
                    for idxValue in values:                 # Run per each list of index
                        trajectoryList.append(data[idxValue[0]: idxValue[1]][idxHeadline])
            returnedData.append(trajectoryList)
        return returnedData

    # ===================================
    # This function is used to split data
    # into each sequences and return the 
    # tensor data for each component
    # -----------------------------------
    def getTensorData(self, dataList        = [None], 
                            trajectId       = None, 
                            getAtIdx        = None,
                            shuffle         = False,
                            getLastOnly     = False,
                            seqLength       = 128,
                            overlap         = 127,
                            typeInput       = 'listOfPandas'):
        '''
        dataList(*)     :   a list of pandas with multiple trajectory which is identified through IDs. Each should be the length of rowxcol (row: IDxrowInEachTrajectory, col: features),
                            Each element should be the same 1st Dimention-shape(0), for example [data1, data2, data3] with the length of (221918, 60) (221918, 6) (221918,)
                            and it must be a list of numpy or pandas consistenly  
                            OR
                            a list of list of pandas. Each sub list is a trajectory, but it should have the same length between parent list. For example [data1, data2, data3], in which 
                            the shape inside is [[(1000,5),(965,5), (1003,5)], [(1000,3), (965,3), (1003,3)], [(1000,7), (965,7), (1003,7)]]. If we use this one, a trajectID will be generated
                            automaticly
        trajectId(*)    :   the matrix to have the same row with pandasData aiming to identify the which row belong to which Id

        getAtIdx        :   Access a specific Idx
        shuffle         :   Shuffle the order of windows or not
        getLastOnly     :   Only get the last element in each window
        typeInput       :   it should be a list of pandas (listOfPandas) or a list of list of pandas (listOfList)

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
        # Attach the ID to each 
        # trajectory the dataList is 
        # a list of traject
        # -----------------------
        # lengthTotal = 0
        # for idx in dataList[0]:
        #     lengthTotal += idx.shape[0]
        if typeInput == "listOfList"  and trajectId is None:
            trajectId, dataList = self.generateTrajectoryId(dataList)

        # print(trajectId)
        # print(len(trajectId),len(dataList) )

        # print(trajectId)
        # print(dataList[0].shape)
        # -----------------------
        # Check if getAtIdx out
        # of range
        # -----------------------
        # if getAtIdx is not None and getAtIdx not in trajectId:
        #     print("Test again the getAtIdx value, it's out of range!!!")

        pandasDataList  = [pd.DataFrame(array) for array in dataList] if isinstance(dataList[0], np.ndarray) else dataList      # Convert to pandas if numpy         
        pandasData      = [value.to_frame() if value.ndim == 1 else value for value in pandasDataList]                          # Add new dimension if pandas has 1 dimension only

        storedDimIdx2   = [value.shape[1] for value in pandasData]                                                              # Get the size of each entry to extract later
        storedDimIdx2   = list(accumulate(storedDimIdx2))
        storedDimIdx1   = [0] + storedDimIdx2[:-1]
        # mergedAll       = np.concatenate(pandasData, axis=1)                                                                  # Merge in the same numpy to suffle and extract train and val, stack with col together because we have the same row
        mergedAll       = pd.concat(pandasData, axis=1)

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
        # numOfTraj = int(trajectId.max().iloc[0])
        numOfTraj = int(trajectId.max())
        if getAtIdx is None:
            for idxId in range(numOfTraj):                                                                  # Get each ID
                trajectory          = mergedAll[trajectId.values == (idxId + 1)]                                   # Extract tracjectory from ID
                # print(trajectory.shape)
                componentTrajectory = trajectory.reset_index(drop = True)                                   # Reset the index of pandas                          

                if len(componentTrajectory) < seqLength:
                    continue
                    # raise ValueError(f"The lenghth of trajectory {idxId + 1} should be larger than the seqLength.")
                sequences           = self.splitAndPadSequences(data                = componentTrajectory, 
                                                                seqLength           = seqLength, 
                                                                overlap             = overlap,
                                                                getLastElementOnly  = getLastOnly)           # Split the each trajectory into each seq
                # print(len(sequences))
                allSeq      = allSeq     + sequences                     # trajectories

        # ---------------------------
        # Get the tensor at a specific
        # index
        # ---------------------------
        else:
            trajectory          = mergedAll[trajectId.values == getAtIdx]                           # Extract tracjectory from ID
            componentTrajectory = trajectory.reset_index(drop = True)     # Get the only sensors for each component
            
            sequences           = self.splitAndPadSequences(data                = componentTrajectory, 
                                                            seqLength           = seqLength, 
                                                            overlap             = overlap,
                                                            getLastElementOnly  = getLastOnly)           # Split the each trajectory into each seq
            
            allSeq      = allSeq     + sequences                     # trajectories

        # -------------------
        # Convert the list 
        # to tensor with np
        # -------------------
        tensor =   np.stack(allSeq, axis=0)   

        if shuffle:
            np.random.shuffle(tensor)

        # -------------------
        # Extract these sensors
        # -------------------
        tensorList    = [tensor[:,:,idx1:idx2]   for (idx1,idx2) in zip(storedDimIdx1,storedDimIdx2)]
        return tensorList


    # -------------------------------------
    # This is the sub function of the getTensor()
    # which is used to generate ID for each list elements
    # -------------------------------------
    def generateTrajectoryId(self,listOfData):
        withIdList          = []
        idPandas            = pd.DataFrame()
        
        # -----------------------
        # Generate Id for data
        # -----------------------
        for curId, idxSubList in enumerate(listOfData[0]):             # [[(1000,5),(965,5), (1003,5)], [(1000,3), (965,3), (1003,3)], [(1000,7), (965,7), (1003,7)]]
            curLength   = idxSubList.shape[0]
            # mtId        = np.full((curLength, 1), curId +1)
            mtId        = pd.Series([float(curId + 1)] * curLength)
            idPandas    = pd.concat([idPandas, mtId], ignore_index=True)

        withIdList = [pd.concat(curData, ignore_index=True) for curData in listOfData]          # Generate the data in continious way
        return idPandas, withIdList

    # ===================================
    # This function is used to extract the
    # trajectory from the data
    # -----------------------------------
    def getTrajectory(self,data, getValueOnly = False, headline = None):

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
                if getValueOnly:
                    trajectoryList[key].append(data[idxValue[0]: idxValue[1]][headline])
                else:
                    trajectoryList[key].append(data[idxValue[0]: idxValue[1]])

        # print(trajectoryList)
        return trajectoryList
    
    # ===================================
    # This function is used to 
    # get the healthy state of machine
    # -----------------------------------
    def splitTrainValTraject(self,effTrajectories, maniTrajectories,cycleTrajectories):
        valEffData      = []
        valManiData     = []
        valCycleData    = []
        trainEff        = []
        trainMani       = []
        trainCycle      = []
        for key,value in effTrajectories.items():
            if len(value) >= 2:
                valEffData      = valEffData    + effTrajectories[key][-1:]
                valManiData     = valManiData   + maniTrajectories[key][-1:]
                valCycleData    = valCycleData  + cycleTrajectories[key][-1:]
                trainEff        = trainEff      + effTrajectories[key][:-1]
                trainMani       = trainMani     + maniTrajectories[key][:-1]
                trainCycle      = trainCycle    + cycleTrajectories[key][:-1]

                
        return  trainEff, trainMani, trainCycle, valEffData, valManiData, valCycleData

    def cleanDataTest(self,data,headline=None):
        data.dropna(subset=headline, inplace=True)
        data.drop_duplicates(subset='tiempo', inplace=True)

        return data

    def cleanData(self, data, headline = None):
        # data    = data.dropna(subset=headline, inplace=True)
        # data    = data.drop_duplicates(subset='tiempo', inplace=True)
        data[headline].interpolate(method='linear')

        
        return data
    
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
    def normalizeBuilinFunction(self, data, scaler = MinMaxScaler(), headline = None):
        aiLogger.info("Normalizing the data ....")
        handledData             = data[headline]
        scaledData              = scaler.fit_transform(handledData)
        # self.scaledData.loc[:, handledData.columns] = scaledData
        data[headline]          = scaledData
        return data
    

    # ===================================
    # This function is used to split data 
    # into sequences.
    # The input is 
    # reactor: 6sensors
    # sep: 7 sensors
    # strip: 5 sensors
    # -----------------------------------
    def splitAllDataIntoSequences(  self,
                                    data        = None,
                                    seqLength   = 1024,
                                    overlap     = 1023):
        
        '''
        data: a list of arrays with  different length, each arrays should have the size of (lengthOfData, featureDim)
        
        
        '''
        aiLogger.info("Spliting data into sequences now ...")

        allSeq   = []
        allLabel = []
         
        for idx in range(len(data)):
            # transposedDt        = data[idx].transpose()
            # print(transposedDt.shape)
            seqData,seqLabel    = self.splitAndPadSequences(data[idx],
                                                            # transposedLb,
                                                            seqLength,
                                                            overlap,
                                                            useLabel = False)     
            
            # print((seqData))
            allSeq          = allSeq    + seqData
            allLabel        = allLabel  + seqLabel

    
        tensorX =   np.stack(allSeq, axis=0)         
        tensorY =   np.stack(allLabel, axis=0)     

        # tensorX   = tensorX.transpose(0, 2, 1)        # resize to num_seq, lengh_seq, and num features
             
        return tensorX,tensorY
    
    # ===================================
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

    generateH5file  = False
    summaryFile     = False

    with open('utility/modelConfig.json', 'r') as configFile:
        config = json.load(configFile)
    myConfig   = config["system"]

    plot                = MatPlot()
    preData             = PreProcessData(config=myConfig)
    data                = preData.getTrainData()
    # print(train[0].shape, train[1].shape,train[2].shape)








    # # --------------------------------
    # # Generate the data if it lacks 
    # # --------------------------------
    # if generateH5file is True:
    #     orginalData     =   preData.dataLoadingOriginalData()
    #     filterData      =   preData.rmRedundantMOF(orginalData)
    #     preData.save2H5Format(filterData)

    # # --------------------------------
    # # Preprocess the data
    # # --------------------------------
    # rawDataH5           = preData.loadCriticalData()

    # if summaryFile is True:
    #     preData.sumarryDataFromFile(rawDataH5)


    # addedVariablesData  = preData.additionEfficiencyVar(rawDataH5)

    # plot.draw1D(addedVariablesData["Eficiencia_Rendimiento"])

    # # print(addedVariablesData["Eficiencia_Enfriamiento"])

    # # print(addedVariablesData["Temperatura_Diferencia_Media"])

    # # print(addedVariablesData["Potencia_Media"])



    # if 0:
    #     normalizedData      = preData.normalizeBuilinFunction(addedVariablesData,  scaler = MinMaxScaler())
    #     trajectory          = preData.getTrajectory(normalizedData)


        # dimReductionModels  =    umap.UMAP( n_components = 2,
        #                                     n_neighbors  = 15, 
        #                                     random_state = 40,
        #                                     min_dist     = 0.5
        #                                     )
        
        # # --------------------------------
        # # Get the label and the trajectory
        # # --------------------------------
        # theTraject          = trajectory["Falla sistema de hidroset-poste"][0]           # Get the trajectory

        
        # lengthData          = len(theTraject)
        # # if lengthData >= 1440:
        # #     sizeEachState   = 1440
        # # else:
        # #     sizeEachState   = 120
        # sizeEachState       = 900 # 15 hours = 900min, or 60 min
        # labels              = np.repeat(np.arange(1, (theTraject.shape[0] // sizeEachState) + 1), sizeEachState)
        # num_full_groups     = theTraject.shape[0] // sizeEachState
        # remainder           = theTraject.shape[0] % sizeEachState
        # if (remainder) > 0:
        #     labels = np.concatenate([labels, np.full(remainder, num_full_groups + 1)])
        # print(labels)

        # efficiencyTraject   = theTraject[preData.efficiencyCr1 + preData.addEfficiencyCr1 ]

        # print(efficiencyTraject)

        # embedding           = dimReductionModels.fit_transform(efficiencyTraject)
                                                    
        

        # plt.figure(figsize=(10, 6))
        # plt.scatter(embedding[:, 0], embedding[:, 1], c= labels, cmap='Spectral', s=10)
        # plt.gca().set_aspect('equal', 'datalim')
        # plt.colorbar(boundaries=np.arange(4)-0.5).set_ticks(np.arange(3))
        # plt.title('Supervised UMAP projection of the Iris dataset', fontsize=24)
        # plt.show()





