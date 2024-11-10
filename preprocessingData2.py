#! "C:\Users\CAEX\Desktop\Diego\Crusher Project\2ndrealdatajournalchile\mmodal\Scripts\python.exe"

import warnings
import numpy                    as np
import pandas                   as pd

from prettytable                import PrettyTable
from sklearn.preprocessing      import StandardScaler, MinMaxScaler
from itertools                  import accumulate



class PreProcessData(object):
    def __init__(self,  
                ):
        
        # ------------------------------
        # Configure some self.crusher1_variables
        # ------------------------------

        # self.machine            = ["crusher1","crusher2","conveyor"]
        self.raw_data            = {"crusher1":[]} #,"crusher2":[],"conveyor":[]}
                
        
        # --------------------------
        # CRUSHER 1
        # --------------------------
        self.crusher1_headline    = [   'tiempo',                           'Descripción Equipo',           'Modo de Falla', 
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
        
        self.crusher1_variables   = [   'corriente(A)', 
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
        

        self.manipulated_cr1=[
            'corriente(A)',
            'nivel_tolva_compensacion(%)',
            'Motor_Fase1(A)',
            'Motor_Fase2(A)',
            'Motor_Fase3(A)',
            'Motor_Fase1(V)',
            'Motor_Fase2(V)',
            'Lubricación_principal(A)',
            'Lubricación_respaldo(A)', 
            ]


        self.efficiency_cr1 = [
            'posicion_poste(%)',
            'Piñón_ext(°C)',
            'Piñón_int(°C)',
            'Lubricación_depósito(%)',
            'Lubricación_entrada_sup(psi)',
            'Lubricación_entrada_inf(psi)',
            'Lubricación_principal(psi)',
            'presion_estanque_aire(bar)'
            ]
        self.addEfficiencyCr1 = ['Eficiencia_Rendimiento',    'Vibracion_piñon_int_norm',  'Vibracion_piñon_ext_norm', 
                                'Desviacion_Vibracion_Int',  'Desviacion_Vibracion_Ext',  'Discrepancia_temperatura_lubricacion']
        self.addedManipulatedCr1 = ['Discrepancia_Corriente_Motor', 'Discrepancia_Voltaje_Motor', 'Sistema Lubricación(A)']


        # --------------------------
        # MODES OF FAILURE
        # --------------------------
        self.failure_modes      = [ 'Válvula defectuosa', 
                                    'Lubricación  deficiente',
                                    'Bomba hidráulica con fuga',
                                    'Fuga de aceite',
                                    'Falla sistema de hidroset-poste',                     
                                    'Mandibulas o revestimiento suelto', 
                                  ] 
        
        self.machine_stops      = [ "Válvula defectuosa",
                                    "Mantención programada", #
                                    "Falta de aceite hidráulico", #
                                    "Lubricación  deficiente",
                                    "Temperatura fuera de rango", #
                                    "Bomba hidráulica con fuga",
                                    "Fuga de aceite",
                                    "Falla en sistema hidráulico", #
                                    "Falla sistema de hidroset-poste",
                                    "Mandibulas o revestimiento suelto"
                                   ]
        
        self.not_failure_modes  = [ "Stand by",
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
    def data_loading(self, type_file):
        if type_file=='excel':
            self.crusher1      = pd.read_excel('data/criticalMOFCrusher1.xlsx', engine='openpyxl')
        elif type_file=='h5':
            self.crusher1      = pd.read_hdf('data/criticalMOFCrusher1.h5')

        self.raw_data['crusher1']       = self.crusher1.copy()
        return self.crusher1

    # Remove nan and suplicates rows
    def clear_data(self):
        self.crusher1.dropna(subset=self.crusher1_variables, inplace=True)
        self.crusher1.drop_duplicates(subset='tiempo', inplace=True)

    

    # ===================================
    # This func divides the time series into 
    # trajectories between events.
    # -----------------------------------
    def split_time_series(self):
        self.crusher1.reset_index(drop=True, inplace=True)

        # --------------------------------------------
        # 'event' column assigns an ID to all rows that 
        # belong to an event, such as maintenance and 
        # failure modes.
        # --------------------------------------------
        filtered_condition                               = self.crusher1['Modo de Falla'].isin(self.machine_stops)
        self.crusher1.loc[filtered_condition, 'event']   = self.crusher1[filtered_condition].groupby(['Inicio Reportado', 
                                                                                                      'Fin Reportado']).ngroup().astype('str')

        # --------------------------------------------
        # With 'event' column, the 'trajectory' 
        # column is created which assigns an ID 
        # to all the rows that are between two events
        # --------------------------------------------
        mask                        = self.crusher1['event'].isnull()
        groups                      = mask.ne(mask.shift()).cumsum().astype('str') 
        self.crusher1['trajectory'] = np.where(mask, groups, np.nan)
        self.crusher1['trajectory'] = pd.factorize(self.crusher1['trajectory'])[0]+1
        self.crusher1['trajectory'] = self.crusher1['trajectory'].astype('str')
        self.crusher1['trajectory'] = self.crusher1['trajectory'].replace('0', np.nan)
        self.trajectories           = self.crusher1.trajectory.dropna().unique()
        self.crusher1.drop(columns='event', inplace=True)
        return self.crusher1
    

    # ===================================
    # This function is used to calculate
    # additional efficient self.crusher1_variables
    # -----------------------------------
    def addition_var(self, window_size=60, use_global_mean=True):
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
        df = self.crusher1

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

        self.new_efficiency_cr1 = ['Eficiencia_Rendimiento',    'Vibracion_piñon_int_norm',  'Vibracion_piñon_ext_norm', 
                                   'Desviacion_Vibracion_Int',  'Desviacion_Vibracion_Ext',  'Discrepancia_temperatura_lubricacion',   
                                    #'Potencia_Media',          'Desviacion_Presion_Aire',   'Discrepancia_flujo_lubricacion'
                                    ]
        
        # New manipulable variables
        self.new_manipulable_cr1 = ['Discrepancia_Corriente_Motor', 'Discrepancia_Voltaje_Motor', 'Sistema Lubricación(A)']
        self.variables_used = self.manipulated_cr1+self.new_manipulable_cr1+self.efficiency_cr1+self.new_efficiency_cr1

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
        

    # ===========================================================================
    # Calculate the moving average on the numerical variables of the crusher data
    # ---------------------------------------------------------------------------
    def moving_average(self, window_size=60):
        self.crusher1_avg = self.crusher1.copy()
        numeric_columns   = self.crusher1_avg.select_dtypes('number').columns
        self.crusher1_avg.loc[:, numeric_columns] = self.crusher1_avg.loc[:, numeric_columns].rolling(window        = window_size,
                                                                                                      center        = True,
                                                                                                      min_periods   = 1).mean()
        return self.crusher1_avg
    

    # =======================================================================
    # The average of the aggregation of the numerical variables is calculated
    # -----------------------------------------------------------------------
    def aggregation(self, time_aggregation):
        self.crusher1["tiempo"] = pd.to_datetime(self.crusher1['tiempo'])
        self.crusher1_avg = self.crusher1.set_index('tiempo').select_dtypes('number').resample(time_aggregation).mean()
        self.crusher1_avg.loc[:, 'trajectory'] = self.crusher1.set_index('tiempo').resample(time_aggregation).agg({'trajectory': 'first'})
        self.crusher1_avg = self.crusher1_avg.dropna().reset_index()


    # =========================================================================
    # Auxiliary function used in "feature_engineering": 
    # determines in which percentage range the sum of the derivative is located
    # -------------------------------------------------------------------------
    def exchange_rate(self, value, first_percentile, second_percentile):
        if abs(value) < first_percentile:
            return 0
        elif abs(value) < second_percentile:
            return 1
        else:
            return 2


    # =====================================================
    # Assigns a description of the trend of the time series
    # -----------------------------------------------------
    def event_description(self, derivative_sign, exchange_rate, variable):
        if   derivative_sign==0:
            return f'{variable} constant'
        elif derivative_sign==-1 and exchange_rate==0:
            return f'{variable} oscillating downward'
        elif derivative_sign==1 and exchange_rate==0:
            return f'{variable} oscillating upward'
        elif derivative_sign==-1 and exchange_rate==1:
            return f'{variable} decreasing'
        elif derivative_sign==1 and exchange_rate==1:
            return f'{variable} increasing'
        elif derivative_sign==-1 and exchange_rate==2:
            return f'{variable} decreasing rapidly'
        elif derivative_sign==1 and exchange_rate==2:
            return f'{variable} increasing rapidly'
        else:
            return 'case not found'


    # =======================================================================
    # Descriptions are added to all variables using exchange rate information
    # -----------------------------------------------------------------------
    def feature_engineering(self, window_size=60):
        scaler  = StandardScaler()
        df      = self.crusher1_avg 
        columns = self.crusher1_variables+self.new_efficiency_cr1+self.new_manipulable_cr1
        self.manipulated_description = []
        self.efficiency_description  = []
        for variable in columns:
            for trajectory in self.trajectories:
                mask            = df['trajectory'] == trajectory
                derivative      = df.loc[mask, variable].diff().copy()
                derivative_sum  = derivative.rolling(window=window_size, min_periods=1).sum()
                derivative_sign = np.sign(derivative.rolling(window=window_size, min_periods=1, center=True).median())

                df.loc[mask, f'{variable}_derivative']      = derivative
                df.loc[mask, f'{variable}_derivative_sum']  = derivative_sum
                df.loc[mask, f'{variable}_derivative_sign'] = derivative_sign

            # Labels are created using information from the derivative
            percentile_85 = df[f'{variable}_derivative_sum'].quantile(0.85)
            percentile_95 = df[f'{variable}_derivative_sum'].quantile(0.95)
            
            df.loc[:, f'{variable}_derivative_sum'] = scaler.fit_transform(df[f'{variable}_derivative_sum'].values.reshape(-1,1))
            df.loc[:, f'{variable}_exchange_rate']  = df.loc[:, f'{variable}_derivative_sum'].apply(lambda row: self.exchange_rate(row, percentile_85, percentile_95))  
            
            # Map is 10 times faster than apply from pandas
            df.loc[:, f'{variable}_description']    = np.array(list(map(lambda derivative_sign, exchange_rate: self.event_description(derivative_sign, exchange_rate, variable), df[f'{variable}_derivative_sign'], df[f'{variable}_exchange_rate']))).reshape(-1,1) 

            # The names of the new columns created are saved
            if variable in self.manipulated_cr1+self.new_manipulable_cr1:
                self.manipulated_description.append(f'{variable}_description')
            elif variable in self.efficiency_cr1+self.new_efficiency_cr1:
                self.efficiency_description.append(f'{variable}_description')
        return df
    
    
    # =======================================================
    # Data that will not be useful for analysis is eliminated
    # -------------------------------------------------------
    def remove_redundant_data(self, df=None, threshold=200):
        if df is None:
            df = self.crusher1_avg

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
    # This function is used to 
    # get the healthy state of machine
    # -----------------------------------
    def get_healthy_state(self):
        self.healthyTrajectories = { '1': 0.7,  '3': 0.5,  '7': 0.5,  '9': 0.5, '11': 0.3, '15': 0.5, 
                                    '17': 0.5, '19': 0.5, '21': 0.5, '23': 0.5, '25': 0.5, '27': 0.5,
                                    '29': 0.5, '37': 0.5, '39': 0.5, '45': 0.3, '49': 0.5, '51': 0.5, 
                                    '55': 0.5, '61': 0.5, '63': 0.5, '65': 0.7, '67': 0.5, '71': 0.5}

        healthy_data    =  [] 
        for trajectory in self.trajectories: #self.healthyTrajectories: 
            percentage          = 0.5 #self.healthyTrajectories[trajectory]
            df_trajectory       = self.crusher1[self.crusher1['trajectory']==trajectory]
            len_of_healthy      = len(df_trajectory)

            healthyPeriod       = df_trajectory[:int(len_of_healthy*percentage)]
            healthy_data.append(healthyPeriod)

        self.healthy_data = pd.concat(healthy_data)
        return self.healthy_data


    # ==============================================================
    # Function to normalize the manipulable and efficiency variables
    # --------------------------------------------------------------
    def normalize_data(self, data, scaler):
        variables                               = data[self.crusher1_variables+self.new_efficiency_cr1+self.new_manipulable_cr1] #self.variables_used]
        scaled_data                             = data.copy()
        scaled_data.loc[:, variables.columns]   = scaler.fit_transform(variables) 
        return scaled_data

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

        # -----------------------
        # Attach the ID to each 
        # trajectory the dataList is 
        # a list of traject
        # -----------------------
        # lengthTotal = 0
        # for idx in dataList[0]:
        #     lengthTotal += idx.shape[0]
        # if typeInput == "listOfList"  and trajectId is None:
        #     trajectId, dataList = self.generateTrajectoryId(dataList)

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
                print("ERORRRRRRRRRRRRRRRRRR")
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
    def addFloatTime(self,data):
        df              = pd.DataFrame(data)
        reference_time  = pd.to_datetime(data['tiempo'].iloc[0])
        data['tiempo']  = pd.to_datetime(df['tiempo'])
        df['hours']     = (data['tiempo'] - reference_time) / pd.Timedelta(hours=1)
        return df

    # =====================================================================
    # Function that returns the data preprocessed using the above functions
    # ---------------------------------------------------------------------
    def getTrainData(   self, scaler              = MinMaxScaler(), 
                            window_size         = 6,
                            time_aggregation    = '10min', 
                            threshold           = 200, 
                            trend_info          = False, 
                            type_file           = 'h5'):
        
        self.data_loading(type_file=type_file)
        self.crusher1 = self.addFloatTime(self.crusher1)
        self.clear_data()
        self.addition_var() 
        df = self.split_time_series()
        self.aggregation(time_aggregation=time_aggregation)

        # Textual information is added about the trend of each variable
        if trend_info:
            warnings.filterwarnings("ignore")
            self.feature_engineering(window_size=window_size)
            warnings.filterwarnings("default")
        
        df          = self.remove_redundant_data(threshold = threshold)
        df_healthy  = self.get_healthy_state()

        scaled_data     = self.normalize_data(data   = df,
                                              scaler = scaler) 
        scaled_healthy  = self.normalize_data(data   = df_healthy, 
                                             scaler = scaler)
        

        effTraj                         = scaled_data[self.efficiency_cr1 + self.addEfficiencyCr1]
        maniTraj                        = scaled_data[self.manipulated_cr1 + self.addedManipulatedCr1]
        cycleTraj                       = scaled_data["hours"]
        trajectID                       = scaled_data["trajectory"]
        seqEff,seqMani,seqCycle         = self.getTensorData(dataList   = [effTraj, maniTraj, cycleTraj], 
                                                             trajectId  = trajectID,
                                                             seqLength  = 48, 
                                                             overlap    = 47,
                                                             typeInput  = "listOfList")


        xTrain = [seqEff, seqMani,seqCycle]

        yTrain = xTrain
        return xTrain, yTrain
    

if __name__ == '__main__':
    preData = PreProcessData()
    _,data,healthyData, scaledHeathy = preData.preprocessing()
    print(data["mantenciones"])