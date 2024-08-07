#! "C:\Users\CAEX\Desktop\Diego\Crusher Project\2ndrealdatajournalchile\mmodal\Scripts\python.exe"

import warnings
import numpy                    as np
import pandas                   as pd

from prettytable                import PrettyTable
from deep_translator            import GoogleTranslator
from sklearn.preprocessing      import StandardScaler, MinMaxScaler


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
            'presion_estanque_aire(bar)',
            'Lubricación_depósito(%)',
            'Motor_Fase1(A)',
            'Motor_Fase2(A)',
            'Motor_Fase3(A)',
            'Motor_Fase1(V)',
            'Motor_Fase2(V)',
            ]


        self.efficiency_cr1 = [
            'posicion_poste(%)',
            'Piñón_ext(°C)',
            'Piñón_int(°C)',
            ]


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


    # Function to translate text from spanish to english
    def translate_text(self, column):
        translator = GoogleTranslator(source='es', target='en')

        # Dict with translations
        translations = {}
        unique_texts = self.crusher1[column].dropna().unique()

        # Translate unique values 
        for text in unique_texts:
            translations[text] = translator.translate(text)
            
        return self.crusher1.loc[:, column].map(translations)
    

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


    # =====================================================================
    # Function that returns the data preprocessed using the above functions
    # ---------------------------------------------------------------------
    def preprocessing(self, scaler, window_size, time_aggregation, threshold =0, type_file='h5'):
        self.data_loading(type_file=type_file)
        self.clear_data()
        self.addition_var() 
        df = self.split_time_series()
        self.aggregation(time_aggregation=time_aggregation)

        warnings.filterwarnings("ignore")
        self.feature_engineering(window_size=window_size)
        warnings.filterwarnings("default")
        
        df          = self.remove_redundant_data(threshold=threshold)
        df_healthy  = self.get_healthy_state()

        scaled_data     = self.normalize_data(data   = df,
                                              scaler = scaler) 
        scaled_healthy  = self.normalize_data(data   = df_healthy, 
                                             scaler = scaler)
        
        return df, scaled_data, df_healthy, scaled_healthy
        