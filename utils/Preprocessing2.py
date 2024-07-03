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
            #'corriente(A)',
            #'velocidad(rpm)',
            'nivel_tolva_compensacion(%)',
            #'Potencia(kW)',
            'presion_estanque_aire(bar)',
            #'Lubricación_principal(A)',
            #'Lubricación_respaldo(A)',
            'Sistema Lubricación(A)',
            'Lubricación_depósito(%)',
            'Motor_Fase1(A)',
            'Motor_Fase2(A)',
            'Motor_Fase3(A)',
            'Motor_Fase1(V)',
            'Motor_Fase2(V)',
            #'Motor_Fase3(V)'
            ]


        self.efficiency_cr1 = [
            #rendimiento(ton),
            'posicion_poste(%)',
            'Piñón_ext(°C)',
            'Piñón_int(°C)',
            #'Piñón_int(mm/s)',
            #'Piñón_ext(mm/s)',
            #'Eje_piñón_aceite(%)',

            # 'Lubricación_retorno(°C)',
            # 'Lubricación_depósito(°C)',
            # 'Lubricación_entrada(°C)',

            # 'Lubricación_principal(psi)',
            # #'Lubricación_respaldo(psi)',
            
            # 'Lubricación_post_filtros(psi)',
            # 'Lubricación_válvula_MPS(psi)',
            # 'Lubricación_entrada_sup(psi)',
            # 'Lubricación_entrada_inf(psi)',

            # 'Lubricación_entrada_sup(m3/h)',
            # 'Lubricación_entrada_inf(m3/h)',
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
            self.conveyor_belt = pd.read_excel('data/Datos Correa CV001 2023.xlsx', engine='openpyxl')
        elif type_file=='h5':
            self.crusher1      = pd.read_hdf('data/criticalMOFCrusher1.h5')
            self.conveyor_belt = pd.read_hdf('data/df_belt.h5')

        self.raw_data['crusher1']       = self.crusher1.copy()
        self.raw_data['conveyor_belt']  = self.conveyor_belt.copy()

        return self.crusher1


    def clear_data(self):
        self.crusher1.dropna(subset=self.crusher1_variables, inplace=True)
        self.crusher1.drop_duplicates(subset='tiempo', inplace=True)

        self.conveyor_belt['Pesómetro (t/h)']   = pd.to_numeric(self.conveyor_belt['Pesómetro (t/h)'], errors='coerce')
        self.conveyor_belt['Corriente (A)']     = pd.to_numeric(self.conveyor_belt['Corriente (A)'], errors='coerce')
        self.conveyor_belt.dropna(subset='Pesómetro (t/h)', inplace=True)

        self.crusher1 = self.crusher1.merge(self.conveyor_belt.drop(columns='Corriente (A)'), on='tiempo')


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
        return self.crusher1
    

    # ===================================
    # This function is used to calculate
    # additional efficient self.crusher1_variables
    # -----------------------------------
    def addition_efficiency_var(self, window_size=60, use_global_mean=True):
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


        #-----------------------------
        # Add efficiency performance:
        # - Calculates efficiency by dividing performance in tons by power in kW.
        #-----------------------------
        step        = 1
        padSize     = window_size- step         # Step by 1 element so overlap is 29
        epsilon     = 1e-10  # A small number to prevent division by zero

        df = self.crusher1

        df['Sistema Lubricación(A)'] = df['Lubricación_respaldo(A)']
        df.loc[df['Lubricación_principal(A)']!=0, 'Sistema Lubricación(A)'] = df['Lubricación_principal(A)']

        try:
            # Calculate the total performance in the rolling window
            df['Total_Rendimiento'] = df['rendimiento(ton)'].rolling(window=window_size, min_periods=1).sum()
            
            # Calculate the energy in the rolling window
            df['Energía_Total'] = df['Potencia(kW)'].rolling(window=window_size, min_periods=1).sum()
            
            # Calculate the efficiency performance
            df['Eficiencia_Rendimiento'] = df['Total_Rendimiento'] / (df['Energía_Total'] + epsilon)
            df.loc[df['Eficiencia_Rendimiento'] > 30, 'Eficiencia_Rendimiento'] = 30 # upper threshold
            df.loc[df['Eficiencia_Rendimiento'] < 0,  'Eficiencia_Rendimiento'] = 0  # lower threshold
            
            
            #print("Eficiencia_Rendimiento calculated.")
        except Exception as e:
            print(f"Error calculating Eficiencia_Rendimiento: {e}")
        #-----------------------------
        # Calculate Z-Scores for internal vibrations:
        # - Normalizes internal vibrations relative to their mean and standard deviation.
        #-----------------------------
        try:
            df['Z_Score_Int'] = (df['Piñón_int(mm/s)'] - df['Piñón_int(mm/s)'].mean()) / (df['Piñón_int(mm/s)'].std() + epsilon)
            #print("Z_Score_Int calculated.")
        except Exception as e:
            print(f"Error calculating Z_Score_Int: {e}")
        #-----------------------------
        # Calculate Z-Scores for external vibrations:
        # - Normalizes external vibrations relative to their mean and standard deviation.
        #-----------------------------
        try:
            df['Z_Score_Ext'] = (df['Piñón_ext(mm/s)'] - df['Piñón_ext(mm/s)'].mean()) / (df['Piñón_ext(mm/s)'].std() + epsilon)
            #print("Z_Score_Ext calculated.")
        except Exception as e:
            print(f"Error calculating Z_Score_Ext: {e}")

        #-----------------------------
        # Calculate the average power using a sliding window:
        # - Smooths the consumed power using a moving average.
        #-----------------------------
        try:
            df['Potencia_Media'] = df['Potencia(kW)'].rolling(window=window_size, min_periods=1).mean()
            #print("Average_Power calculated.")
        except Exception as e:
            print(f"Error calculating Potencia_Media: {e}")

        #-----------------------------
        # Calculate the standard deviation of internal vibrations
        #-----------------------------
        try:
            df['Desviacion_Vibracion_Int'] = df['Piñón_int(mm/s)'].rolling(window=window_size, min_periods=1).std()
            #print("Desviacion_Vibracion_Int calculated.")
        except Exception as e:
            print(f"Error calculating Desviacion_Vibracion_Int: {e}")

        #-----------------------------
        # Calculate the standard deviation of external vibrations
        #-----------------------------
        try:
            df['Desviacion_Vibracion_Ext'] = df['Piñón_ext(mm/s)'].rolling(window=window_size, min_periods=1).std()
            #print("Desviacion_Vibracion_Ext calculated.")
        except Exception as e:
            print(f"Error calculating Desviacion_Vibracion_Ext: {e}")

        #-----------------------------
        # Calculate the discrepancy of current between motor phases
        #-----------------------------
        try:
            df['Discrepancia_Corriente_Motor'] = df[['Motor_Fase1(A)', 'Motor_Fase2(A)', 'Motor_Fase3(A)']].std(axis=1).rolling(window=window_size, min_periods=1).mean()
            df['Discrepancia_Voltaje_Motor']   = df[['Motor_Fase1(V)', 'Motor_Fase2(V)']].std(axis=1).rolling(window=window_size, min_periods=1).mean()
            #print("Discrepancia_Corriente calculated.")
        except Exception as e:
            print(f"Error calculating Discrepancia_Corriente: {e}")

        #-----------------------------
        # Calculate the standard deviation of air tank pressure
        #-----------------------------
        try:
            df['Desviacion_Presion_Aire'] = df['presion_estanque_aire(bar)'].rolling(window=window_size, min_periods=1).std()
            #print("Desviacion_Presion_Aire calculated.")
        except Exception as e:
            print(f"Error calculating Desviacion_Presion_Aire: {e}")

        # Variable Diego
        df['Discrepancia_temperatura_lubricacion']  = df[['Lubricación_retorno(°C)','Lubricación_depósito(°C)','Lubricación_entrada(°C)']].std(axis=1).rolling(window=window_size, min_periods=1).mean()
        df['Discrepancia_presion_lubricacion']      = df[['Lubricación_principal(psi)', 'Lubricación_post_filtros(psi)',
                                                          'Lubricación_válvula_MPS(psi)','Lubricación_entrada_sup(psi)',
                                                          'Lubricación_entrada_inf(psi)']].std(axis=1).rolling(window=window_size, min_periods=1).mean()
        df['Discrepancia_flujo_lubricacion']        = df[['Lubricación_entrada_sup(m3/h)', 'Lubricación_entrada_inf(m3/h)']].std(axis=1).rolling(window=window_size, min_periods=1).mean()

        self.new_efficiency_cr1 = ['Eficiencia_Rendimiento',  'Z_Score_Int',  'Z_Score_Ext', 
                                   'Desviacion_Vibracion_Int',     'Desviacion_Vibracion_Ext',    
                                    #'Potencia_Media',  #'Discrepancia_Corriente',   'Desviacion_Presion_Aire'
                                   'Discrepancia_temperatura_lubricacion', 'Discrepancia_presion_lubricacion', 'Discrepancia_flujo_lubricacion'
                                    ]
        self.new_manipulable_cr1 = ['Discrepancia_Corriente_Motor', 'Discrepancia_Voltaje_Motor']
        
        #-----------------------------
        # Use the global mean to fill NaN if specified to use the global mean.
        #-----------------------------
        if use_global_mean:
            try:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                #print("NaN values filled with the global mean.")
            except Exception as e:
                print(f"Error filling NaN values: {e}")
        # updates the eficiency variables from crusher 1
        return df
        

    def moving_average(self, window_size=60):
        self.crusher1_avg = self.crusher1.copy()
        self.crusher1_avg.loc[:, self.crusher1_variables] = self.crusher1_avg.loc[:, self.crusher1_variables].rolling(window        = window_size,
                                                                                                                      min_periods   = 1,
                                                                                                                      center        = True).mean()
        return self.crusher1_avg
    


    def exchange_rate_labels(self, value, first_percentile, second_percentile):
        if abs(value) < first_percentile:
            return 0
        elif abs(value) < second_percentile:
            return 1
        else:
            return 2
        

    def event_description(self, row, variable):
        if   row[f'{variable}_derivative_sign']==0:
            return f'{variable} constant'
        elif row[f'{variable}_derivative_sign']==-1 and row[f'{variable}_exchange_rate']==0:
            return f'{variable} oscillating downward'
        elif row[f'{variable}_derivative_sign']==1 and row[f'{variable}_exchange_rate']==0:
            return f'{variable} oscillating upward'
        elif row[f'{variable}_derivative_sign']==-1 and row[f'{variable}_exchange_rate']==1:
            return f'{variable} decreasing'
        elif row[f'{variable}_derivative_sign']==1 and row[f'{variable}_exchange_rate']==1:
            return f'{variable} increasing'
        elif row[f'{variable}_derivative_sign']==-1 and row[f'{variable}_exchange_rate']==2:
            return f'{variable} decreasing rapidily'
        elif row[f'{variable}_derivative_sign']==1 and row[f'{variable}_exchange_rate']==2:
            return f'{variable} increasing rapidily'
        else:
            return 'case not found'


    def feature_engineering(self, window_size=60):
        scaler  = StandardScaler()
        df      = self.crusher1_avg #.copy()

        self.manipulated_description = []
        self.efficiency_description  = []
        for variable in self.crusher1_variables+self.new_efficiency_cr1+self.new_manipulable_cr1:
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
            df.loc[:, f'{variable}_exchange_rate']  = df.loc[:, f'{variable}_derivative_sum'].apply(lambda row: self.exchange_rate_labels(row, percentile_85, percentile_95))
            df.loc[:, f'{variable}_description']    = df.loc[:, [f'{variable}_derivative_sign', f'{variable}_exchange_rate']].apply(lambda row: self.event_description(row, variable), axis=1)
            df.loc[:, f'{variable}_cum_group']      = (df[f'{variable}_derivative_sign'] != df[f'{variable}_derivative_sign'].shift()).cumsum().astype('str')

            # Information about the stopping periods is added when the machine is started again
            df.loc[:, f'{variable}_previous_detention'] = None
            for idx, event in df[['Modo de Falla', 'Inicio Reportado', 'Fin Reportado']].drop_duplicates(keep='last')['Modo de Falla'].to_dict().items():
              df.loc[idx+1, f'{variable}_previous_detention'] = event
            df.loc[:, f'{variable}_previous_detention'] = df.groupby(f'{variable}_cum_group')[f'{variable}_previous_detention'].ffill()

            # The names of the new columns created are saved
            if variable in self.manipulated_cr1+self.new_manipulable_cr1:
                self.manipulated_description.append(f'{variable}_description')
            elif variable in self.efficiency_cr1+self.new_efficiency_cr1:
                self.efficiency_description.append(f'{variable}_description')
        return df


    def remove_redundant_data(self, df=None, threshold=2000):
        if df is None:
            df = self.crusher1
        self.crusher1.drop(self.crusher1[self.crusher1['tiempo'].between('2023-11-14 20:01:00',
                                                                   '2023-11-17 04:30:00')].index, inplace=True)
        
        self.crusher1.loc[:, self.crusher1_avg.filter(like='description').columns] = self.crusher1_avg.filter(like='description').iloc[:-1]
        self.crusher1.loc[:, self.crusher1_avg.filter(like='previous_detention').columns] = self.crusher1_avg.filter(like='previous_detention').iloc[:-1]

        for idxFeature in self.not_failure_modes:
            self.crusher1 = self.crusher1[self.crusher1['Modo de Falla'] != idxFeature]

        count                   = self.crusher1.groupby('trajectory').size().reset_index(name='count')
        self.crusher1           = pd.merge(self.crusher1, count, on='trajectory')
        self.crusher1           = self.crusher1[self.crusher1['count']>threshold] 
        self.crusher1['period'] = self.crusher1['tiempo'].dt.strftime('%d-%m-%Y %H')
        self.trajectories       = self.crusher1.trajectory.dropna().unique() 
        self.crusher1.drop(columns='count', inplace=True)
 
        return self.crusher1


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


    def normalize_data(self, data, scaler):
        variables                                  = data.select_dtypes('number')
        scaled_data                                = scaler.fit_transform(variables)
        self.scaled_data                           = data.copy()
        self.scaled_data.loc[:, variables.columns] = scaled_data
        return self.scaled_data


    def preprocessing(self, scaler, type_file='h5', window_size=60):
        self.data_loading(type_file=type_file)
        self.clear_data()
        self.addition_efficiency_var()
        self.split_time_series()        
        self.moving_average(window_size=window_size)
        warnings.filterwarnings("ignore")
        self.feature_engineering(window_size=window_size)
        warnings.filterwarnings("default")
        df          = self.remove_redundant_data()
        df_healthy  = self.get_healthy_state()

        scaled_data     = self.normalize_data(data   = df,
                                              scaler = scaler) 
        scaled_healthy  = self.normalize_data(data   = df_healthy, 
                                              scaler = scaler)
        
        return df, df_healthy, scaled_data, scaled_healthy
        