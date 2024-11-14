import torch.nn                     as nn
import pandas                       as pd
from model_rul.RulExperiment        import ExperimentRul

# Data is loaded
df = pd.read_hdf('dataset\processed_crusher.h5')
df['cycles'] =  df.groupby('trajectory').cumcount() # The cycles are the variable t from the paper 

# Lists of variables to use
manipulated_cr1=['corriente(A)','nivel_tolva_compensacion(%)','Motor_Fase1(A)','Motor_Fase2(A)',
                'Motor_Fase3(A)','Motor_Fase1(V)','Motor_Fase2(V)','Lubricación_principal(A)','Lubricación_respaldo(A)']

efficiency_cr1 = ['posicion_poste(%)','Piñón_ext(°C)','Piñón_int(°C)','Lubricación_depósito(%)','Lubricación_entrada_sup(psi)',
                  'Lubricación_entrada_inf(psi)','Lubricación_principal(psi)','presion_estanque_aire(bar)']

new_efficiency_cr1 = ['Eficiencia_Rendimiento',    'Vibracion_piñon_int_norm',  'Vibracion_piñon_ext_norm', 
                      'Desviacion_Vibracion_Int',  'Desviacion_Vibracion_Ext',  'Discrepancia_temperatura_lubricacion']
        
new_manipulable_cr1 = ['Discrepancia_Corriente_Motor', 'Discrepancia_Voltaje_Motor', 'Sistema Lubricación(A)']

# The input variables to the functions are set
input_columns = manipulated_cr1+new_manipulable_cr1 #W
output_columns = efficiency_cr1+new_efficiency_cr1 #X
window_size = 48

# Model parameters
model_params = {'out_channels':[128,64,12], 
                'dim_x': len(output_columns),
                'dim_w': len(input_columns), 
                'window_size': window_size,
                'z_size': 1, 
                'kernel_size': 85, 
                'stride': 2, 
                'alpha': 1, 
                'constraint':'F'}

# Training function parameters
config = {'data': df,
          'input_columns': input_columns,
          'output_columns': output_columns,
          'window_size': window_size,
          'batch_size': 64,
          'criterion': nn.L1Loss(),
          'epochs': 8,
          'lr': 1e-4,
          'plot_loss': True,
          'directory': 'saved_models/rul_model_main_10epoch.pt',
          'model_params': model_params}

experiment = ExperimentRul(**config)
if __name__ == "__main__":
    experiment.train()
    experiment.extract_latent_space()
    experiment.plot_learning_curves()
    experiment.plot_rul()

