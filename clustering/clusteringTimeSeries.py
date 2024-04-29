#! "C:\Users\CAEX\Desktop\Diego\Crusher Project\2ndrealdatajournalchile\mmodal\Scripts\python.exe"

import numpy as np
import pandas as pd
from tqdm                           import tqdm

from matplotlib.colors              import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster                import DBSCAN, KMeans
from sklearn.preprocessing          import StandardScaler, MinMaxScaler
from sklearn.decomposition          import PCA

import pacmap
import umap.umap_ as umap

from utils.plotFigures      import lineAndScatterPlot, scatterPlot2D, scatterPlot3D


class ClusteringTimeSeries: 
    def __init__(self, data, dataNormalized, clusteringModels, dimReductionModels):
        self.data               = data
        self.dataNormalized     = dataNormalized
        self.trajectories       = self.data.trajectory.dropna().unique()

        self.clusteringModels   = clusteringModels
        self.dimReductionModels = dimReductionModels
        self.algorithms         = list(self.clusteringModels.keys())


    # ===================================
    # This func reduces the dimension of 
    # the data and finds the clusters in 
    # the embedded space.
    # -----------------------------------
    def findClusters(self, data, clusteringModel, dimReductionModel):
        transformedData = dimReductionModel.fit_transform(data)
        clusterLabels   = clusteringModel.fit_predict(transformedData)
        return transformedData, clusterLabels
    

    # =====================================
    # This func finds the clusters for each 
    # trajectory of the time series
    # -------------------------------------
    def getResults(self):
        self.clustersResults    = {}
        for trajectory in tqdm(self.trajectories[:2]):
            dfFiltered          = self.dataNormalized[self.data['trajectory']==trajectory]
            dfResults           = dfFiltered.copy()
            dfResults['tiempo'] = self.data[self.data['trajectory']==trajectory]['tiempo'] 

            for algorithm in self.algorithms:
                clusteringModel                    = self.clusteringModels[algorithm]
                dimReductionModel                  = self.dimReductionModels[algorithm]
                transformedData, clusterLabels     = self.findClusters(dfFiltered, clusteringModel, dimReductionModel)

                dfResults[f'{algorithm}_x']        = transformedData[:, 0]
                dfResults[f'{algorithm}_y']        = transformedData[:, 1]
                dfResults[f'{algorithm}_z']        = transformedData[:, 2]
                dfResults[f'cluster_{algorithm}']  = clusterLabels.astype('str')

            self.clustersResults[trajectory] = dfResults

        return self.clustersResults


    # ===============================
    # This func returns the results 
    # obtained for a single sequence
    # -------------------------------
    def getTrajectory(self, idx):
        trajectory       = self.trajectories[idx]
        trajectoryResult = self.clustersResults[trajectory]

        return trajectoryResult 
    
    # ===================================
    # This func graphs different 
    # figures with the clustering results
    # ------------------------------------
    def plotFigure(self, typePlot, idxTrajectory, idxVariable, algorithm):
        data     = self.getTrajectory(idxTrajectory)
        variable = data.select_dtypes('number').columns[idxVariable]

        if typePlot=='lineplot':
            lineAndScatterPlot(data, 
                               x='tiempo', 
                               y=variable, 
                               color=f'cluster_{algorithm}')
            
        elif typePlot=='scatter2D':
            scatterPlot2D(data, 
                          x=f'{algorithm}_x', 
                          y=f'{algorithm}_y', 
                          color=f'cluster_{algorithm}')
            
        elif typePlot=='scatter3D':
            scatterPlot3D(data, 
                          x=f'{algorithm}_x', 
                          y=f'{algorithm}_y', 
                          z=f'{algorithm}_z', 
                          color=f'cluster_{algorithm}')
            

    # ===================================
    # This graph function several figures 
    # with the groups obtained in the 
    # embedding space.
    # -----------------------------------
    def subplotsScatter3D(self, n_rows, save=False):
        cmap   = ListedColormap(sns.color_palette("hls", 256).as_hex())
        n_cols = len(self.algorithms)

        for i in range(n_rows):
            fig, axes   = plt.subplots(1, n_cols, figsize=(15, 5), subplot_kw={"projection": "3d"})
            trajectory  = self.trajectories[i]
            data        = self.getTrajectory(i)

            for j in range(n_cols):
                algorithm   = self.algorithms[j]
                scatterplot = axes[j].scatter(xs=data[f'{algorithm}_x'], 
                                              ys=data[f'{algorithm}_y'], 
                                              zs=data[f'{algorithm}_z'], 
                                              c=data[f'cluster_{algorithm}'].astype('int'),   
                                              s=20, 
                                              marker='o', 
                                              cmap=cmap, 
                                              alpha=0.3)
                
                axes[j].set_xlabel(f'{algorithm}_x')
                axes[j].set_ylabel(f'{algorithm}_y')
                axes[j].set_zlabel(f'{algorithm}_z')
                axes[j].legend(*scatterplot.legend_elements(), loc='upper left')
                
            fig.suptitle(f'Dimensionality reduction for trajectory {trajectory}', fontsize=10, y=0.88)

            if save:
                plt.savefig(f'clustering_figures\{trajectory}.png')
                plt.close(fig)
            else:
                plt.show()
