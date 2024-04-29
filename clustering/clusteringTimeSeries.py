#! "C:\Users\CAEX\Desktop\Diego\Crusher Project\2ndrealdatajournalchile\mmodal\Scripts\python.exe"

import numpy as np
import pandas as pd
from tqdm                           import tqdm

from matplotlib.colors              import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import euclidean_distances
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
        for trajectory in tqdm(self.trajectories[:6]):
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
        self.trajectory       = self.trajectories[idx]
        trajectoryResult = self.clustersResults[self.trajectory]

        return trajectoryResult


    # =====================================
    # This func calculates the average 
    # of each cluster in the embedded space
    # -------------------------------------
    def getCentroids(self, data, algorithm):
        embeddingData   = data[[f'{algorithm}_x', f'{algorithm}_y', f'{algorithm}_z']]
        clusterLabels   = data[f'cluster_{algorithm}'].astype('int')

        centroids = []
        for label in clusterLabels.unique():
            if label == -1: # Outliers are ignored
                continue

            cluster_points = embeddingData[clusterLabels == label]
            centroid       = np.mean(cluster_points, axis=0).values
            centroids.append(centroid)

        return embeddingData, clusterLabels, np.array(centroids)


    # ======================================
    # This function returns similar clusters 
    # in different trajectories based on the 
    # distances of the centroids
    # --------------------------------------
    def getSimilarClusters(self, algorithm, threshold):
        L               = len(self.clustersResults)
        iterator        = 0 
        self.similarClusters = pd.DataFrame()
        for idxTrajectory1 in range(L):
            for idxTrajectory2 in range(idxTrajectory1+1, L):

                # =========================
                # Calculates centroids     
                # for pairs of trajectories
                # -------------------------
                dataTrajectory1                             = self.getTrajectory(idxTrajectory1) 
                trajectory1                                 = self.trajectory
                embeddingData1, clusterLabels1, centroids1  = self.getCentroids(dataTrajectory1, algorithm)

                dataTrajectory2                             = self.getTrajectory(idxTrajectory2)
                trajectory2                                 = self.trajectory
                embeddingData2, clusterLabels2, centroids2  = self.getCentroids(dataTrajectory2, algorithm)

                # ==========================
                # centroids whose distance 
                # is less than the threshold 
                # are considered similar
                # --------------------------
                distances                                   = euclidean_distances(centroids1, centroids2)
                for i, row in enumerate(distances):
                    for j, distance in enumerate(row):
                        if distance < threshold:
                            cluster1, cluster2              = (i,j) 
                            pointsCluster1                  = embeddingData1[clusterLabels1 == cluster1].copy()
                            pointsCluster2                  = embeddingData2[clusterLabels2 == cluster2].copy() 
                            pointsCluster1.loc[:,'label']   = f'cluster: {cluster1}, trajectory:{trajectory1}'
                            pointsCluster2.loc[:,'label']   = f'cluster: {cluster2}, trajectory:{trajectory2}'
                            
                            dfResuts                        = pd.concat([pointsCluster1, pointsCluster2])
                            dfResuts.loc[:, 'group']        = iterator
                            self.similarClusters            = pd.concat([self.similarClusters, dfResuts], ignore_index=True)
                            iterator+=1

        return self.similarClusters 
    
    
    # ===================================
    # This func graphs different 
    # figures with the clustering results
    # ------------------------------------
    def plotFigure(self, typePlot, idxTrajectory, idxVariable, algorithm):
        data       = self.getTrajectory(idxTrajectory)
        variable   = data.select_dtypes('number').columns[idxVariable]

        if typePlot  == 'lineplot':
            lineAndScatterPlot(data, 
                               x='tiempo', 
                               y=variable, 
                               color=f'cluster_{algorithm}')
            
        elif typePlot == 'scatter2D':
            scatterPlot2D(data, 
                          x=f'{algorithm}_x', 
                          y=f'{algorithm}_y', 
                          color=f'cluster_{algorithm}',
                          title=f'Clusters from trajectory {self.trajectory}')
            
        elif typePlot == 'scatter3D':
            scatterPlot3D(data, 
                          x=f'{algorithm}_x', 
                          y=f'{algorithm}_y', 
                          z=f'{algorithm}_z', 
                          color=f'cluster_{algorithm}',
                          title=f'Clusters from trajectory {self.trajectory}')
            

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
