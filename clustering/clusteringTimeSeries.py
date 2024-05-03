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
        self.n_components       = dimReductionModels[self.algorithms[0]].n_components


     #=========================================
    # this func trains a dimensionality 
    # reduction model using only one trajectory
    # -----------------------------------------
    def train(self, trainTrajectory):
        self.trainTrajectory = trainTrajectory
        trainData = self.dataNormalized[self.data['trajectory']==self.trainTrajectory]
        for algorithm in self.algorithms:
            self.dimReductionModels[algorithm].fit(trainData)


    # ===================================
    # This func reduces the dimension of 
    # the data and finds the clusters in 
    # the embedded space.
    # -----------------------------------
    def findClusters(self, data, clusteringModel, dimReductionModel):
        transformedData = dimReductionModel.transform(data) #fit_transform previous version
        clusterLabels   = clusteringModel.fit_predict(transformedData)
        return transformedData, clusterLabels
    

    # =====================================
    # This func finds the clusters for each 
    # trajectory of the time series
    # -------------------------------------
    def getResults(self):
        self.clustersResults = {}
        axis = ['x', 'y', 'z']

        for trajectory in tqdm(self.trajectories[:6]):
            dfFiltered          = self.dataNormalized[self.data['trajectory']==trajectory]
            dfResults           = dfFiltered.copy()
            dfResults['tiempo'] = self.data[self.data['trajectory']==trajectory]['tiempo'] 


            for algorithm in self.algorithms:
                clusteringModel                    = self.clusteringModels[algorithm]
                dimReductionModel                  = self.dimReductionModels[algorithm]
                transformedData, clusterLabels     = self.findClusters(dfFiltered, clusteringModel, dimReductionModel)
                dfResults[f'cluster_{algorithm}']  = clusterLabels.astype('str')

                for i in range(self.n_components):
                    dfResults[f'{algorithm}_{axis[i]}'] = transformedData[:, i]

            self.clustersResults[trajectory] = dfResults
        return self.clustersResults
    
    
    # ===============================
    # This func returns the results 
    # obtained for a single sequence
    # -------------------------------
    def getTrajectory(self, idx):
        self.trajectory  = self.trajectories[idx]
        trajectoryResult = self.clustersResults[self.trajectory]
        
        return trajectoryResult


    # =====================================
    # This func calculates the average 
    # of each cluster in the embedded space
    # -------------------------------------
    def getCentroids(self, data, algorithm):
        if self.n_components==2:
            embeddingData = data[[f'{algorithm}_x', f'{algorithm}_y']]
        else:
            embeddingData = data[[f'{algorithm}_x', f'{algorithm}_y', f'{algorithm}_z']]
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
                               x     = 'tiempo', 
                               y     = variable, 
                               color = f'cluster_{algorithm}',
                               title = f'Clusters from trajectory {self.trajectory}')
            
        elif typePlot == 'scatter2D':
            scatterPlot2D(data, 
                          x     = f'{algorithm}_x', 
                          y     = f'{algorithm}_y', 
                          color = f'cluster_{algorithm}',
                          title = f'Clusters from trajectory {self.trajectory}')
            
        elif typePlot == 'scatter3D':
            scatterPlot3D(data, 
                          x     = f'{algorithm}_x', 
                          y     = f'{algorithm}_y', 
                          z     = f'{algorithm}_z', 
                          color = f'cluster_{algorithm}',
                          title = f'Clusters from trajectory {self.trajectory}')
            

    # ================================
    # This func can graph several 
    # figures with of embedding spaces 
    # of different trajectories
    # --------------------------------
    def plotEmbedding(self, nfigures):
        ncols = len(self.algorithms)
        trainData  = self.clustersResults[self.trainTrajectory]

        if self.n_components == 3:
            for i in range(nfigures):
                fig, axes  = plt.subplots(nrows      = 1, 
                                          ncols      = ncols, 
                                          figsize    = (5*ncols, 4),
                                          subplot_kw = {"projection": "3d"})
                
                trajectory = self.trajectories[i]
                data       = self.getTrajectory(i)

                for j in range(ncols):
                    algorithm = self.algorithms[j]
                    axes[j].scatter(xs     = trainData[f'{algorithm}_x'], 
                                    ys     = trainData[f'{algorithm}_y'], 
                                    zs     = trainData[f'{algorithm}_z'], 
                                    label  = f'trajectory {self.trainTrajectory} (train)',            
                                    marker = '.',
                                    alpha  = 0.3,
                                    s      = 10 )
                    
                    axes[j].scatter(xs     = data[f'{algorithm}_x'], 
                                    ys     = data[f'{algorithm}_y'], 
                                    zs     = data[f'{algorithm}_z'], 
                                    label  = f'trajectory {trajectory}',            
                                    marker = '.',
                                    alpha  = 0.3,
                                    s      = 10 )
                    
                    axes[j].set_xlabel(f'{algorithm}_x')
                    axes[j].set_ylabel(f'{algorithm}_y')
                    axes[j].set_ylabel(f'{algorithm}_z')
                    axes[j].legend(fontsize = 8, 
                                   loc      = 'upper left')
                    
                fig.tight_layout()
                fig.suptitle(f'Dimensionality reduction for trajectory {trajectory}', 
                             fontsize = 10, 
                             y        = 1.04)
            
                
        else:
            for i in range(nfigures):
                fig, axes  = plt.subplots(nrows   = 1, 
                                          ncols   = ncols, 
                                          figsize = (5*ncols, 4))
                
                trajectory = self.trajectories[i]
                data       = self.getTrajectory(i)

                for j in range(ncols):
                    algorithm = self.algorithms[j]
                    axes[j].scatter(x      = trainData[f'{algorithm}_x'], 
                                    y      = trainData[f'{algorithm}_y'], 
                                    label  = f'trajectory {self.trainTrajectory} (train)',            
                                    marker = '.',
                                    alpha  = 0.3,
                                    s      = 10 )
                    
                    axes[j].scatter(x      = data[f'{algorithm}_x'], 
                                    y      = data[f'{algorithm}_y'], 
                                    label  = f'trajectory {trajectory}',            
                                    marker = '.',
                                    alpha  = 0.3,
                                    s      = 10 )
                    
                    axes[j].set_xlabel(f'{algorithm}_x')
                    axes[j].set_ylabel(f'{algorithm}_y')
                    axes[j].legend(fontsize = 8, 
                                   loc      = 'upper left')
                    
                fig.tight_layout()
                fig.suptitle(f'Dimensionality reduction for trajectory {trajectory}', 
                             fontsize = 10, 
                             y        = 1.04)
            

    # ==================================
    # This func can graph several 
    # figures with the clusters obtained 
    # in the embedding space
    # ----------------------------------
    def plotResults(self, nfigures, save=False):
        cmap  = ListedColormap(sns.color_palette("hls", 256).as_hex())
        ncols = len(self.algorithms)

        if self.n_components == 3:
            for i in range(nfigures):
                fig, axes  = plt.subplots(nrows      = 1, 
                                          ncols      = ncols, 
                                          figsize    = (5*ncols, 4),
                                          subplot_kw = {"projection": "3d"})
                
                trajectory = self.trajectories[i]
                data       = self.getTrajectory(i)

                for j in range(ncols):
                    algorithm = self.algorithms[j]
                    scatterplot = axes[j].scatter(xs     = data[f'{algorithm}_x'], 
                                                  ys     = data[f'{algorithm}_y'], 
                                                  zs     = data[f'{algorithm}_z'], 
                                                  c      = data[f'cluster_{algorithm}'].astype('int'), 
                                                  cmap   = cmap,                
                                                  marker = '.',
                                                  alpha  = 0.3,
                                                  s      = 10 )
                    
                    axes[j].set_xlabel(f'{algorithm}_x')
                    axes[j].set_ylabel(f'{algorithm}_y')
                    axes[j].set_ylabel(f'{algorithm}_z')
                    axes[j].legend(*scatterplot.legend_elements(), 
                                   loc = 'upper left')
                    
                fig.tight_layout()
                fig.suptitle(f'Clusters in embedding space for trajectory {trajectory}', 
                             fontsize = 10, 
                             y        = 1)
                
        else:
            for i in range(nfigures):
                fig, axes  = plt.subplots(nrows   = 1, 
                                          ncols   = ncols, 
                                          figsize = (5*ncols, 4))
                
                trajectory = self.trajectories[i]
                data       = self.getTrajectory(i)

                for j in range(ncols):
                    algorithm = self.algorithms[j]
                    scatterplot = axes[j].scatter(x      = data[f'{algorithm}_x'], 
                                                  y      = data[f'{algorithm}_y'], 
                                                  c      = data[f'cluster_{algorithm}'].astype('int'), 
                                                  cmap   = cmap,                
                                                  marker = '.',
                                                  alpha  = 0.3,
                                                  s      = 10 )
                    
                    axes[j].set_xlabel(f'{algorithm}_x')
                    axes[j].set_ylabel(f'{algorithm}_y')
                    axes[j].legend(*scatterplot.legend_elements(), 
                                   bbox_to_anchor = (1, 1),
                                   loc            = 'upper left')
                    
                fig.tight_layout()
                fig.suptitle(f'Clusters in embedding space for trajectory {trajectory}', 
                             fontsize = 10, 
                             y        = 1.04)
                    
            if save:
                plt.savefig(f'clustering_figures\{trajectory}.png')
                plt.close(fig)
            else:
                plt.show()

    # ==============================
    # This func can graph the points 
    # obtained by each cluster for 
    # some variable
    # ------------------------------
    def plotVarByCluster(self, idxTrajectory, variable, algorithm):
        dataTrajectory = self.getTrajectory(idxTrajectory)
        clusterLabels  = dataTrajectory[f'cluster_{algorithm}'].unique()
        clusterLabels  = clusterLabels[clusterLabels!=str(-1)]

        ymax           = dataTrajectory[dataTrajectory[f'cluster_{algorithm}'].isin(clusterLabels)][variable].max()
        ymin           = dataTrajectory[dataTrajectory[f'cluster_{algorithm}'].isin(clusterLabels)][variable].min()
        ncols          = len(clusterLabels)

        fig, axes      = plt.subplots(nrows=1, ncols=ncols, figsize=(2*ncols, 4))
        for j, cluster in enumerate(clusterLabels):
            dataFiltered = dataTrajectory[dataTrajectory[f'cluster_{algorithm}']==cluster]
            axes[j].scatter(x = dataFiltered.reset_index().index, 
                            y = dataFiltered[variable], 
                            s = 5)
            axes[j].set_title(f'Cluster {cluster}', 
                              fontsize = 10)
            axes[j].set_ylim(ymin, ymax)
        
            if j==0:
                axes[j].set_ylabel(f'{variable}')
            else:
                axes[j].yaxis.set_visible(False)

        fig.tight_layout()
        fig.suptitle(f'Comparison of the {variable} by clusters for trajectory {self.trajectory}', 
                     fontsize = 15, 
                     y        = 1.04)
        plt.show()