#! "C:\Users\CAEX\Desktop\Diego\Crusher Project\2ndrealdatajournalchile\mmodal\Scripts\python.exe"

import numpy    as np
import pandas   as pd
import networkx as nx

from tqdm                           import tqdm
from sklearn.metrics.pairwise       import euclidean_distances


class ClusteringTimeSeries: 
    def __init__(self, data, dataNormalized, clusteringModels, dimReductionModels):
        self.data               = data
        self.dataNormalized     = dataNormalized
        self.columns            = dataNormalized.columns
        self.trajectories       = self.data.trajectory.dropna().unique()
        self.clustersResults    = {}

        self.clusteringModels   = clusteringModels
        self.dimReductionModels = dimReductionModels
        self.algorithms         = list(self.dimReductionModels.keys())
        self.n_components       = dimReductionModels[self.algorithms[0]].n_components



    def trainPrevious(self, trainTrajectory):
        self.trainTrajectory = trainTrajectory
        trainData = self.dataNormalized[self.data['trajectory']==self.trainTrajectory]
        self.embeddingTrain = trainData.copy()
        for algorithm in self.algorithms:
            self.dimReductionModels[algorithm].fit(trainData)
            if self.n_components == 2:
                self.embeddingTrain.loc[:, [f'{algorithm}_x', 
                                            f'{algorithm}_y']] = self.dimReductionModels[algorithm].transform(trainData)
            else:
                self.embeddingTrain.loc[:, [f'{algorithm}_x', 
                                            f'{algorithm}_y',
                                            f'{algorithm}_z']] = self.dimReductionModels[algorithm].transform(trainData)

    # ==================================
    # this func trains a dimensionality 
    # reduction model using healthy data
    # ----------------------------------
    def train(self, supervised=False):
        trainData = self.dataNormalized
        self.embedding = trainData.copy()
        for algorithm in self.algorithms:
            # Train algorithm
            if supervised:
                self.dimReductionModels[algorithm].fit(trainData, self.data['label'])
            else:
                self.dimReductionModels[algorithm].fit(trainData)
                
            # Get the embedding data for visualization
            if self.n_components == 2:
                self.embedding.loc[:, [f'{algorithm}_x', 
                                            f'{algorithm}_y']] = self.dimReductionModels[algorithm].transform(trainData)
            else:
                self.embedding.loc[:, [f'{algorithm}_x', 
                                            f'{algorithm}_y',
                                            f'{algorithm}_z']] = self.dimReductionModels[algorithm].transform(trainData)
                
                
    # ================================
    # This func finds the clusters for 
    # a trajectory of the time series
    # --------------------------------
    def getResults2(self, trajectory=None): 
        if trajectory is not None:
            mask = self.data['trajectory'] == trajectory
        else:
            mask = slice(None)
        
        df_filtered                 = self.embedding[mask].copy() 
        df_filtered['tiempo']       = self.data.loc[mask, 'tiempo']
        df_filtered['trajectory']   = self.data.loc[mask, 'trajectory']
        #similarClusters['period']   = self.data.loc[mask, 'period']

        # Apply dimensionality reduction and clustering
        for algorithm in self.algorithms:
            clusteringModel     = self.clusteringModels[algorithm]
            clusterLabels       = clusteringModel.fit_predict(df_filtered[[f'{algorithm}_x', f'{algorithm}_y']])
            df_filtered[f'cluster_{algorithm}'] = clusterLabels.astype('str')
            #df_filtered[f'cluster'] = clusterLabels.astype('str')

        # Store results in cluster Results
        if trajectory is not None:
            self.clustersResults[trajectory] = df_filtered
        else:
            self.embedding = df_filtered


    # ===================================
    # This func reduces the dimension of 
    # the data and finds the clusters in 
    # the embedded space.
    # -----------------------------------
    def findClusters(self, data, clusteringModel, dimReductionModel):
        transformedData = dimReductionModel.transform(data) 
        clusterLabels   = clusteringModel.fit_predict(transformedData)
        return transformedData, clusterLabels

    # ================================
    # This func finds the clusters for 
    # a trajectory of the time series
    # --------------------------------
    def getResults(self, trajectory):
        axis = ['x', 'y', 'z']
        
        dfFiltered                = self.dataNormalized[self.data['trajectory']==trajectory]
        similarClusters           = dfFiltered.copy()
        similarClusters['tiempo'] = self.data[self.data['trajectory']==trajectory]['tiempo'] 
        similarClusters['period'] = self.data[self.data['trajectory']==trajectory]['period'] 

        for algorithm in self.algorithms:
            clusteringModel                          = self.clusteringModels[algorithm]
            dimReductionModel                        = self.dimReductionModels[algorithm]
            transformedData, clusterLabels           = self.findClusters(dfFiltered, clusteringModel, dimReductionModel)
            similarClusters[f'cluster_{algorithm}']  = clusterLabels.astype('str')

            for i in range(self.n_components):
                similarClusters[f'{algorithm}_{axis[i]}'] = transformedData[:, i]

        self.clustersResults[trajectory] = similarClusters
    
    
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
        for label in np.sort((clusterLabels.unique())):
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
                            
                            pointsCluster1.loc[:, 'cluster'] = cluster1
                            pointsCluster2.loc[:, 'cluster'] = cluster2
                            
                            pointsCluster1.loc[:, 'trajectory'] = trajectory1
                            pointsCluster2.loc[:, 'trajectory'] = trajectory2

                            pointsCluster1.loc[:,'label']   = f'cluster: {cluster1}, trajectory: {trajectory1}'
                            pointsCluster2.loc[:,'label']   = f'cluster: {cluster2}, trajectory: {trajectory2}'

                            
                            dfResuts                        = pd.concat([pointsCluster1, pointsCluster2])
                            dfResuts.loc[:, 'group']        = iterator
                            self.similarClusters            = pd.concat([self.similarClusters, dfResuts], ignore_index=True)
                            iterator+=1

        return self.similarClusters 
    

    # ==========================================
    # This func groups all similar clusters 
    # using the connected components of a graph
    # ------------------------------------------
    def unifyClusters(self, algorithm):

        # The information is filtered to obtain the number of similarities
        if self.n_components==2:
            dfSimilarClusters = self.similarClusters.drop(columns = [f'{algorithm}_x', 
                                                                     f'{algorithm}_y']).drop_duplicates().reset_index(drop=True)
        else:
            dfSimilarClusters = self.similarClusters.drop(columns = [f'{algorithm}_x', 
                                                                     f'{algorithm}_y',
                                                                     f'{algorithm}_z']).drop_duplicates().reset_index(drop=True)
            
        dfSimilarClusters['trajectory'] = dfSimilarClusters['trajectory'].astype('int')

        # ==============================
        # List with similarities between trajectory and cluster 
        # Each element is: (trajectory i, cluster n) is similar to (trajectory j, cluster m)
        # ------------------------------
        similarityTuples                = [((group['trajectory'].iloc[0], group['cluster'].iloc[0]), 
                                            (group['trajectory'].iloc[1], group['cluster'].iloc[1]))
                                            for _, group in dfSimilarClusters.groupby('group')]  
        
        # Create a graph
        self.Graph = nx.DiGraph()
        # Add nodes and edges to the graph
        for sim_tuple in similarityTuples:
            source_node = sim_tuple[0]
            target_node = sim_tuple[1]
            self.Graph.add_edge(source_node, target_node)

        # Find connected components: number of real clusters
        self.connectedComponents = list(nx.weakly_connected_components(self.Graph))

        # Add real cluster to results
        for key, df in self.clustersResults.items():
            realClusters = []
            for _, row in df.iterrows():
                flag = False
                x    = int(key) 
                y    = int(row[f'cluster_{algorithm}'])  

                # Find in which connected component the tuple (x, y) is located
                for i, component in enumerate(self.connectedComponents):
                    if (x, y) in component:
                        flag = True
                        realClusters.append(i) 
                        break  

                if flag == False:
                    realClusters.append(-1) 

            df[f'real_cluster_{algorithm}'] = realClusters
