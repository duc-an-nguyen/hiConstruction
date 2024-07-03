
#! "C:\Users\CAEX\Desktop\Diego\Crusher Project\2ndrealdatajournalchile\mmodal\Scripts\python.exe"

import pandas                           as pd
import networkx                         as nx
import seaborn                          as sns
import matplotlib.pyplot                as plt
import plotly.express                   as px
import plotly.graph_objects             as go

from matplotlib.colors                  import ListedColormap
from clustering.clusteringTimeSeries    import ClusteringTimeSeries
from utils.plotFigures                  import lineAndScatterPlot, scatterPlot2D, scatterPlot3D


class ClusteringResults(ClusteringTimeSeries):
    def __init__(self, data, dataNormalized, clusteringModels, dimReductionModels):
        super().__init__(data, dataNormalized, clusteringModels, dimReductionModels)

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
        ncols           = len(self.algorithms)

        if self.n_components==2:
            for i in range(nfigures):
                fig, axes  = plt.subplots(nrows   = 1, 
                                          ncols   = ncols, 
                                          figsize = (5*ncols, 4))       
                if ncols == 1:
                    axes = [axes]
                
                trajectory = self.trajectories[i]
                data       = self.getTrajectory(i)

                for j in range(ncols):
                    algorithm = self.algorithms[j]
                    axes[j].scatter(x      = self.embeddingTrain[f'{algorithm}_x'], 
                                    y      = self.embeddingTrain[f'{algorithm}_y'], 
                                    label  = 'train data', 
                                    #label  = f'trajectory {self.trainTrajectory} (train)',            
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
        
        else:    
            for i in range(nfigures):
                fig, axes  = plt.subplots(nrows      = 1, 
                                          ncols      = ncols, 
                                          figsize    = (5*ncols, 4),
                                          subplot_kw = {"projection": "3d"})
                if ncols == 1:
                    axes = [axes]
                
                trajectory = self.trajectories[i]
                data       = self.getTrajectory(i)

                for j in range(ncols):
                    algorithm = self.algorithms[j]
                    axes[j].scatter(xs     = self.embeddingTrain[f'{algorithm}_x'], 
                                    ys     = self.embeddingTrain[f'{algorithm}_y'], 
                                    zs     = self.embeddingTrain[f'{algorithm}_z'], 
                                    label  = 'train data', 
                                    #label  = f'trajectory {self.trainTrajectory} (train)',            
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
            

    # ==================================
    # This func can graph several 
    # figures with the clusters obtained 
    # in the embedding space
    # ----------------------------------
    def plotClusters(self, nfigures, label_color='cluster', save=False):
        cmap  = ListedColormap(sns.color_palette("hls", 256).as_hex()[10:])
        ncols = len(self.algorithms)

        if self.n_components == 2:
            for i in range(nfigures):
                fig, axes  = plt.subplots(nrows   = 1, 
                                          ncols   = ncols, 
                                          figsize = (5*ncols, 4))            
                if ncols == 1:
                    axes = [axes]
                
                trajectory = self.trajectories[i]
                data       = self.getTrajectory(i)

                for j in range(ncols):
                    algorithm = self.algorithms[j]
                    scatterplot = axes[j].scatter(x      = data[f'{algorithm}_x'], 
                                                  y      = data[f'{algorithm}_y'], 
                                                  c      = data[f'{label_color}_{algorithm}'].astype('int'), 
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
        
        else:
            for i in range(nfigures):
                fig, axes  = plt.subplots(nrows      = 1, 
                                          ncols      = ncols, 
                                          figsize    = (5*ncols, 4),
                                          subplot_kw = {"projection": "3d"})
                if ncols == 1:
                    axes = [axes]
                
                trajectory = self.trajectories[i]
                data       = self.getTrajectory(i)

                for j in range(ncols):
                    algorithm = self.algorithms[j]
                    scatterplot = axes[j].scatter(xs     = data[f'{algorithm}_x'], 
                                                  ys     = data[f'{algorithm}_y'], 
                                                  zs     = data[f'{algorithm}_z'], 
                                                  c      = data[f'{label_color}_{algorithm}'].astype('int'), 
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
    def plotVarByCluster(self, idxTrajectory, variable, algorithm, label_color='real_cluster'):
        dataTrajectory = self.getTrajectory(idxTrajectory)
        clusterLabels  = dataTrajectory[f'{label_color}_{algorithm}'].unique()
        clusterLabels  = clusterLabels[clusterLabels!=-1]

        ymax           = dataTrajectory[dataTrajectory[f'{label_color}_{algorithm}'].isin(clusterLabels)][variable].max()
        ymin           = dataTrajectory[dataTrajectory[f'{label_color}_{algorithm}'].isin(clusterLabels)][variable].min()
        ncols          = len(clusterLabels)

        fig, axes      = plt.subplots(nrows   = 1, 
                                      ncols   = ncols, 
                                      figsize = (2*ncols, 4))
        for j, cluster in enumerate(clusterLabels):
            dataFiltered = dataTrajectory[dataTrajectory[f'{label_color}_{algorithm}']==cluster]
            axes[j].scatter(x = dataFiltered.reset_index().index, 
                            y = dataFiltered[variable], 
                            s = 5)
            
            axes[j].set_ylim(ymin, ymax)
            axes[j].set_title(f'Cluster {cluster}', 
                              fontsize = 8)

            if j==0:
                axes[j].set_ylabel(f'{variable}')
            else:
                axes[j].yaxis.set_visible(False)

        fig.tight_layout()
        fig.suptitle(f'Comparison of the {variable} by clusters for trajectory {self.trajectory}', 
                     fontsize = 15, 
                     y        = 1.04)
        plt.show()


    # ====================================
    # This func plots a graph that shows 
    # the similarity between the different 
    # clusters of all trajectories
    # ------------------------------------
    def plotGraph(self):
        # Assign colors to nodes based on the connected component they belong to
        colors      = ['red', 'royalblue', 'green', 'orange', 'purple', 'yellow'] # Change length of colors
        nodeColors = {}
        edgeColors = {}

        for i, component in enumerate(self.connectedComponents):
    
            # Add colors to node and edges
            for node in component:
                nodeColors[node] = colors[i % len(colors)]

            for edge in self.Graph.edges():
                    if edge[0] in component and edge[1] in component:
                        edgeColors[edge] = colors[i % len(colors)]

        # Add position to graph        
        pos = {}
        for node in self.Graph.nodes():
            pos[node] = (node[0], node[1])

        # Create figure
        plt.figure(figsize  = (8, 6))
        nx.draw(G           = self.Graph, 
                pos         = pos, 
                with_labels = True, 
                node_size   = 700, 
                font_size   = 10, 
                arrowsize   = 20,
                node_color  = [nodeColors[node] for node in self.Graph.nodes()],  
                edge_color  = [edgeColors[edge] for edge in self.Graph.edges()], 
                )

        plt.xlabel('trajectories')
        plt.ylabel('clusters')
        plt.title("Assignment found between similar clusters")
        plt.show()


    def plotGraphPlotly(self):
        colors = px.colors.qualitative.Plotly 

        # Add position to graph
        pos = {}
        for node in self.Graph.nodes():
            pos[node] = (node[0], node[1])
            
        fig = go.Figure()    
        legend_items = [] 

        # Add components to the figure
        for i, component in enumerate(self.connectedComponents):
            node_trace = []
            edge_trace = []
            node_color = colors[i % len(colors)]

            # Add legend item for each component
            if i not in legend_items:
                node_trace.append(go.Scattergl(x            = [None], 
                                               y            = [None],
                                               mode         = 'markers',
                                               marker       = dict(size=10, color=node_color),
                                               name         = f'Component {i}',
                                               legendgroup  = 'group'+str(i),
                                               showlegend   = True))
                legend_items.append(i)

            for node in component:
                node_trace.append(go.Scattergl(x            = [pos[node][0]], y=[pos[node][1]],
                                               mode         = 'markers',
                                               marker       = dict(size=15, color=node_color),
                                               text         = str(node),
                                               hoverinfo    = 'text',
                                               legendgroup  = 'group'+str(i),
                                               showlegend   = False))

            for edge in self.Graph.edges():
                if edge[0] in component and edge[1] in component:
                    edge_trace.append(go.Scattergl(x           = [pos[edge[0]][0], pos[edge[1]][0]],
                                                   y           = [pos[edge[0]][1], pos[edge[1]][1]],
                                                   mode        = 'lines',
                                                   line        = dict(color=node_color, width=2),
                                                   hoverinfo   = 'none',
                                                   legendgroup = 'group'+str(i),
                                                   showlegend  = False))

            fig.add_traces(node_trace + edge_trace)

        fig.update_layout(title         = "Assignment found between similar clusters",
                          xaxis_title   = "trajectories",
                          yaxis_title   = "clusters",
                          showlegend    = True,
                          width         = 900,
                          height        = 700)
        fig.show()







        