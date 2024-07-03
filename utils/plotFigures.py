#! "C:\Users\CAEX\Desktop\Diego\Crusher Project\2ndrealdatajournalchile\mmodal\Scripts\python.exe"

import matplotlib.gridspec              as gridspec
import matplotlib.pyplot                as plt
import plotly.express                   as px
import plotly.graph_objects             as go
import seaborn                          as sns

from matplotlib.colors                  import ListedColormap
from plotly.subplots                    import make_subplots


def linePlot(data, x, y, color=None, title=None):
    if color is not None:
        data        = data.copy()
        data[color] = data[color].astype('str')

    fig = px.line(data, 
                  x     = x, 
                  y     = y, 
                  color = color,
                  title = title)
    fig.show()

def lineAndScatterPlot(data, x, y, color, title='Time series'):
    data        = data.copy()
    data[color] = data[color].astype('str')
    fig = px.scatter(data, 
                     x      = x, 
                     y      = y, 
                     color  = color)
    
    line_plot = px.line(data, 
                        x   = x, 
                        y   = y)

    fig.update_layout(title=title)
    fig.add_trace(line_plot.data[0])
    fig.update_traces(line=dict(color='black'))
    fig.show()

def scatterPlot2D(data, x, y, color=None, title='Scatterplot 2D', color_discrete_sequence=None, n_colors=None):

    if color_discrete_sequence is not None:
        color_discrete_sequence = px.colors .sample_colorscale(colorscale   = "Plasma_r", 
                                                               samplepoints = [n/(n_colors-1) for n in range(n_colors)])
    if color is not None:
        data        = data.copy()
        data[color] = data[color].astype('str')
        
    fig = px.scatter(data, 
                     x      = x, 
                     y      = y, 
                     color  = color,
                     title  = title, 
                     width  = 600, 
                     height = 500,
                     color_discrete_sequence = color_discrete_sequence)
    fig.show()

def scatterPlot3D(data, x, y, z, color, title='Scatterplot 3D'):
    fig = px.scatter_3d(data,
                        x        = x, 
                        y        = y, 
                        z        = z,
                        color    = color,
                        title    = title, 
                        size_max = 10)
    
    fig.update_layout(width  = 600, 
                      height = 500)
    fig.show()


# =============================================
# All trajectories colored by label are plotted
# ---------------------------------------------
def plotLabels(dataframe, variable='Piñón_ext(°C)', colorscale='HSV'):
    fig     = go.Figure()
    color   = px.colors.sample_colorscale(colorscale    = colorscale, 
                                          samplepoints  = int(dataframe['label'].astype('float').max()+2))

    legend_items = []

    for trajectory in dataframe.trajectory.unique():
        dataFiltered = dataframe[dataframe['trajectory'] == trajectory]
        lineTrace = []
        for i, label in enumerate(dataFiltered.label.unique()):
            data = dataFiltered[dataFiltered['label'] == label]

            lineTrace.append(go.Scattergl(x             = data['tiempo'],
                                          y             = data[variable],
                                          mode          = 'lines',
                                          name          = label,
                                          text          = f'trajectory {trajectory}',
                                          hoverinfo     = 'all',
                                          line          = dict(color=color[i]),
                                          legendgroup   = 'group' + str(label),
                                          showlegend    = ('label ' + label not in legend_items),
                                          ))

            if label not in legend_items:
                lineTrace.append(go.Scattergl(x             = [None], 
                                              y             = [None],
                                              mode          = 'markers',
                                              marker        = dict(size=10, color=color[i]),
                                              name          = f'label {label}',
                                              legendgroup   = 'group' + str(label),
                                              showlegend    = False
                                              ))
                legend_items.append('label ' + label)

        fig.add_traces(lineTrace)

    fig.update_layout(
        title       = f'"{variable}" colored by label',
        xaxis_title = 'tiempo',
        yaxis_title = 'Piñón_ext(°C)',
        showlegend  = True
        )
    fig.show()


# =============================================================
# A figure is created containing a time series colored by hour, 
# and the embedding results for each trajectory colored by hour
# -------------------------------------------------------------
def plotResults(trajectory, 
                data1, 
                data2, 
                variable    = 'Piñón_int(mm/s)', 
                palette     = 'Spectral', 
                hue         = 'period',
                algorithm   = 'umap'):

        # Figure and axes
        fig  = plt.figure(figsize = (10, 6))
        gs   = gridspec.GridSpec(nrows          = 2, 
                                 ncols          = 2, 
                                 height_ratios  = [0.5, 1])
        axes = [fig.add_subplot(gs[0, :]),
                fig.add_subplot(gs[1, 0]),
                fig.add_subplot(gs[1, 1])]

        # Subfigure
        sns.lineplot(
            x       = 'tiempo', 
            y       = variable, 
            data    = data1, 
            ax      = axes[0], 
            hue     = hue,
            palette = palette
        )

        sns.scatterplot(
            x       = f'{algorithm}_x', 
            y       = f'{algorithm}_y', 
            data    = data1, 
            ax      = axes[1], 
            hue     = hue,
            palette = palette,
            marker  = '.', 
            alpha   = 0.3, 
            s       = 30
        )

        sns.scatterplot(
            x       = f'{algorithm}_x', 
            y       = f'{algorithm}_y', 
            data    = data2, 
            ax      = axes[2], 
            hue     = hue,
            palette = palette,
            marker  = '.', 
            alpha   = 0.3, 
            s       = 30
        )

        # Updating axis names
        axes[0].set_xlabel(' ')
        axes[0].legend_.remove()
        axes[1].legend_.remove()
        axes[2].legend_.remove()

        axes[0].set_title(label     = f'"{variable}" colored by one hour', 
                          fontsize  = 10)
        axes[1].set_title(label     = 'Idea 1: Unsupervised training with healthy data', 
                          fontsize  = 10)
        axes[2].set_title(label     = 'Idea 2: Supervised training', 
                          fontsize  = 10)

        fig.tight_layout()
        fig.suptitle(t          = f'Results colored by time for trajectory {trajectory}', 
                     y          = 1.02, 
                     fontsize   = 16);   


# ======================================================================
# A figure is created using plotly that containing a time series colored 
# by hour, and the embedding results for a trajectory colored by hour
# ----------------------------------------------------------------------
def plotResultsPlotly(trajectory, 
                      data1, 
                      data2,
                      algorithm   = 'umap', 
                      variable    = 'Piñón_int(mm/s)', 
                      palette     = 'Spectral', 
                      color       = 'period'):

    palette = px.colors.sample_colorscale(colorscale     = palette, 
                                          samplepoints   = len(data1.period.unique())) 

    # Create the structure of the figure
    fig = make_subplots(
        rows                = 2, 
        cols                = 2,
        row_heights         = [0.5, 1],
        vertical_spacing    = 0.12, 
        specs               = [[{"colspan": 2}, None],
                               [{},                {}]],
        
        # titles for each subfigure
        subplot_titles  = (
            f'"{variable}" colored by one hour',
            'Idea 1: Unsupervised training with healthy data',
            'Idea 2: Supervised training'
            ),
        )

    # subplot 1: time serie
    line_fig = px.line(data_frame               = data1, 
                       x                        = 'tiempo', 
                       y                        = variable, 
                       color                    = color, 
                       color_discrete_sequence  = palette)

    # add traces to main figure
    for trace in line_fig.data:
        trace.showlegend    = False
        fig.add_trace(trace = trace, 
                        row   = 1, 
                        col   = 1)


    # subplot 2: idea 1 
    scatter_fig1 = px.scatter(data_frame                = data1, 
                              x                         = f'{algorithm}_x', 
                              y                         = f'{algorithm}_y', 
                              color                     = color, 
                              color_discrete_sequence   = palette, 
                              opacity                   = 0.3)

    # add traces to main figure
    for trace in scatter_fig1.data:
        trace.showlegend    = False
        fig.add_trace(trace = trace, 
                      row   = 2, 
                      col   = 1)


    # subplot 3: idea 2 
    scatter_fig2 = px.scatter(data_frame                = data2, 
                              x                         = f'{algorithm}_x', 
                              y                         = f'{algorithm}_y', 
                              color                     = color, 
                              color_discrete_sequence   = palette, 
                              opacity                   = 0.3)

    # add traces to main figure
    for trace in scatter_fig2.data:
        fig.add_trace(trace = trace, 
                      row   = 2, 
                      col   = 2)

    # update main figure title parameters
    fig.update_layout(
        title_text  = f'Results colored by time for trajectory {trajectory}',
        title_y     = 0.92,
        title_x     = 0.47,
        height      = 650,
        width       = 1000,
        margin      = dict(t=100, b=50)
    )

    # update axes of each subfigure
    fig.update_yaxes(title_text=variable, titlefont=dict(size=12), row=1, col=1)
    fig.update_xaxes(title_text=f'{algorithm}_x', titlefont=dict(size=12), row=2, col=1)
    fig.update_yaxes(title_text=f'{algorithm}_y', titlefont=dict(size=12), row=2, col=1)
    fig.update_xaxes(title_text=f'{algorithm}_x', titlefont=dict(size=12), row=2, col=2)
    fig.update_yaxes(title_text=f'{algorithm}_y', titlefont=dict(size=12), row=2, col=2)

    # update title size from each subfigure
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=14)

    fig.show() 



def plot_learning_curves(train_acc_curve, val_acc_curve, train_loss_curve, val_loss_curve):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(train_loss_curve, label="Train loss")
    ax[0].plot(val_loss_curve, label="Valid loss")
    ax[0].legend()
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")

    ax[1].plot(train_acc_curve, label="Train acc")
    ax[1].plot(val_acc_curve, label="Valid acc")
    ax[1].legend()
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")

    fig.tight_layout()
    fig.suptitle('Comparación de las curvas de aprendizaje entre train y validation', fontsize=15, y = 1.05)
    plt.show()
