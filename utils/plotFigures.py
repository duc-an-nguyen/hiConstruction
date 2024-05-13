#! "C:\Users\CAEX\Desktop\Diego\Crusher Project\2ndrealdatajournalchile\mmodal\Scripts\python.exe"

from matplotlib.colors  import ListedColormap
import plotly.express as px
import plotly.graph_objects as go

def linePlot(data, x, y, color, title=None):
    fig = px.line(data, 
                  x     = x, 
                  y     = y, 
                  color = color,
                  title = title)
    fig.show()

def lineAndScatterPlot(data, x, y, color, title='Time series'):
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

def scatterPlot2D(data, x, y, color, title='Scatterplot 2D'):
    fig = px.scatter(data, 
                     x      = x, 
                     y      = y, 
                     color  = color,
                     title  = title, 
                     width  = 600, 
                     height = 500)
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


