from utility.logger             import aiLogger
import matplotlib.pyplot        as plt
import plotly.graph_objects     as go
import os

class PlotlyDraw():
    def __init__(self):
        pass
        
    # ==========================
    # This function is used to
    # draw multiple or single chart
    # --------------------------
    def plot_matrix(self,data, title, makeImportantEvent = None, withLine = False, withShap=False, opacity = 0.3):
        fig = go.Figure()

        # ----------------------
        # Draw single matrix
        if type(data) is not list:
            fig.add_trace(go.Scatter(y=data, mode='lines'))  # Adding a line trace 

        # ----------------------
        # Draw multiple matrix
        else:
            for idxData,idxTitle in zip(data,title):
                # print(idxTitle,idxData)
                fig.add_trace(go.Scatter(y=idxData, mode='lines',name=idxTitle))  # Adding a line trace


        # ----------------------
        # Make important event
        if makeImportantEvent is not None:
            if withLine:
                for time in makeImportantEvent:
                    fig.add_vline(x=time, line_width=2, line_dash="dash", line_color="red")
            elif withShap:
                for period in makeImportantEvent:
                    fig.add_shape(
                        # Rectangle reference to the axes
                        type="rect",
                        x0=period['start'], y0=0,
                        x1=period['end'], y1=1,
                        xref="x", yref="paper",
                        fillcolor=period['color'],
                        opacity=period['opacity'],
                        layer="below",
                        line_width=0,)
            else:
                aiLogger.warning("Please add which type of making event!!!")


            


        


        # ----------------------
        # Update layout and show
        fig.update_layout(title="Results",xaxis_title='Index',yaxis_title='Value')             
        fig.show() 

    
        
        