from utility.logger             import aiLogger
import matplotlib.pyplot        as plt
import plotly.graph_objects     as go
import os

class PlotlyDraw():
    def __init__(self):
        # Assign a unique color to each event
        self.colors = [
            '#1f77b4',  # muted blue
            '#ff7f0e',  # safety orange
            '#2ca02c',  # cooked asparagus green
            '#d62728',  # brick red
            '#9467bd',  # muted purple
            '#8c564b',  # chestnut brown
            '#e377c2',  # raspberry yogurt pink
            '#7f7f7f',  # middle gray
            '#bcbd22',  # curry yellow-green
            '#17becf'   # blue-teal
        ]

        
    # ==========================
    # This function is used to
    # draw multiple or single chart
    # --------------------------
    def plot_matrix(self,   xData = None, 
                            yData = None, 
                            title = None, 
                            highLighHtPos = None, 
                            makeImportantEvent = None, 
                            withLine = False, 
                            withShap = False, 
                            opacity = 0.3):
        fig = go.Figure()

        # ----------------------
        # Draw single matrix
        if type(yData) is not list:
            fig.add_trace(go.Scatter(y=yData, mode='lines'))  # Adding a line trace 

        # ----------------------
        # Draw multiple matrix
        else:
            for idxData,idxTitle in zip(yData,title):
                # print(idxTitle,idxData)
                fig.add_trace(go.Scatter(y=idxData, mode='lines',name=idxTitle))  # Adding a line trace



        # ----------------------
        # Highligh the healthy data
        if highLighHtPos is not None:
            for idxHtPos in range(len(highLighHtPos["startingRow"])):
                lenOfHealthyRow         = int((highLighHtPos["endingRow"][idxHtPos] - highLighHtPos["startingRow"][idxHtPos]) * highLighHtPos["percent"][idxHtPos])

                startingPos             = highLighHtPos["startingRow"][idxHtPos] - 2
                endingPos               = highLighHtPos["startingRow"][idxHtPos] - 2 + lenOfHealthyRow

                fig.add_vrect(  x0              = startingPos, 
                                x1              = endingPos,
                                fillcolor       = "green", 
                                opacity         = 0.1,
                                layer           = "below", 
                                line_width      = 0,
                        )

        # ----------------------
        # Make important event
        if makeImportantEvent is not None:
            if withLine:
                for time in makeImportantEvent:
                    fig.add_vline(x=time, line_width=2, line_dash="dash", line_color="red")
                    
            elif withShap:
                # Generate plot for each event and time period
                for event, periods in makeImportantEvent.items():
                    for start_end in periods:
                        start, end = start_end
                        fig.add_vrect(  x0              = start, 
                                        x1              = end,
                                        fillcolor       = "#17becf", 
                                        opacity         = 0.2,
                                        layer           = "below", 
                                        line_width      = 0,
                                        annotation_text = event
                           
                        )
            else:
                aiLogger.warning("Please add which type of making event!!!")
        



        # ----------------------
        # Update layout and show
        fig.update_layout(title="Results",xaxis_title='Index',yaxis_title='Value',height=600, width=20000)
        fig.show() 

    
        
        