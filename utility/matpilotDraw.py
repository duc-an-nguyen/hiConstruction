import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utility.logger import aiLogger
import numpy as np
class MatPlot(object):
    def __init__(self):
        pass
        
    def draw1D(self,yCoor  = [0,10],
                    yLabel = "Y Value",
                    xLabel = "Time Unit",
                    title  = "XYAxis"):
        plt.plot(yCoor)

        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.grid(True)
        plt.show()

    def draw2DWithEvent(self, xCoor=[0, 10],
           yCoor=[0, 10],
           yLabel="Y Value",
           xLabel="Time Unit",
           title="XYAxis",
           saveFig=False,
           direction="checkpoint/",
           events=None,
           savedName="default", 
           figsize  = (10,5)):
    
        plt.figure(figsize=figsize)
        
        # Define a list of colors for different trajectories
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta']
        
        if events is not None and len(events) > 0:
            # Ensure events are sorted
            events = sorted(events)
            
            # Start the first segment before the first event
            start_idx = 0
            
            for i, event in enumerate(events):
                # Find the index where the event occurs in xCoor
                event_idx = np.searchsorted(xCoor, event)
                
                # Plot the trajectory segment with a different color
                plt.plot(xCoor[start_idx:event_idx], yCoor[start_idx:event_idx], color=colors[i % len(colors)], label=f'Trajectory {i+1}')
                
                # Draw the event line
                plt.axvline(x=event, color='red', linestyle='--')
                
                # Update the start index for the next trajectory
                start_idx = event_idx
            
            # Plot the last segment after the last event
            plt.plot(xCoor[start_idx:], yCoor[start_idx:], color=colors[len(events) % len(colors)], label=f'Trajectory {len(events) + 1}')
        
        else:
            # If no events, plot the whole line as a single trajectory
            plt.plot(xCoor, yCoor, color='blue', label='Trajectory 1')

        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.grid(True)
        plt.legend(loc='upper left')


        # plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        # plt.tight_layout(rect=[0, 0, 0.85, 1])


        if saveFig:
            if not os.path.exists(direction):
                os.makedirs(direction)
            plt.savefig(os.path.join(direction, savedName + '.png'))
            plt.close()
        else:
            plt.show()


    def draw2D(self,xCoor       = [0,10],
                    yCoor       = [0,10],
                    yLabel      = "Y Value",
                    xLabel      = "Time Unit",
                    color       = "blue",
                    title       = "XYAxis",
                    saveFig     = False,
                    direction   = "checkpoint/",
                    events       = None,
                    savedName   = "default",
                    figsize  = (10,5)):
    
        plt.figure(figsize=figsize) 
        plt.plot(xCoor, yCoor, color=color)

        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.grid(True)

        if events is not None: 
            for event in events:
                plt.axvline(x=event, color='red', linestyle='--', label='Event')


        if saveFig:
            
            if not os.path.exists(direction):
                os.makedirs(direction)
            plt.savefig(direction + savedName + '.png')
            plt.close()
        else:
            plt.show()



        

    def draw2DMultipleY(self,
                        xCoor  = [0,10],
                        yCoor       = [[0,10],[10,1]],
                        yLabel      = "Y Value",
                        xLabel      = "Time Unit",
                        title       = "XYAxis",
                        saveFig     = False,
                        direction   = "checkpoint/",
                        savedName   = "default",
                        useLegend   = True,
                        legend      = ["line1","line2"],
                        color       = ["blue","green"],
                        yLim       = [],):
        
        plt.figure()
        for idx,yValue in enumerate(yCoor):
            plt.plot(xCoor,yValue, color=color[idx])

        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)

        if len(yLim) != 0:
            plt.ylim(yLim)
        if useLegend:
            plt.legend(legend, loc='upper left')
        plt.grid(True)
        if saveFig:
            
            if not os.path.exists(direction):
                os.makedirs(direction)
            plt.savefig(direction + savedName + '.png')
            plt.close()
        else:
            plt.show()

        
        

    # =========================
    # This function is used to draw
    # and saw values from multiple line
    def saveModelLossFig(self,values,savedNamed):
        plt.figure()
        for value in values:
            plt.plot(value)
        plt.title('Model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(savedNamed + '.png')


    def subPlot(self,
                    inX1     = None,
                    inX2     = None,
                    inX3     = None,
                    inY1     = None,
                    inY2     = None,
                    inY3     = None,
                    title1  = "Img1",
                    title2  = "Img2",
                    title3  = "Img3",
                    subPlotName = "Name",
                    numRow  = 1,
                    numCol  = 2):
        
        fig, axes = plt.subplots(numRow, numCol, figsize=(15, 5))
        axes[0].plot(inX1,inY1)
        axes[0].set_title(title1)
        axes[0].grid(True)  # Add grid
        

        axes[1].plot(inX2,inY2)
        axes[1].set_title(title2)
        axes[1].grid(True)  # Add grid
        

        axes[2].plot(inX3,inY3)
        axes[2].set_title(title3)
        axes[2].grid(True)  # Add grid        

        fig.suptitle(subPlotName, fontsize=14)
        # Adjust layout
        plt.tight_layout()
        plt.show()

    # =============================
    # Plot multiple image in subplot
    def subPlotMultiImg(self,
                        inX         = [],
                        inY         = [],
                        # inYLabel    = ["reactor","separator", "stripper"],
                        subPlotName = "Name",
                        numRow      = 2,
                        numCol      = 2,
                        color       = None,
                        xLim        = [],
                        yLim        = []
                        ):
        
        if type(inY)  is not list:
            aiLogger.error("The inY should be a list of component HI, if you use one component only, please put them into the list as well")

        # ------------------------
        # The size of figure to 
        # display
        # ------------------------
        subplotWidth    = 4  # inches
        subplotHeight   = 4  # inches

        figWidth    = numCol * subplotWidth
        figHeight   = numRow * subplotHeight

        # ------------------------
        # Create subplot
        # ------------------------
        fig, axes = plt.subplots(numRow, numCol, figsize=(figWidth,figHeight), sharey=False)
        # print("=====================", axes)
        axes = axes.flatten()


        # ------------------------
        #  Plot 1D signal without time
        # ------------------------
        if len(inX) == 0:
            for idx in range(len(inY[0])):              # Run for each hi element inside each component
                for idxComp in range(len(inY)):          # Run for each component
                    if color is None:
                        axes[idx].plot(inY[idxComp][idx])
                    else:
                        axes[idx].plot(inY[idxComp][idx],color=color[idxComp])

                axes[idx].set_title("T" + str(idx+1))
                axes[idx].grid(True)  # Add grid
        
        # ------------------------
        # Plot 2D signal with time
        # ------------------------
        else:
            for idxEachHi in range(len(inY[0])):            # Run for each hi element inside each component
                for idxComp in range(len(inY)):             # Run for each component
                    if color is None:
                        axes[idxEachHi].plot(inX[idxEachHi], inY[idxComp][idxEachHi])
                    else:
                        axes[idxEachHi].plot(inX[idxEachHi], inY[idxComp][idxEachHi],color=color[idxComp])

                axes[idxEachHi].set_title("T" + str(idxEachHi+1))
                axes[idxEachHi].grid(True)  # Add grid
                if (len(yLim) != 0):
                    axes[idxEachHi].set_ylim(yLim) 
        plt.tight_layout()
        plt.show()

    def subPlotMultiImgPlotly(self,
                        inX         = [],
                        inY         = [],
                        subPlotName = "Name",
                        numRow      = 2,
                        numCol      = 2,
                        color       = None,
                        xLim        = [],
                        yLim        = []):

        if type(inY) is not list:
            raise ValueError("The inY should be a list of component HI. If you use one component only, please put them into the list as well.")

        fig = make_subplots(rows=numRow, cols=numCol, subplot_titles=[f"T{i+1}" for i in range(len(inY[0]))])

        for idxEachHi in range(len(inY[0])):
            row = (idxEachHi // numCol) + 1
            col = (idxEachHi % numCol) + 1

            for idxComp in range(len(inY)):
                line_color = color[idxComp] if color else None

                if len(inX) == 0:
                    fig.add_trace(go.Scatter(y=inY[idxComp][idxEachHi], mode='lines', line=dict(color=line_color)), row=row, col=col)
                else:
                    fig.add_trace(go.Scatter(x=inX[idxEachHi], y=inY[idxComp][idxEachHi], mode='lines', line=dict(color=line_color)), row=row, col=col)

            if len(yLim) == 2:
                fig.update_yaxes(range=yLim, row=row, col=col)

        fig.update_layout(height=300*numRow, width=400*numCol, title_text=subPlotName, showlegend=False)
        fig.show()

        # =============================
    # Plot multiple image in subplot
    def subPlot2D(self,
                        inX         = [],
                        inY         = [],
                        inX1        = [],
                        inY1        = [],
                        subPlotName = "Name",
                        title       = [],
                        numRow      = 2,
                        numCol      = 2,
                        ):
        
        if type(inY)  is not list:
            aiLogger.error("The inY should be a list")

        # ------------------------
        # The size of figure to 
        # display
        # ------------------------
        subplotWidth    = 6  # inches
        subplotHeight   = 4  # inches

        figWidth    = numCol * subplotWidth
        figHeight   = numRow * subplotHeight

        # ------------------------
        # Create subplot
        # ------------------------
        fig, axes = plt.subplots(numRow, numCol, figsize=(figWidth,figHeight), sharey=False)
        axes = axes.flatten()


        # ------------------------
        #  Plot 1D signal without time
        # ------------------------
        if len(inX) == 0:
            for idx in range(len(inY[0])):          
                axes[idx].plot(inY[idx])

                axes[idx].set_title("T" + str(idx+1))
                axes[idx].grid(True)  # Add grid
        
        # ------------------------
        # Plot 2D signal with time
        # ------------------------
        else:
            for idxY in range(len(inY)):              
                axes[idxY].plot(inX[idxY], inY[idxY], color = "blue")
                if len(inX1) != 0:
                    axes[idxY].plot(inX1[idxY], inY1[idxY], color = "green", alpha = 0.7)

                if len(title) != 0:
                    axes[idxY].set_title(title[idxY])
                axes[idxY].grid(True)  # Add grid

                

                    # if len(title) != 0:
                    #     axes[idxY].set_title(title[idxY])
                    # axes[idxY].grid(True)  # Add grid
        fig.patch.set_facecolor('lightgrey')


        plt.tight_layout()
        plt.show()

    # def subPlotMultiImg(self,
    #                     inX        = [],
    #                     inY        = [],
    #                     subPlotName = "Name",
    #                     numRow      = 25,
    #                     numCol      = 4):
        
    #     plt.figure(figsize=(25, 4))

    #     # Plot each trajectory
    #     for trajectory in inY:
    #         plt.plot(trajectory)

    #     # Set labels and title if needed
    #     plt.xlabel('Index')
    #     plt.ylabel('Y Axis')
    #     plt.title('100 Trajectories on the Same Subplot')

    #     # Show the plot
    #     plt.show()
    # def saveFig(self,):
    #     plt.plot(xCoor, yCoor)

    #     plt.title(title)
    #     plt.xlabel(xLabel)
    #     plt.ylabel(yLabel)
    #     plt.grid(True)
    #     plt.show()




    
        
        