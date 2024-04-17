import matplotlib.pyplot as plt
import os
class MatPlot(object):
    def __init__(self):
        pass
        
    def draw1D(self,yCoor  = [0,10],
                    yLabel = "Y Value",
                    xLabel = "Time Unit",
                    title  = "XYAxis"):
        # plt.figure(figsize=(100, 30))

        plt.plot(yCoor)

        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.grid(True)
        plt.show()

    def draw2D(self,xCoor       = [0,10],
                    yCoor       = [0,10],
                    yLabel      = "Y Value",
                    xLabel      = "Time Unit",
                    title       = "XYAxis",
                    saveFig     = False,
                    direction   = "checkpoint/",
                    savedName   = "default"):
        plt.figure()
        plt.plot(xCoor, yCoor)

        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.grid(True)
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
                        legend      = ["line1","line2"]):
        
        plt.figure()
        for yValue in yCoor:
            plt.plot(xCoor,yValue)

        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.legend(legend, loc='upper left')
        plt.grid(True)
        if saveFig:
            
            if not os.path.exists(direction):
                os.makedirs(direction)
            plt.savefig(direction + savedName + '.png')
            plt.close()
        else:
            plt.show()


    def draw3D(self, 
                yCoor1  = [0,10], 
                yCoor2  = [0,10],
                yCoor3  = [0,10],
                xLabel  = "X",
                yLabel  = "Y",
                zLabel  = "Z",
                title   = "3D Visualization"
               ):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the data
        scatter = ax.scatter(yCoor1,yCoor2,yCoor3
                            )

        # Add labels and title
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        ax.set_zlabel(zLabel)
        ax.set_title(title)

        # Show the plot
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

    # def saveFig(self,):
    #     plt.plot(xCoor, yCoor)

    #     plt.title(title)
    #     plt.xlabel(xLabel)
    #     plt.ylabel(yLabel)
    #     plt.grid(True)
    #     plt.show()




    
        
        