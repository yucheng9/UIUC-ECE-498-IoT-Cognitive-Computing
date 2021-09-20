'''a function to plot the loss value across training steps'''
import matplotlib.pyplot as plt

def function(lossValues):
    plt.plot(lossValues)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Loss")
    plt.show()