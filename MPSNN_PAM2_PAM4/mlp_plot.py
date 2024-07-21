import matplotlib.pyplot as plt
import numpy as np


class mlp_plot:
    def __init__(self):
        self.plt = plt
        self.fig = self.plt.figure(1)
        self.fig.show()
        self.fig.canvas.draw()


    def plotloss(self,train_loss, validate_loss):
        self.plt.plot(train_loss)
        self.plt.plot(validate_loss,'r--')
        self.fig.canvas.draw()
    
    def plotoutput(self, label, output):
        self.plt.plot(label,'r--')
        self.plt.plot(output, 'b-')
        self.fig.canvas.draw()

    def plotboth(self, train_loss,validate_loss,label, output):
        self.fig.clf()
        self.plt.subplot(211)
        self.plt.plot(train_loss,'b-')
        self.plt.plot(validate_loss,'r--')
        self.plt.subplot(212)
        self.plt.plot(label, 'b-')
        self.plt.plot(output, 'r--')
        self.fig.canvas.draw()


