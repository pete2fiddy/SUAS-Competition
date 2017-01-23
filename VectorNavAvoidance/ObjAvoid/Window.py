'''
Created on Jan 10, 2017

@author: phusisian
'''
from graphics import *
from time import sleep

class Window:


    REFRESH_TIME = .1
    def __init__(self, drawable, dim):
        self.dim = dim
        self.drawable = drawable
        self.graphWin = GraphWin("Masses", self.dim[0], self.dim[1])
        self.centerPoint = (720, 510)
        self.drawLoop()

    def getGraphWin(self):
        return self.graphWin

    def getCenterPoint(self):
        return self.centerPoint

    def drawLoop(self):
        while(True):
            self.drawable.draw(self)
            sleep(Window.REFRESH_TIME)
            coverRect = Rectangle(Point(0, 0), Point(self.dim[0], self.dim[1]))
            #coverRect.setFill("white")
            #coverRect.draw(self.graphWin)