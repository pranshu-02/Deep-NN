from data_loader import *
from neural_net import *

X,Y,X1,Y1 = load_data()
NeuralNet(X,Y,X1,Y1,[784,100,100,10],learning_rate = 0.0009, lambd = 17)
