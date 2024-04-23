import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy


fname = 'Reports/GKAI.html'

table = pd.read_html(fname, header=0, encoding='utf-8', keep_default_na=False)
data = table[0]
data['stan'] = (data['xGP/90']+1)

#Model class
class Model(nn.Module):
    # input layer (13 stats)-->
    # hidden layer-->
    # output
    def __init__(self, stats = 13, h1 =3, output=1):
        super().__init__()
        self.l1 = nn.Linear(stats, h1)
        self.l2 = nn.Linear(h1, output)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return (x)

torch.manual_seed(69)
model = Model()

#Test split
X = data[['Ref','Agi','Aer','Com','Han','Pos','Cmd','Kic','1v1','Thr','Ant','Cnt','Dec']]
Y = data['stan']

X = X.values
Y = Y.values

from sklearn.model_selection import train_test_split

#split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.4, random_state = 30)

#convert to tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_train)
Y_train = torch.LongTensor(Y_train)
Y_test = torch.LongTensor(Y_test)

#error
loss = nn.SmoothL1Loss()

#optimiser
optimiser = torch.optim.SGD(model.parameters(), lr=0.01)

#train model
epochs = 100
losses = []

for i in range(epochs):
    #prediction
    Y_pred = model.forward(X_train)

    #error
    #error = loss(Y_pred, Y_train)
    #losses.append(error.detetch().numpy())
    if i%99 == 0:
        print( Y_pred)
        print('train')
        print(Y_train)

    #propagation
    #optimiser.zero_grad()
    #error.backward()
    #optimiser.step()
