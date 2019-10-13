import numpy as np
import pandas as pd
from BasicML import *


# Data-Collection

df = getdata.gdata('ex1data1.csv', name=['Area', 'Prices'])
df = pd.DataFrame(df)


# Data-PreProcessing

df = pd.DataFrame(df, dtype='float')
m = len(df)
print("m = ", m)


# Data-Selection
ind=[]
for i in range(0,m):
    ind.append(i)
one = pd.DataFrame(np.ones(shape=(m, 1)), columns=['One'], index=ind)
df = pd.concat([one, df.loc[:, ['Area', 'Prices']]], axis=1)
#print(df)


# Data-Visualisation
visdata.plot_data([df['Area'], df['Prices']], m='ro', xl='Area', yl='Prices', fig=1)

# Hypothesis Function h(x) = t0 + t1.x

X = df[['One','Area']].values
Y = df['Prices'].values
Y = np.reshape(Y, (Y.shape[0],1))
#print("X=", X.shape, "\tY=", Y.shape, "\tt=", t.shape)

mod = Model.LinearRegression(X, Y, d=3)

cost = mod.train(alpha=0.00000035, n=2000, batch_size=X.shape[0], reg=0.03)
Y_ = mod.predict(X)
visdata.plot_cost(J=cost[:, 1], i=cost[:, 0], interval=10, fig=2)
visdata.plot_data(x=X[:,1], y=Y_, xl='Area', yl='Prices', line=True, fig=1)





# Gradient Descent
#def grad()


