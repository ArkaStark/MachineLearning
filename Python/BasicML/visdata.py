import seaborn as sns
import matplotlib.pyplot as mp
from mpl_toolkits import  mplot3d
import numpy as np
import math


def plot_data(d=[], x=0, y=0, m='rx', xl='', yl='', zl='', line=False, fig=1):
    figure = mp.figure(fig)
    if len(d) == 0:
        if line:
            x = np.reshape(x, x.shape[0])
            y = np.reshape(y, y.shape[0])
            sns.lineplot(x=x, y=y, figure=figure)
        else:
            sns.scatterplot(x, y, m, figure)
    else:
        if len(d) > 2:
            print(len(d))
            ax=mp.axes(projection='3d')
            ax.scatter3D(d[0], d[1], d[2])
            if line:
                ax.plot3D(d[0], d[1], d[2], 'gray')
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_zlabel(zl)
        else:
            mp.scatter(d[0], d[1], marker=m[1], c=m[0], figure=figure)
            if line:
                mp.plot(d[0], d[1], 'b-', figure)
    mp.xlabel(xl)
    mp.ylabel(yl)
    figure.canvas.draw()


def plot_cost(Cost, interval=1, fig=1):
    fig = mp.figure(fig, figsize=(10, 10))
    cost, b, i, m = Cost['J'], Cost['batch_size'], Cost['iter'], Cost['m']
    if b:
        b = math.ceil(m/b)
        b_J = cost.flatten()
        b_J = [b_J[x] for x in range(i*b) if (x+1) % interval == 0]
        _b = [int(x+1) for x in range(i*b) if (x+1) % interval == 0]
        mp.plot(_b, b_J, label='Cost per batch')
    else:
        b = 1
    J = [cost[x, -1] for x in range(i) if (x+1) % interval == 0]
    _i = [int((x+1)*b) for x in range(i) if (x+1) % interval == 0]
    mp.plot(_i, J, linestyle='--', label='Cost per epoch')
    mp.legend()
    #mp.xticks(_i*b, np.int64(_i))
    mp.xlabel('Iterations every '+str(interval)+' interval')
    mp.ylabel('Cost')
    fig.canvas.draw()
