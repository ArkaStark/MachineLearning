import seaborn as sns
import matplotlib.pyplot as mp
from mpl_toolkits import  mplot3d
import numpy as np


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


def plot_cost(J, i, interval=1, fig=1):
    figure = mp.figure(fig)
    if interval != 1:
        _J = [J[int(x-1)] for x in i if x%interval==0]
        _i = [int(x) for x in i if x%interval==0]
        sns.lineplot(_i, _J)
    else:
        sns.lineplot(i, J)
    mp.xlabel('Iterations every '+str(interval)+' interval')
    mp.ylabel('Cost')
    figure.canvas.draw()
