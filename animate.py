import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
import time

#arquivo contendo as trajetorias
FILENAME='trajectories_parallel.csv'
df=pd.read_csv(FILENAME)

#funcao para atualizar os dados de acordo com os passos de tempo
def update_graph(num):
    caso = int(num)
    caso=num*0.01
    data=df[df['time']==caso]
    graph.set_data (data.x, data.y)
    graph.set_3d_properties(data.z)
    title.set_text('3D Test, time={}'.format(caso))
    return title, graph, 

#definicao do ambiente de plotagem
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')

#plotagem do estado inicial
data=df[df['time']==0]
graph, = ax.plot(data.x, data.y, data.z, linestyle="", marker="o")

#animacao
ani = animation.FuncAnimation(fig, update_graph, 100, interval=40,blit=True)

#ajuste dos eixos
quadsize=60
ax = plt.axes(projection='3d')
ax.set_xlim3d(-quadsize, quadsize)
ax.set_ylim3d(-quadsize, quadsize)
ax.set_zlim3d(-quadsize, quadsize)

plt.show()
