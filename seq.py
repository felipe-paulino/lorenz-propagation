import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from time import time
import os

#parametros das equacoes de lorenz
sigma=10.
rho=28.
beta=8/3

#equacoes de movimento
def drdt(r): 
    return np.array([
        sigma*(r[1]-r[0]),
        r[0]*(rho-r[2])-r[1],
        (r[0]*r[1])-(beta*r[2])
    ])

#definicao do runge-kutta de quarta ordem
def rk4_nextStep(r,step):
    #como dxdt retorna um vetor, k1, k2,, k3 e k4 também são vetores
    k1 = step*drdt(r)
    k2 = step*drdt(r + 0.5*k1) 
    k3 = step*drdt(r + 0.5*k2) 
    k4 = step*drdt(r + k3)
    r=r+(1/ 6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return r
     
#Definicao das variaveis para criacao do cubo de condicoes inicias.
lquad=10
l=lquad*2+1  #quantidades de pontos em cada dimensao 
             #(1 ponto no eixo, lquad pontos a esquerda e lquad pontos a direita)

n_estados=l**3  #quantidade de estados totais
dist=2          #distancia entre pontos

#criando cubo. x,y,z sao arrays tridimensionais (lxlxl)
x0v=np.arange(-lquad*dist,(lquad+1)*dist,dist,dtype='double')
x3d,y3d,z3d=np.meshgrid(x0v,x0v,x0v, sparse=False)

#achatamento dos arrays 3D em arrays 1D
x0=x3d.flatten()
y0=y3d.flatten()
z0=z3d.flatten()

#definicao das variaveis de tempo
ti=0.
tf=1
t=ti
timeStep=0.01
numIter=(int)((tf-ti)/timeStep)


#numero de execucoes
COUNT=10

#nome dos arquivos para os quais serao exportados os dados
FL_RUNTIMES='runTime_seq.csv'
FILENAME='trajectories_seq.csv'

#cria dataframe para os tempos de execucao
df_runtime = pd.DataFrame({"version": [], "execution_time": []})

for k in range (COUNT):
    #carregando vetores x,y,z com os estados iniciais
    x=np.copy(x0)
    y=np.copy(y0)
    z=np.copy(z0)
    #criando dataframe de trajetorias e carregado estados iniciais
    tdf=np.full(n_estados,0.0)
    df = pd.DataFrame({"time": tdf ,"x" : x, "y" : y, "z" : z})
    #inicia medicao do tempo
    start_time = time()
    
    #loop temporal
    for j in range (numIter):
        #loop espacial (para os estados)
        for i in range(n_estados):
            pos=np.array([x[i],y[i],z[i]])
            newPos=rk4_nextStep(pos,timeStep)
            x[i]=newPos[0]
            y[i]=newPos[1]
            z[i]=newPos[2]
        #concatena novo instante de tempo no dataframe de trajetorias
        t=(j+1)*timeStep
        tdf=np.full(l**3,t)
        df2 = pd.DataFrame({"time": tdf ,"x" : x, "y" : y, "z" : z})
        df=pd.concat([df,df2])
    #termina medicao de tempo
    run_time = time() - start_time
    #adiciona tempo de execucao ao dataframe de tempos de execucao
    df_runtime = df_runtime.append(pd.DataFrame({"version": ["sequential"], 
                                     "execution_time": [run_time]}))   
    print(run_time)

#exporta dados
df.to_csv(FILENAME)
df_runtime.to_csv(FL_RUNTIMES)





