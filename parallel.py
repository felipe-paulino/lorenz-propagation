from time import time
import pyopencl as cl
import pandas as pd
import numpy as np
import os

#parametros das equacoes de lorenz
sigma=10.
rho=28.
beta=8/3


#Definicao das variaveis para criacao do cubo de condicoes inicias.
lquad=10

l=lquad*2+1   #quantidades de pontos em cada dimensao 
              #(1 ponto no eixo, lquad pontos a esquerda e lquad pontos a direita)

n_estados=l**3 #quantidade de estados totais

dist=2          #distancia entre pontos

#criando cubo. x,y,z sao arrays tridimensionais (lxlxl)
x0v=np.arange(-lquad*dist,(lquad+1)*dist,dist,dtype=np.float32)
x,y,z=np.meshgrid(x0v,x0v,x0v, sparse=False)

#achatamento dos arrays 3D em arrays 1D
h_x0=x.flatten()
h_y0=y.flatten()
h_z0=z.flatten()

#definicao das variaveis de tempo
ti=0.
tf=1
t=ti
timeStep=0.01
numIter=(int)((tf-ti)/timeStep) 

#nome dos arquivos para os quais serao exportados os dados
FL_RT = 'run_time.csv' #arquivo para os tempos de execucao
FILENAME='trajectories_parallel.csv' #arquivos para as trajetorias

#cria dataframe para os tempos de execucao
df_rt = pd.DataFrame({"version": [], "execution_time": []})

#definicao de ambiente opencl
platforms = cl.get_platforms()
context = cl.Context(
        dev_type=cl.device_type.ALL,
        properties=[(cl.context_properties.PLATFORM, platforms[0])]) #0=nvidia #1=intel

queue = cl.CommandQueue(context)
kernelsource = open("prop.cl").read()
program = cl.Program(context, kernelsource).build()
prop = program.prop
prop.set_scalar_arg_dtypes([np.int32,np.int32,np.float32,np.float32,np.float32,np.float32, None, None, None])

#definicao do numero de repeticoes para calculo de media e desvio padrao
COUNT=10
#numero de passos de tempo por work item
numStepsWI=20
#numero de execucoes no dispositivo opencl
numQ=numIter//numStepsWI

for k in range (COUNT):
    #criando data frame para as trajetorias
    tdf=np.full(n_estados,0.0)
    df = pd.DataFrame({"time": tdf ,"x" : h_x0, "y" : h_y0, "z" : h_z0})
    
    #inicia medicao de tempo
    start_time = time()
    
    #definicao dos buffers do host
    h_x=np.concatenate([np.zeros((numStepsWI-1)*n_estados).astype(np.float32),h_x0])
    h_y=np.concatenate([np.zeros((numStepsWI-1)*n_estados).astype(np.float32),h_y0])
    h_z=np.concatenate([np.zeros((numStepsWI-1)*n_estados).astype(np.float32),h_z0])
    
    #definicao dos buffers do dispositivo
    d_x = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_x)
    d_y = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_y)
    d_z = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_z)
    
    #A cada iteracao sao executados numStepsWI passos de tempo pos cada work item
    for j in range(numQ):
        prop(queue, (n_estados,), None,numStepsWI, n_estados, sigma,rho,beta,timeStep, d_x,d_y, d_z)
        queue.finish()
        cl.enqueue_copy(queue, h_x, d_x)
        cl.enqueue_copy(queue, h_y, d_y)
        cl.enqueue_copy(queue, h_z, d_z)
        
        #concatenando os numStepsWI passos de tempo calculados no dataframe
        tdf = np.array([np.ones(n_estados)*((i+1)+(j*numStepsWI))*timeStep for i in range(numStepsWI)]).flatten()
        df2 = pd.DataFrame({"time": tdf ,"x" : h_x, "y" : h_y, "z" : h_z})
        df=pd.concat([df,df2])
    #termina medicao de tempo
    run_time = time() - start_time
    df_rt = df_rt.append(pd.DataFrame({"version": [str(numStepsWI)+" time steps per work item"], 
                                     "execution_time": [run_time]}))
    print(run_time)
#exporta dados
df_rt.to_csv(FL_RT, index = False, header = True)
df.to_csv(FILENAME)

