# Script made to test Game Theory methodology 
from matplotlib import pyplot as plt
from matplotlib import cm
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from termcolor import cprint
from sklearn.metrics import mean_squared_error, r2_score
    
def normal(x,avg,std):
    pi =math.pi
    upper_term = -(x-avg)**2/(2*std**2)
    f = 1/((2*pi)**0.5 *std) * math.exp(upper_term)
    return f

def include_3dnormal(df,x_column,y_column, avg_x,std_x,avg_y,std_y,factor,z_column_out):
    df['z_x']=df[x_column].apply(lambda x: factor*normal(x,avg_x,std_x))
    df['z_y']=df[y_column].apply(lambda x: factor*normal(x,avg_y,std_y))
    df[z_column_out]=df['z_x']*df['z_y']
    df.drop(columns =['z_x','z_y'],inplace = True)
    return df

def plot_3D(df,x_variable='x',y_variable='y',z_variable='z'):
    '''The data frame should contain the name of all the 3 variables given in the function'''
    #Just to show image:
    pointer =0
    x = samples_df[x_variable].values
    y = samples_df[y_variable].values
    z = samples_df[z_variable].values
    X,Y,Z=[],[],[]
    image_factor = 1
    for i in range(n_variablesx//image_factor):
        X.append(x[pointer:pointer+n_variablesy])
        Y.append(y[pointer:pointer+n_variablesy])
        Z.append(z[pointer:pointer+n_variablesy])
        pointer += n_variablesy*image_factor

    X =np.array(X)
    Y =np.array(Y)
    Z =np.array(Z)

    plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='winter', edgecolor='none')
    ax.set_title('surface')
    ax.set_xlabel(x_variable)
    ax.set_ylabel(y_variable)
    ax.set_zlabel(z_variable)
    plt.show()

def get_score(position):
    pass

class player:
    def __init__(self,position, space_range,gambles=5,start_variable=0, score=0):
        self.position=position
        self.space_range=space_range
        self.n_of_variables = len(position)
        self.analysed_variable = start_variable
        self.score = score
        self.gambles = gambles
        self.test_positions=[]

    def update(self):
        test_positions =[]
        start,end = self.space_range[self.analysed_variable][0],self.space_range[self.analysed_variable][1]
        step = (end-start)/(self.gambles-1)
        new_variable_positions=np.arange(start=start,stop=end+step,step=step)
        for gamble in range(self.gambles):
            new_position = self.position.copy()
            new_position[self.analysed_variable] = new_variable_positions[gamble]
            test_positions.append(new_position)
        self.score =1
        self.test_positions = test_positions

 
#Define variables
n_variablesx=50
n_variablesy=50
avg_x,std_x = 15,7
avg_y,std_y = 38,10
factor =200
samples =[]
for i in range(n_variablesx):
    for j in range(n_variablesy):
        samples.append([i,j])

samples_df=pd.DataFrame(samples)
samples_df.rename(columns={0:'x',1:'y'},inplace = True )

samples_df = include_3dnormal(samples_df,'x','y', avg_x,std_x,avg_y,std_y,factor,'z1')
samples_df = include_3dnormal(samples_df,'x','y', avg_x*2,std_x,avg_y/2,std_y,factor,'z2')
samples_df['z']=samples_df['z1']+samples_df['z2']

#Just to show image:
#plot_3D(samples_df)

#Now the we have our system we can worry about the players
player1 = player(position = [25,25],space_range= [[0,50],[0,50]],gambles=10)
print(player1.test_positions)
player1.update()
print(player1.test_positions)