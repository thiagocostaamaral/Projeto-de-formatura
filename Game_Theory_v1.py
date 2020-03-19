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
    df['z_y']=df[y_column].apply(lambda y: factor*normal(y,avg_y,std_y))
    df[z_column_out]=df['z_x']*df['z_y']
    df.drop(columns =['z_x','z_y'],inplace = True)
    return df

def plot_3D(df,x_variable='x',y_variable='y',z_variable='z',datapoints = [], view_2D=False):
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
    if view_2D == False:
        plt.figure(figsize=(7,7))
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='winter', edgecolor='none', alpha=0.5)
        if len(datapoints)>0:
            ax.scatter(datapoints[:, 0], datapoints[:, 1], datapoints[:, 2], c='black', marker='o')
        ax.set_title('surface')
        ax.set_xlabel(x_variable)
        ax.set_ylabel(y_variable)
        ax.set_zlabel(z_variable)

    else:
        plt.figure(figsize=(7,7))
        ax = plt.axes()
        if len(datapoints)>0:
            ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.coolwarm)
            ax.scatter(datapoints[:, 0], datapoints[:, 1], c='black', marker='o')
        ax.set_title('surface')
        ax.set_xlabel(x_variable)
        ax.set_ylabel(y_variable)
    plt.show()

def move_score(position):
    """Function to get score of a position given by the player
    
    Arguments:
        position {[array]} -- [position of the player move]
    
    Returns:
        [float] -- [returns the score of the position given]
    """
    avg_x,std_x = 15,7
    avg_y,std_y = 38,10
    factor =200
    z1 = factor*normal(position[0],avg_x,std_x)   * factor * normal(position[1],avg_y,std_y)
    z2 = factor*normal(position[0],avg_x*2,std_x) * factor * normal(position[1],avg_y/2,std_y)
    score = z1 + z2
    return score

class player:
    def __init__(self,position, space_ranges,gambles=5,start_variable=0, factor_space = 5,score=0):
        self.position=position
        self.space_ranges=space_ranges
        self.n_of_variables = len(position)
        self.analysed_variable = start_variable
        self.score = score
        self.gambles = gambles
        self.test_positions=[]
        self.test_scores=[]
        self.fails = 0
        self.factor_space = factor_space

    def update_tests(self,game_function):
        test_positions =[]
        test_scores =[]
        start,end = self.space_ranges[self.analysed_variable][0],self.space_ranges[self.analysed_variable][1]
        step = (end-start)/(self.gambles-1)
        new_variable_positions=np.arange(start=start,stop=end+step,step=step)
        for gamble in range(self.gambles):
            new_position = self.position.copy()
            new_position[self.analysed_variable] = new_variable_positions[gamble]
            new_score = game_function(new_position)
            test_positions.append(new_position)
            test_scores.append(new_score)
        self.test_positions =test_positions
        self.test_scores = test_scores
    
    def update_variable(self):
        if self.analysed_variable + 1 >= len(self.position):
            self.analysed_variable = 0
        else:
            self.analysed_variable += 1
    
    def update_space_ranges(self):
        variable =0
        for space_range in self.space_ranges:
            new_min = self.position[variable] - (self.position[variable] - space_range[0])/self.factor_space
            new_max = self.position[variable] + (space_range[1] - self.position[variable])/self.factor_space
            self.space_ranges[variable][0] = new_min
            self.space_ranges[variable][1] = new_max
            variable +=1

    def update(self):
        initial_score = self.score
        for gamble in range(len(self.test_scores)):
            if self.test_scores[gamble] >= self.score:
                self.score = self.test_scores[gamble]
                self.position = self.test_positions[gamble]

        if self.score > initial_score:
            cprint('Player have found a better position','green')
            print('New position = ',self.score)
            print('New score = ',self.position,end='\n\n')
            self.fails = 0
        else:
            cprint('Player did not found a better position \n','red')
            self.fails +=1
        
        if self.fails >= self.n_of_variables:
            cprint('No more increase on score possible -> Player will decrease space range','yellow')
            self.update_space_ranges()
            self.fails = 0
        

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

#Now the we have our system we can worry about the players
player1 = player(position = [1,1], space_ranges = [[0,50],[0,50]], gambles=10,start_variable=1)
interations = 20

player_walk =[]
player_walk.append(player1.position + [player1.score])

for interation in range(interations):
    cprint('----------\nInteration = '+str(interation),'blue')
    player1.update_tests(move_score)
    player1.update()
    player1.update_variable()

    #Just to get path of player
    player_walk.append(player1.position + [player1.score])

#Just to show image:
player_walk=np.array(player_walk)
plot_3D(samples_df,datapoints= player_walk)
plot_3D(samples_df,datapoints= player_walk,view_2D=True)