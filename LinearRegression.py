from turtle import color
import numpy as np
from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt

# Reading csv file for which we have to train model
df=pd.read_csv("MachineearningWithPython\FuelConsumptionCo2.csv")

#print("\n",df.head)# This will give an idea that how our dataset is looking

#Creating a random choice so we do not suffer from bias or train and test should be fair  
msk=np.random.rand(len(df))<0.8
train=df[msk] #Giving a 80% of data to train
test=df[~msk] #Giving a 20% of data to test

#print(np.shape(train),np.shape(test)) # confirm the size of train and test data set

#Creating a train_x,train_y,test_x and test_y for Linear Regression model
train_x,train_y,test_x,test_y=train[['ENGINESIZE']],train[['CO2EMISSIONS']],test[['ENGINESIZE']],test[['CO2EMISSIONS']]

#Plotting a graph of training set for an over all look 
plt.scatter(train_x,train_y,color='red')
#plt.show()

# Fiting the Linear Regression model for choosing best fit line with suiable intercept and coefficent
reg=linear_model.LinearRegression()
reg.fit(train_x,train_y)

# Predicting the output for test_x that what our model will preidict for test_x
predictions=reg.predict(test_x)

#ME is the cost fuction or error that how much our model is far from the actual output/test_y
me=np.mean(np.abs(test_y-predictions))
print("Mean absolute error: ",*me)

# Ploting a graph toshow that how our best fit line is looking on raining dataset
plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='blue')
plt.plot(train_x,reg.predict(train_x),'-r')
plt.show()