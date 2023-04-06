""" ML Project
    Casey Burton
    RIN: 661957333
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Import the data from the csv file #
fname = r'C:\Users\casey\Desktop\Spring 2023\Machine Learning\Project\ML_Project_Data\ML_data.csv'
data = pd.read_csv(fname, sep = ';', header = 0)

# Re-format and re-shape data #
data = data.rename(columns = {data.columns[0] : 'Incident Energy [eV]'})
for i in range(1, data.shape[1]):
    data = data.rename(columns = {data.columns[i] : data.columns[i][6:]})
    data = data.rename(columns = {data.columns[i] : data.columns[i][:-5]})
    data = data.rename(columns = {data.columns[i] : data.columns[i]})

Inc_E = data['Incident Energy [eV]'] # Incident neutron energies in eV
data = data.drop(columns = 'Incident Energy [eV]')
temp = data.columns.to_numpy(dtype = np.float64)
data = data.T.values

temp = temp.reshape(temp.shape[0], 1)
data = np.concatenate((temp, data), axis = 1)

X = data[:, 0] #Temperature 'features' in deg K
X = X.reshape(X.shape[0], 1)
y = data[:, 1:] #row vector 'targets' of n cross section

model = Sequential()
model.add(Dense(612, input_shape = (1,)))
model.add(Dense(612, activation = 'relu'))
model.add(Dense(612, activation = 'relu'))
model.add(Dense(612, activation = 'relu'))
model.add(Dense(612, activation = 'relu'))
model.add(Dense(612, activation = 'relu'))
model.add(Dense(612, activation = 'relu'))
model.add(Dense(612, activation = 'relu'))
model.add(Dense(y.shape[1], activation = 'relu'))

model.compile(optimizer='adam', loss='mse')

history = model.fit(X, y, epochs = 100)

preds = model.predict([25])
print(preds)

