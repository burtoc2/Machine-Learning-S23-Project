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
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import time

# Import the data from the csv file #
fname = r'C:\Users\casey\Desktop\Spring 2023\Machine Learning\Project\ML_Project_Data\ML_data.csv' #Change to match the location of the ML_data.csv file!
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
y = data.T.values
y = y.flatten()

temp = temp.reshape(temp.shape[0], 1)

list_nums = range(temp.size)
X = np.empty((1, 2))
for T in range(temp.size):
    exec("list" + str(T+1) + " = np.array([])")
    for E in Inc_E:
        exec("list" + str(T+1) + " = np.append(" + "list" + str(T+1) + ", [temp[T][0],E])")
    exec("list" + str(T+1) + " = list" + str(T+1) + ".reshape((int(list" + str(T+1) + ".size/2),2))")
    exec('X = np.concatenate((X, list' + str(T+1) + '), axis=0)')

y = y.reshape(temp.size, Inc_E.size)
y = y.flatten()
X = np.delete(X,0,0)

model = Sequential()
model.add(Dense(128, activation = 'relu', input_dim = 2))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))
opt = keras.optimizers.Adam(learning_rate= 1.0e-3)
mse = tf.keras.losses.MeanSquaredError()
met = tf.keras.metrics.MeanSquaredError()

model.compile(loss = mse, optimizer = opt, metrics = [met])
history = model.fit(X, y, epochs = 100, batch_size = 7500)
preds = model.predict(X[:26499])

fig, ax = plt.subplots()

ax.loglog(X[:26499,1], y[:26499], label = 'Actual')
ax.loglog(X[:26499,1], preds, label = 'Predicted')
ax.set_xlabel('Neutron Energy [eV]')
ax.set_ylabel('Cross Section [b]')
ax.set_title('Total Neutron Cross Section at 25 K')
ax.legend()

plt.figure()
plt.semilogy(history.history['loss'], label = 'Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Mean Squared Error')

