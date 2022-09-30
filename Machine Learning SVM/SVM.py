# Written by Cameron Cooke, November 2021

import pandas as pd
from sklearn import svm
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv("Dataset - SpiralWithCluster.csv")

xTrain = data[['x', 'y']]
yTrain = data['SpectralCluster'].astype('category')

svm_Model = svm.SVC(kernel='linear', random_state=20211111, max_iter=-1, decision_function_shape='ovr')
svm_Model.fit(xTrain, yTrain)

# determine values for the equation of the separating hyperplane
print("Intercept: ", svm_Model.intercept_)
print("Coef: ", svm_Model.coef_)

# determine misclassification rate of the dataset using this hyperplane
y_predictClass = svm_Model.predict(xTrain)
print('Misclassification rate = ', 1 - metrics.accuracy_score(yTrain, y_predictClass))

# Step 1
# separate values by predicted SpectralCluster (y_predictClass), 0 into red and 1 into blue
redX = []  # The purpose of these arrays is to make the legend work in the plot.
redY = []  # Without having separate arrays like this to do separate plt.scatter() calls for each color, the legend
blueX = []  # only ever showed one color. Perhaps there's a much simpler way to do this, but this works.
blueY = []
i = 0
for i in range(len(y_predictClass)):
    if y_predictClass[i] == 0:
        redX.append(data['x'][i])
        redY.append(data['y'][i])
    else:
        blueX.append(data['x'][i])
        blueY.append(data['y'][i])
    i += 1
# plot points
# plt.scatter(data[data.SpectralCluster == 0]['x'], data[data.SpectralCluster == 0]['y'], c='red')  # actual
# plt.scatter(data[data.SpectralCluster == 1]['x'], data[data.SpectralCluster == 1]['y'], c='blue')
plt.scatter(redX, redY, c='red')  # predicted
plt.scatter(blueX, blueY, c='blue')
# label axes
plt.xlabel("x-coordinate")
plt.ylabel("y-coordinate")
# create legend
plt.legend(["SpectralCluster 0", "SpectralCluster 1"])
# create title
plt.title("Predicted Spectral Cluster")
# draw hyperplane
x = np.linspace(-5, 5, num=100)
y = -0.1622687 * x + 0.0033450
plt.plot(x, y, c='black', linestyle=':')
# add gridlines
plt.grid()
# set plot size
plt.xlim(-4.5, 4.5)
plt.ylim(-4.5, 4.5)
# show plot
plt.show()

# Step 2
# express data in polar coordinates
radii = []
thetas = []
print("Data in polar coordinates:")
for x, y in zip(data['x'], data['y']):
    radii.append(np.sqrt(x**2 + y**2))
    thetas.append(np.arctan2(y, x))
data['radius'] = radii
data['theta'] = thetas
print(data[['radius', 'theta']])

# plot data in polar coordinates
# clear old plot data
plt.clf()
# plot points
plt.scatter(data[data.SpectralCluster == 0]['radius'], data[data.SpectralCluster == 0]['theta'], c='red')
plt.scatter(data[data.SpectralCluster == 1]['radius'], data[data.SpectralCluster == 1]['theta'], c='blue')
# label axes
plt.xlabel("radius")
plt.ylabel("angle theta")
# create legend
plt.legend(["SpectralCluster 0", "SpectralCluster 1"])
# create title
plt.title("Spectral Cluster - Polar Coordinates")
# add gridlines
plt.grid()
# show plot
plt.show()

# Step 3
groupTemp = [0 for x in range(len(data['theta']))]
# figure out group variable
for i in range(len(data['theta'])):
    if data['SpectralCluster'][i] == 0:
        if data['theta'][i] > -0.1:
            groupTemp[i] = 1
        else:
            groupTemp[i] = 3
    else:
        if data['theta'][i] > 3.1:
            groupTemp[i] = 0
        else:
            groupTemp[i] = 2

data['group'] = groupTemp

# plot points
plt.clf()
plt.scatter(data[data.group == 0]['radius'], data[data.group == 0]['theta'], c='red')
plt.scatter(data[data.group == 1]['radius'], data[data.group == 1]['theta'], c='blue')
plt.scatter(data[data.group == 2]['radius'], data[data.group == 2]['theta'], c='green')
plt.scatter(data[data.group == 3]['radius'], data[data.group == 3]['theta'], c='black')
# label axes
plt.xlabel("radius")
plt.ylabel("angle theta")
# create legend
plt.legend(["Group 0", "Group 1", "Group 2", "Group 3"])
# create title
plt.title("Polar Coordinate Groups")
# add gridlines
plt.grid()
# show plot
plt.show()

# Step 4
# group 0 vs group 1
xTrain = data[(data.group == 0) | (data.group == 1)][['radius', 'theta']]
yTrain = data[(data.group == 0) | (data.group == 1)]['group']

svm_Model01 = svm.SVC(kernel='linear', random_state=20211111, max_iter=-1, decision_function_shape='ovr')
svm_Model01.fit(xTrain, yTrain)

print("Group 0 vs Group 1")
print("Intercept: ", svm_Model01.intercept_)
print("Coef: ", svm_Model01.coef_)

# group 1 vs group 2
xTrain = data[(data.group == 1) | (data.group == 2)][['radius', 'theta']]
yTrain = data[(data.group == 1) | (data.group == 2)]['group']

svm_Model12 = svm.SVC(kernel='linear', random_state=20211111, max_iter=-1, decision_function_shape='ovr')
svm_Model12.fit(xTrain, yTrain)

print("\nGroup 1 vs Group 2")
print("Intercept: ", svm_Model12.intercept_)
print("Coef: ", svm_Model12.coef_)

# group 2 vs group 3
xTrain = data[(data.group == 2) | (data.group == 3)][['radius', 'theta']]
yTrain = data[(data.group == 2) | (data.group == 3)]['group']

svm_Model23 = svm.SVC(kernel='linear', random_state=20211111, max_iter=-1, decision_function_shape='ovr')
svm_Model23.fit(xTrain, yTrain)

print("\nGroup 2 vs Group 3")
print("Intercept: ", svm_Model23.intercept_)
print("Coef: ", svm_Model23.coef_)

# recreate previous plot
plt.clf()
plt.scatter(data[data.group == 0]['radius'], data[data.group == 0]['theta'], c='red')
plt.scatter(data[data.group == 1]['radius'], data[data.group == 1]['theta'], c='blue')
plt.scatter(data[data.group == 2]['radius'], data[data.group == 2]['theta'], c='green')
plt.scatter(data[data.group == 3]['radius'], data[data.group == 3]['theta'], c='black')
# label axes
plt.xlabel("radius")
plt.ylabel("angle theta")
# create legend
plt.legend(["Group 0", "Group 1", "Group 2", "Group 3"])
# set title
plt.title("Polar Coordinate Groups with Hyperplanes")
# add gridlines
plt.grid()
# set plot size
plt.xlim(1, 4.5)
plt.ylim(-3.25, 3.5)

# add hyperplane 0 1
# 0.5450639t = 0.2045079 + 0.9335913r
# t = (0.2045079 + 0.9335913r) / 0.5450639
r01 = np.linspace(-0, 5, num=100)
t01 = (0.2045079 + 0.9335913 * r01) / 0.5450639
plt.plot(r01, t01, c='black', linestyle=':')

# add hyperplane 1 2
# 0.8632727t = -3.9255955 + 1.9665191r
# t = (-3.9255955 + 1.9665191r) / 0.8632727
r12 = np.linspace(-0, 5, num=100)
t12 = (-3.9255955 + 1.9665191 * r12) / 0.8632727
plt.plot(r12, t12, c='green', linestyle=':')

# add hyperplane 2 3
# 0.7981819t = -6.0563573 + 1.7801821r
# t = (-6.0563573 + 1.7801821r) / 0.7981819
r23 = np.linspace(-0, 5, num=100)
t23 = (-6.0563573 + 1.7801821 * r23) / 0.7981819
plt.plot(r23, t23, c='purple', linestyle=':')

# show plot
plt.show()

# Step 5
# set up plot points and formatting
plt.clf()
plt.scatter(data[data.SpectralCluster == 0]['x'], data[data.SpectralCluster == 0]['y'], c='red')
plt.scatter(data[data.SpectralCluster == 1]['x'], data[data.SpectralCluster == 1]['y'], c='blue')
# label axes
plt.xlabel("x-coordinate")
plt.ylabel("y-coordinate")
# create legend
plt.legend(["SpectralCluster 0", "SpectralCluster 1"])
# set title
plt.title("Cartesian Coordinate Spectral Clusters with Hyperplanes")
# add gridlines
plt.grid()
# set plot size
plt.xlim(-4.5, 4.5)
plt.ylim(-4.5, 4.5)

# add hyperplanes
#x01 = r01 * np.cos(t01)
#y01 = r01 * np.sin(t01)
#plt.plot(x01, y01, c='black', linestyle=':')

x12 = r12 * np.cos(t12)
y12 = r12 * np.sin(t12)
plt.plot(x12, y12, c='green', linestyle=':')

x23 = r23 * np.cos(t23)
y23 = r23 * np.sin(t23)
plt.plot(x23, y23, c='purple', linestyle=':')

# show plot
plt.show()
