import numpy as np
import pandas as pd
import pickle
import sys
from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize=(12,6))
case_id = '100'

with open('../3D_Wing_Data/3D_surface_data.pickle', 'rb') as handle:
    dataframes = pickle.load(handle)

dfs = dataframes[case_id]['000']

for i, df in enumerate(dfs):
    data = df.values
    x_3d, y_3d = data[:,0], data[:,1]
    if i==0:
        ax.scatter(x_3d, y_3d, c='r', s=10, label='3D')
    else:
        ax.scatter(x_3d, y_3d, c='r', s=10)

with open('../2D_Wing_Data/2D_surface_data.pickle', 'rb') as handle:
    dataframes = pickle.load(handle)

dfs = dataframes[case_id]['000']

data = dfs[0].values
x_2d, y_2d = data[:,0], data[:,1]
ax.scatter(x_2d, y_2d, c='k', s=10, label='2D')

ax.set_title('Raw_Data', fontsize=24)
ax.legend(fontsize=20)
plt.savefig('wing.png')
plt.close()