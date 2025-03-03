import numpy as np
import pandas as pd
import pickle
import sys
from matplotlib import pyplot as plt
import random

def list_to_dict(l):
    d = {}
    for i, el in enumerate(l):
        c = el.split('_')[0]
        d[c] = i
    return d

# Load 2D/3D aero condition data
fname = 'data/2D/aero_conditions.pkl'
with open(fname, 'rb') as handle:
    aero_conditions_dict_2D = pickle.load(handle)

fname = 'data/3D/aero_conditions.pkl'
with open(fname, 'rb') as handle:
    aero_conditions_dict_3D = pickle.load(handle)

cases_2D = list(aero_conditions_dict_2D.keys())
cases_3D = list(aero_conditions_dict_3D.keys())

cases = list(set.intersection(set(cases_2D), set(cases_3D)))

# Check that aero conditions are same between 2D and 3D
for case in cases:
    if aero_conditions_dict_2D[case]['mach'] - aero_conditions_dict_3D[case]['mach'] > 1e-6:
        print(case)
    if aero_conditions_dict_2D[case]['reynolds'] - aero_conditions_dict_3D[case]['reynolds'] > 1e-6:
        print(case)

# Load 2D/3D aero data lists (mapping between case numbering and numpy index)
fname = 'data/2D/index_lookup.txt'
with open(fname, 'rb') as handle:
    index_lookup_2D = pickle.load(handle)
index_lookup_dict_2D = list_to_dict(index_lookup_2D)

fname = 'data/3D/index_lookup.txt'
with open(fname, 'rb') as handle:
    index_lookup_3D = pickle.load(handle)
index_lookup_dict_3D = list_to_dict(index_lookup_3D)

cases_2D = list(index_lookup_dict_2D.keys())
cases_3D = list(index_lookup_dict_3D.keys())

cases = list(set.intersection(set(cases_2D), set(cases_3D)))
random.shuffle(cases)

# Load 2D/3D aero data and save in common dict
geo_array_2D, field_array_2D = np.load('data/2D/geometry_array.npy'), np.load('data/2D/field_array.npy')
geo_array_3D, field_array_3D = np.load('data/3D/geometry_array.npy'), np.load('data/3D/field_array.npy')

print(geo_array_2D.shape, field_array_2D.shape)
print(geo_array_3D.shape, field_array_3D.shape)

train_cases, val_cases, test_cases = cases[:int(0.8*len(cases))], cases[int(0.8*len(cases)): int(0.9*len(cases))], cases[int(0.9*len(cases)):]
#train_cases, val_cases, test_cases = cases[:10], cases[10: 15], cases[15:20]

for split_cases, split_name in zip([train_cases, val_cases, test_cases], ['train', 'val', 'test']):
    TL_data_dict = {}
    for case in split_cases:
        tmp_dict = {}

        idx_2d = index_lookup_dict_2D[case]
        tmp_dict['2D'] = [geo_array_2D[idx_2d, ...], field_array_2D[idx_2d, ...]]

        idx_3d = index_lookup_dict_3D[case]
        tmp_dict['3D'] = [geo_array_3D[idx_3d, ...], field_array_3D[idx_3d, ...]]

        tmp_dict['mach'] = aero_conditions_dict_2D[case]['mach']
        tmp_dict['reynolds'] = aero_conditions_dict_2D[case]['reynolds']

        TL_data_dict[case] = tmp_dict

    fname = 'data/'+split_name+'_tl_data.pkl'
    with open(fname, 'wb') as handle:
        pickle.dump(TL_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''
for case_id in cases:

    fig, ax = plt.subplots(figsize=(12,6))

    x_2d, y_2d = TL_data_dict[case_id]['2D'][0][0, :, 0], TL_data_dict[case_id]['2D'][0][0, :, 1]
    ax.scatter(x_2d, y_2d, c='k', s=20)

    for j in range(9):
        x_3d, y_3d = TL_data_dict[case_id]['3D'][0][j, :, 0], TL_data_dict[case_id]['3D'][0][j, :, 1]
        ax.scatter(x_3d, y_3d, c='r', s=10)

    plt.savefig('images/wing_'+case_id+'.png')
    plt.close()
'''


