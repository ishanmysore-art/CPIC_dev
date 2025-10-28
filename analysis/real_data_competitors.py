import h5py, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from dca import analysis, data_util

M1 = data_util.load_sabes_data('/home/rui/Data/M1/indy_20160627_01.mat')
print("m1, input dim={}".format(M1['M1'].shape[-1]))
HC = data_util.load_kording_paper_data('/home/rui/Data/HC/example_data_hc.pickle')
print("hc, input dim={}".format(HC['neural'].shape[-1]))
weather = data_util.load_weather_data('/home/rui/Data/TEMP/temperature.csv')
print("temp, input dim={}".format(weather.shape[-1]))
ms = data_util.load_accel_data('/home/rui/Data/motion_sense/A_DeviceMotion_data/std_6/sub_19.csv')
print("ms, input dim={}".format(ms.shape[-1]))


T_pi_vals = np.array([1,2,3])

# m1_dims = np.array([5,10,20,30])
# temp_dims = np.array([3,4,5,6])
# hc_dims = np.array([5,10,15,25])

offsets = np.array([5, 10, 15])

win = 3
n_cv = 5
n_init = 5

m1_dims = np.array([5])
hc_dims = np.array([5])
temp_dims = np.array([5])
ms_dims = np.array([5])


# M1_results = analysis.run_analysis(M1['M1'], M1['cursor'], T_pi_vals, dim_vals=m1_dims, offset_vals=offsets,
#                                    num_cv_folds=n_cv, decoding_window=win, n_init=n_init, verbose=True)
good_ts = 22000
HC_results = analysis.run_analysis(HC['neural'][:good_ts], HC['loc'][:good_ts], T_pi_vals, dim_vals=hc_dims, offset_vals=offsets,
                                   num_cv_folds=n_cv, decoding_window=win, n_init=n_init, verbose=True)
# HC_results = analysis.run_analysis(HC['neural'], HC['loc'], T_pi_vals, dim_vals=hc_dims, offset_vals=offsets,
# #                                    num_cv_folds=n_cv, decoding_window=win, n_init=n_init, verbose=True)
# weather_results = analysis.run_analysis(weather, weather, T_pi_vals, dim_vals=temp_dims, offset_vals=offsets,
#                                    num_cv_folds=n_cv, decoding_window=win, n_init=n_init, verbose=True)
# ms_results = analysis.run_analysis(ms, ms, T_pi_vals, dim_vals=ms_dims, offset_vals=offsets,
#                                    num_cv_folds=n_cv, decoding_window=win, n_init=n_init, verbose=True)

import pdb; pdb.set_trace()

with open("res/m1_stochastic_infonce/result_cpt.pkl", "wb") as f:
    pickle.dump(M1_results, f)
with open("res/hc_stochastic_infonce/result_cpt.pkl", "wb") as f:
    pickle.dump(HC_results, f)
with open("res/temp_stochastic_infonce/result_cpt.pkl", "wb") as f:
    pickle.dump(weather_results, f)
with open("res/ms_stochastic_infonce/result_cpt.pkl", "wb") as f:
    pickle.dump(ms_results, f)

# visualization

import pdb; pdb.set_trace()