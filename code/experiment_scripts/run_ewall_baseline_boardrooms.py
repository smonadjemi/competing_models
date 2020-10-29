"""
Contains simulation function
"""

import sys
sys.path.append('../implementation')
# uncommon way to import, but done for easy organization of files
import numpy as np
import Models
import copy
import itertools
from scipy import stats
import collections
from utils import printProgressBar
import Data
import matlab.engine
import io
from scipy.stats import ks_2samp
import json
from sklearn.preprocessing import QuantileTransformer


def ewall_simulate(data, ui_data, dimensions, all_c_dimensions, all_d_dimensions):
    results = {}
    # loop over each time step to calculate model
    printProgressBar(0, len(ui_data), prefix="Computing Model Evidences:")
    for time_step in range(0, len(ui_data)):
        observed_clicks = ui_data[:time_step + 1]
        to_store = {}

        # process discrete attributes
        for d_attr in all_d_dimensions:
            categories = np.unique(data[d_attr])

            # proportions in full data
            l = len(data[d_attr])
            c = collections.Counter(data[d_attr])
            proportions = np.array([c[t] / l for t in categories])

            # interaction data
            c_ui = collections.Counter(observed_clicks[d_attr])
            click_counts = np.array([c_ui[t] if t in c_ui.keys() else 0 for t in categories])

            expected_counts = np.array(
                [int(np.ceil(len(observed_clicks) * proportions[list(categories).index(t)])) for t in categories])

            chisq, b_Ad_attr = stats.chisquare(click_counts, f_exp=expected_counts, ddof=-1)
            to_store[f'p_{d_attr}'] = 1 - b_Ad_attr



        # process continuous attributes
        for c_attr in all_c_dimensions:
            ui_data_sample = observed_clicks[c_attr]
            ui_data_sample_transformed = QuantileTransformer(output_distribution='normal',
                                                             n_quantiles=min(len(ui_data_sample), 1000))\
                .fit_transform(np.reshape(ui_data_sample, (-1, 1)))

            ui_data_sample_transformed_reshaped = list(np.reshape(ui_data_sample_transformed, (-1,)))

            full_data_sample = data[c_attr]
            full_data_sample_transformed = QuantileTransformer(output_distribution='normal',
                                                             n_quantiles=min(len(full_data_sample), 1000)) \
                .fit_transform(np.reshape(full_data_sample, (-1, 1)))

            full_data_sample_transformed_reshaped = list(np.reshape(full_data_sample_transformed, (-1,)))

            b_Ad_attr = ks_2samp(ui_data_sample_transformed_reshaped, full_data_sample_transformed_reshaped).pvalue
            to_store[f'p_{c_attr}'] = 1 - b_Ad_attr

        results[time_step] = to_store

    return results



"""
some constants
"""
store_directory = 'outputs/boardrooms_ewall_1'
dataset = 'boardrooms'
tasks = [1, 2, 3]

full_data = Data.load_data(dataset)
all_d_dimensions = ['industry']
all_c_dimensions = ['mktcap', 'unrelated', 'female', 'age', 'tenure', 'medianpay']
dimensions = ['mktcap', 'unrelated', 'female', 'age', 'tenure', 'medianpay', 'industry']



"""
Build interaction/data models for all time steps
"""
for task in tasks:
    ui_data_full = Data.load_ui_data(dataset, group=task)
    # print("%d participants for task %s" % (len(ui_data), task))
    for participant in ui_data_full.keys():
        ui_data = ui_data_full[participant]
        if len(ui_data) < 3:
            continue

        print(f'group {task}; participant {participant}')

        results = ewall_simulate(full_data, ui_data, dimensions, all_c_dimensions, all_d_dimensions)

        with open('%s/ewbaseline_%d_%d.json' % (store_directory, task, participant), 'w+') as file:
            json.dump(results, file)
        file.close()



