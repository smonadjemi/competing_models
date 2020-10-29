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
import met


def baseline_simulate(data, ui_data, location_variable_names, dimensions):

    location_x_var_name, location_y_var_name = location_variable_names

    """
    Build interaction/data models for all time steps
    """
    interaction_models = {dim: {t: None for t in range(len(ui_data) + 1)} for dim in dimensions}
    data_models = {dim: None for dim in dimensions}

    # the interaction model at t=0
    location_var_name = 'location'
    type_var_name = 'type'

    type_categories = np.unique(data[type_var_name])
    alpha = np.ones(type_categories.shape)

    # the static data model (does not change over time)
    d_reshaped = np.array([list(item) for item in data[[location_x_var_name, location_y_var_name]]])

    categories = np.unique(data[type_var_name])
    l = len(data[type_var_name])
    c_data = collections.Counter(data[type_var_name])
    proportions = np.array([c_data[t] / l for t in categories])

    printProgressBar(0, len(ui_data), prefix="Updating Models: ")
    for click_index in range(len(ui_data)):
        click = ui_data[click_index]

        loc_time_click = np.array([(click[location_x_var_name], click[location_y_var_name], click_index)],
                                  dtype=[(location_x_var_name, 'float'), (location_y_var_name, "float"),
                                         ('timstamp', "float")])
        type_click = click[type_var_name]

        printProgressBar(click_index+1, len(ui_data), prefix="Updating Models: ")

    """
    Find model evidence for all time steps
    """

    # maps time steps to a dictionary of model evidences. this will be passed to the visualization function
    model_belief_dicts = {t: {} for t in range(len(ui_data) + 1)}
    pmf_models = {t: {} for t in range(len(ui_data) + 1)}

    # the number of dimensions
    N = len(dimensions)

    # True means click models, False means data models. Embedding this into a dictionary for easy access.
    MODELS_DICTIONARY = {True: interaction_models,
                         False: data_models}

    INTERACTION_MATTERS = True
    INTERACTION_DOES_NOT_MATTER = False

    # model configurations, we have 2**n of them.
    model_configs = list(itertools.product([False, True], repeat=N))

    # in the beginning, we have uniform belief over all models.
    model_evidence_dict = {m: 1 / 2 ** N for m in model_configs}
    model_belief_dicts[0] = model_evidence_dict

    results = {}
    # loop over each time step to calculate model
    printProgressBar(0, len(ui_data), prefix="Computing Model Evidences:")
    for time_step in range(0, len(ui_data)):
        if True:
            observed_clicks = ui_data[:time_step+1]

            # compare the categorical

            categories = np.unique(data[type_var_name])
            c_ui = collections.Counter(observed_clicks[type_var_name])
            click_counts = [c_ui[t] if t in c_ui.keys() else 0 for t in categories]
            data_counts = [c_data[t] for t in categories]

            #print(click_counts, data_counts)

            expected_counts = np.array([int(np.ceil(len(observed_clicks) * proportions[list(categories).index(t)])) for t in categories])

            chisq, b_Ad_type = stats.chisquare(click_counts, f_exp=expected_counts, ddof=-1)

            #met_obj = met.Multinom(data_counts, click_counts)
            #b_Ad_type = met_obj.twosided_exact_test()


            # compare latitude
            s1_lat = observed_clicks['lat']
            #s1_lat_transformed = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(s1_lat), 1000)).fit_transform(np.reshape(s1_lat, (-1, 1)))
            #s1_lat_t_reshaped = list(np.reshape(s1_lat_transformed, (-1, )))

            s2_lat = data['lat']
            #s2_lat_transformed = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(s2_lat), 1000)).fit_transform(np.reshape(s2_lat, (-1, 1)))
            #s2_lat_t_reshaped = list(np.reshape(s2_lat_transformed, (-1, )))

            b_Ad_lat = ks_2samp(s1_lat, s2_lat, mode='exact').pvalue





            # compare longitude
            s1_lng = observed_clicks['lng']
            #s1_lng_transformed = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(s1_lng), 1000)).fit_transform(np.reshape(s1_lng, (-1, 1)))
            #s1_lng_t_reshaped = list(np.reshape(s1_lng_transformed, (-1, )))

            s2_lng = data['lng']
            #s2_lng_transformed = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(s2_lng), 1000)).fit_transform(np.reshape(s2_lng, (-1, 1)))
            #s2_lng_t_reshaped = list(np.reshape(s2_lng_transformed, (-1, )))

            b_Ad_lng = ks_2samp(s1_lng, s2_lng, mode='exact').pvalue

            to_store = {'p_type': 1 - b_Ad_type, 'p_lat': 1 - b_Ad_lat, 'p_lng': 1 - b_Ad_lng}

        else:
            to_store = {'p_type': 1, 'p_lat': 1, 'p_lng': 1}

        results[time_step] = to_store
        # print(results)

        printProgressBar(time_step, len(ui_data), prefix="Computing Model Evidences:")



    # print(pmf_models)

    return data_models, interaction_models, results




"""
some constants
"""
DATA_DIRECTORY = 'outputs/model_posteriors_37(test)'
# DATA_DIRECTORY = 'outputs/ewall_exact_2sided'
#DATA_DIRECTORY = 'outputs/ewall_chisq_8df'
MODEL_NAMES = {(True, True): "mixed",
               (True, False): "geo-based",
               (False, True): "type-based",
               (False, False): "none"}

TASK_TO_QUESTION_TO_PARTICIPANTS = {'type-based': {'q1': [7, 8, 9, 10, 11, 12, 13, 15, 16],
                                                   'q2': [1, 2, 3, 4, 5, 17, 18, 19, 20, 21, 22]},
                                    'mixed': {'q3': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                                              'q4': [1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]},
                                    'geo-based': {'q5': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                                                  'q6': [1, 2, 3, 4, 5, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]}}

LOCATION_VAR_NAMES_FOR_DATASET = {"synthetic": ('x', 'y'),
                                  "crime": ('lat', 'lng')}

DATASET = 'crime'

if DATASET == 'synthetic':
    TASK_TO_QUESTION_TO_PARTICIPANTS = {'type-based': {'q1': list(range(0, 10))},
                                        'mixed': {'q2': list(range(0, 10))},
                                        'geo-based': {'q3': list(range(0, 10))}}


"""
parameters
"""

EXPLORATION_TYPE = 'mixed'
PARTICIPANT = 20





location_variable_names = LOCATION_VAR_NAMES_FOR_DATASET[DATASET]
location_x_var_name, location_y_var_name = location_variable_names
dimensions = ['location', 'type']

"""
Load the data
"""
data = Data.load_data(DATASET)


"""
Build interaction/data models for all time steps
"""
for task in TASK_TO_QUESTION_TO_PARTICIPANTS.keys():
    ui_data = Data.load_ui_data(DATASET, task, data=data)
    # print("%d participants for task %s" % (len(ui_data), task))
    for question in TASK_TO_QUESTION_TO_PARTICIPANTS[task].keys():
        for participant in TASK_TO_QUESTION_TO_PARTICIPANTS[task][question]:
            ui_data_participant = ui_data[participant]
            print(task, question, participant)
            data_models, interaction_models, results = baseline_simulate(data, ui_data_participant, location_variable_names, dimensions)

            with open('%s/ewbaseline_%s_%s_%d.json' % (DATA_DIRECTORY, task, question, participant), 'w+') as file:
                json.dump(results, file)
            file.close()



