"""
This program stores the model parameter for every session of task as a json file
"""

import sys
sys.path.append('../implementation')
# uncommon way to import, but done for easy organization of files

import Data
import json
from simulation import simulate
#from simulation_with_libs import  simulate_with_libs
import os

"""
some constants
"""
DATA_DIRECTORY = 'outputs/model_posteriors_36'
MODEL_NAMES = {(True, True): "mixed",
               (True, False): "geo-based",
               (False, True): "type-based",
               (False, False): "none"}

TASK_TO_QUESTION_TO_PARTICIPANTS = {'type-based': {'q1': [6, 7, 8, 9, 10, 11, 12, 13, 15, 16],
                                                   'q2': [0, 1, 2, 3, 4, 5, 17, 18, 19, 20, 21, 22]},
                                    'mixed': {'q3': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                                              'q4': [1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]},
                                    'geo-based': {'q5': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                                                  'q6': [1, 2, 3, 4, 5, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]}}

LOCATION_VAR_NAMES_FOR_DATASET = {"synthetic": ('x', 'y'),
                                  "crime": ('lat', 'lng')}

DATASET = 'crime'
GT_DATASET = 'crime_gt'

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
        gt_data = Data.load_ui_data(GT_DATASET, question, data=data)[0]
        for participant in TASK_TO_QUESTION_TO_PARTICIPANTS[task][question]:
            ui_data_participant = ui_data[participant-1]
            print(task, question, participant)
            data_models, interaction_models, model_belief_dicts, pmf_models, target_analysis_dict, ncp_results_dict, sorted_points = simulate(data, ui_data_participant, gt_data, location_variable_names, dimensions)
            posterior_dict = {t: {MODEL_NAMES[k]: model_belief_dicts[t][k] for k in model_belief_dicts[t].keys()} for t in model_belief_dicts.keys()}

            with open('%s/%s_%s_%d.json' % (DATA_DIRECTORY, task, question, participant), 'w+') as file:
                json.dump(posterior_dict, file)
            file.close()

            with open('%s/target_analysis_%s_%s_%d.json' % (DATA_DIRECTORY, task, question, participant), 'w+') as file:
                json.dump(target_analysis_dict, file)
            file.close()

            with open('%s/ncp_%s_%s_%d.json' % (DATA_DIRECTORY, task, question, participant), 'w+') as file:
                json.dump(ncp_results_dict, file)
            file.close()
