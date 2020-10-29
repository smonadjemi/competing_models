import sys
sys.path.append('../implementation')
# uncommon way to import, but done for easy organization of files

import Data
import MetaModel
import MetaData
import json
import numpy as np
from ncp_utils import predict_interactions_within_top, aggregate_interaction_predictions

"""
This file runs the crime dataset with metamodel involving gaussian and categorical attributes
"""

#TASK_TYPES = ['type-based', 'geo-based', 'mixed']


TASK_TYPES = ['full-open-ended']
DATA_DIRECTORY = 'outputs/sample_restaurant'

data = Data.load_data('restaurant_sample')
#ui_data = Data.load_ui_data('crime', task_type='type-based')[10]
cattr = ['lat', 'lng']
dattr = ['type']

model_configs = {'type-based': {'c_dims': [], 'd_dims': ['type']},
                 'geo-based': {'c_dims': ['lat', 'lng'], 'd_dims': []},
                 'mixed': {'c_dims': ['lat', 'lng'], 'd_dims': ['type']},
                 'none': {'c_dims': [], 'd_dims': []}}

withins = [1, 5, 10, 20, 50, 100]


''' loop over every session '''

this_ui_data = Data.load_ui_data('restaurant_sample')

models = {model_name: MetaModel.CombinedMetaModel(data, model_configs[model_name]['c_dims'],
                                                  model_configs[model_name]['d_dims'], model_name) for
          model_name in model_configs.keys()}

"""variables to save results"""
model_posteriors = {t: None for t in range(len(this_ui_data)+1)}
model_log_evidences = {t: None for t in range(len(this_ui_data) + 1)}
ncp_raw_results = {t: None for t in range(1, len(this_ui_data)+1)}
# print(hover)

sum_of_likelihoods = sum([models[m].get_model_evidence() for m in models.keys()])
mp = {mname: models[mname].get_model_evidence() / sum_of_likelihoods for mname in models.keys()}
model_posteriors[0] = mp

model_log_evidences[0] = {mname: models[mname].get_log_model_evidence() for mname in models.keys()}

for interaction_index in range(len(this_ui_data)):
    interaction = this_ui_data[interaction_index]

    """
    was interaction predicted given current state of models?
    """
    ncp = predict_interactions_within_top(withins, models, interaction)
    ncp_raw_results[interaction_index + 1] = ncp

    for model in models:
        models[model].update(interaction)

    max_log_likelihood = max([models[m].get_log_model_evidence() for m in models.keys()])
    sum_of_likelihoods = sum([np.exp(models[m].get_log_model_evidence() - max_log_likelihood) for m in models.keys()])
    mp = {mname: np.exp(models[mname].get_log_model_evidence() - max_log_likelihood) / sum_of_likelihoods for mname in models.keys()}
    # print(mp)
    model_posteriors[interaction_index+1] = mp
    predict_interactions_within_top(withins, models, interaction)
    model_log_evidences[interaction_index + 1] = {mname: models[mname].get_log_model_evidence() for mname in
                                                  models.keys()}

"""save results to files"""

with open('%s/%s_%s_%d.json' % (DATA_DIRECTORY, 0, 0, 0), 'w+') as file:
    json.dump(model_posteriors, file)
file.close()




with open('%s/loglikelihood_%s_%s_%d.json' % (DATA_DIRECTORY, 0, 0, 0), 'w+') as file:
    json.dump(model_log_evidences, file)
file.close()




