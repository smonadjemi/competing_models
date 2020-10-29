import sys
sys.path.append('../implementation')
# uncommon way to import, but done for easy organization of files

import Data
import json
import numpy as np
from itertools import chain, combinations
import MetaModel
from ncp_utils import aggregate_interaction_predictions, predict_interactions_within_top


"""
parameters
"""
DATA_DIRECTORY = './outputs/boardrooms_metamodel_1'
DATASET = 'boardrooms'
UI_DATASET = 'boardrooms'
task_type = 1
# participant = 150

overall_accuracy = {k: [] for k in [1, 5, 10, 20, 50, 100]}

data = Data.load_data(DATASET)
full_ui_data = Data.load_ui_data(UI_DATASET, group=task_type)


all_d_dimensions = ['industry']
all_c_dimensions = ['mktcap', 'unrelated', 'female', 'age', 'tenure', 'medianpay']


dimensions = ['mktcap', 'unrelated', 'female', 'age', 'tenure', 'medianpay', 'industry']
all_attr_combinations = list(chain.from_iterable(combinations(dimensions, r) for r in range(len(dimensions)+1)))



model_configs = {}

for attr_combo in all_attr_combinations:
    d_attr = [x for x in attr_combo if x in all_d_dimensions]
    c_attr = [x for x in attr_combo if x in all_c_dimensions]

    model_name = 'None'
    if d_attr or c_attr:
        model_name = '-'.join(d_attr+c_attr)

    # print(model_name, c_attr, d_attr)
    model_configs[model_name] = {'c_dims': c_attr, 'd_dims': d_attr}




withins = [1, 5, 10, 20, 50, 100]


''' loop over every session '''
for task_type in [1, 2, 3]:
    ui_data_full = Data.load_ui_data(UI_DATASET, group=task_type)
    for participant in ui_data_full.keys():
        #if task_type != 'type-based' or participant != 2:
        #    continue
        this_ui_data = ui_data_full[participant]

        if len(this_ui_data) < 3:
            continue
        # print(this_ui_data)
        print('%s %s'%(task_type, str(participant)))
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

            model_log_evidences[interaction_index + 1] = {mname: models[mname].get_log_model_evidence() for mname in
                                                          models.keys()}

        """save results to files"""

        with open('%s/%s_%d.json' % (DATA_DIRECTORY, task_type, participant), 'w+') as file:
            json.dump(model_posteriors, file)
        file.close()

        ncp_agg_results = aggregate_interaction_predictions(ncp_raw_results, withins)
        # print(ncp_agg_results)
        with open('%s/ncp_%s_%d.json' % (DATA_DIRECTORY, task_type, participant), 'w+') as file:
            json.dump(ncp_agg_results, file)
        file.close()


        with open('%s/loglikelihood_%s_%d.json' % (DATA_DIRECTORY, task_type, participant), 'w+') as file:
            json.dump(model_log_evidences, file)
        file.close()


