import sys
sys.path.append('../implementation')
# uncommon way to import, but done for easy organization of files

import Data
from simulation_rental import batch_animated_visualization
from simulation_rental import simulate
from simulation_with_libs import simulate_with_libs
import res
import json
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain, combinations
import MetaModel
from ncp_utils import aggregate_interaction_predictions, predict_interactions_within_top
import random
from sklearn.neighbors import KernelDensity
from Models import multivariate_t_pdf


def plot_data():

    # instantiate and fit the KDE model
    kde_mkcap = KernelDensity(bandwidth=25000, kernel='gaussian')
    kde_unrelated = KernelDensity(bandwidth=0.2, kernel='gaussian')
    kde_female = KernelDensity(bandwidth=0.2, kernel='gaussian')

    d_mkcap = data['mktcap']
    # d_mkcap = d_mkcap[d_mkcap[:] < 10000]

    kde_mkcap.fit(d_mkcap.reshape(-1, 1))
    kde_unrelated.fit(data['unrelated'].reshape(-1, 1))
    kde_female.fit(data['female'].reshape(-1, 1))

    # score_samples returns the log of the probability density
    logprob_mkcap = kde_mkcap.score_samples(x_mktcap.reshape(-1, 1))
    logprob_unrelated = kde_unrelated.score_samples(x_unrelated.reshape(-1, 1))
    logprob_female = kde_female.score_samples(x_female.reshape(-1, 1))


    # plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
    # plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
    # plt.ylim(-0.02, 0.22)

    fig = plt.figure(figsize=(7.5, 15))
    gs = fig.add_gridspec(3, 2)

    ax_mkcap = fig.add_subplot(gs[0, 0:2])

    ax_mkcap.set_title(f'Market Capitalization - full data')
    ax_mkcap.set_xlabel('Market Capitalization')
    # ax_mkcap.set_ylabel('longitude')
    #ax_mkcap.set_yticks([0, 0.5, 1])
    ax_mkcap.spines['right'].set_visible(False)
    ax_mkcap.spines['top'].set_visible(False)
    ax_mkcap.set_yticks([])
    ax_mkcap.set_ylabel('probability density function')
    ax_mkcap.plot(x_mktcap, np.exp(logprob_mkcap), c='C1')

    #ax_mkcap.scatter(data['mktcap'], np.zeros(len(data['mktcap']))+0.1, marker='+')

    ax_unrelated = fig.add_subplot(gs[1, 0:2])
    ax_unrelated.set_title(f'Unrelated Ratio - full data')
    ax_unrelated.set_xlabel("Unrelated Ratio")
    ax_unrelated.spines['right'].set_visible(False)
    ax_unrelated.spines['top'].set_visible(False)
    ax_unrelated.set_yticks([])
    ax_unrelated.set_ylabel('probability density function')
    #ax_unrelated.set_ylim(0, 1)
    ax_unrelated.plot(x_unrelated, np.exp(logprob_unrelated), c='C2')
    # ax_unrelated.xaxis.set_tick_params(rotation=90)
    # ax_unrelated.scatter(data['unrelated'], np.zeros(len(data['unrelated'])) + 0.1, marker='+')


    ax_female = fig.add_subplot(gs[2, 0:2])
    ax_female.set_title(f'Female Ratio - full data')
    ax_female.set_xlabel("Female Ratio")
    ax_female.spines['right'].set_visible(False)
    ax_female.spines['top'].set_visible(False)
    ax_female.set_ylabel('probability density function')
    ax_female.set_yticks([])
    #ax_female.set_ylim(0, 1)
    ax_female.plot(x_female, np.exp(logprob_female), c='C0')
    # ax_female.scatter(data['female'], np.zeros(len(data['female'])) + 0.1, marker='+')

    plt.savefig('figures/%s_%d.png' % ('full_data', int(random.random()*1000)), dpi=fig.dpi)
    plt.show()



def plot(t):

    this_model = models['mktcap-unrelated-female']

    fig = plt.figure(figsize=(7.5, 15))
    gs = fig.add_gridspec(3, 2)

    ax_mkcap = fig.add_subplot(gs[0, 0:2])

    ax_mkcap.set_title(f'Market Capitalization - interaction distribution at {t}')
    ax_mkcap.set_xlabel('Market Capitalization')
    # ax_mkcap.set_ylabel('longitude')
    #ax_mkcap.set_yticks([0, 0.5, 1])
    ax_mkcap.spines['right'].set_visible(False)
    ax_mkcap.spines['top'].set_visible(False)
    ax_mkcap.set_yticks([])
    ax_mkcap.set_ylabel('probability density function')

    y_mkcap = [this_model.c_model.get_probability_for_dims(np.array([(x)], dtype=[('mktcap', 'float')]), ['mktcap']) for x in x_mktcap]
    ax_mkcap.plot(x_mktcap, y_mkcap, c='C1')
    #ax_mkcap.scatter(data['mktcap'], np.zeros(len(data['mktcap']))+0.1, marker='+')

    ax_unrelated = fig.add_subplot(gs[1, 0:2])
    ax_unrelated.set_title(f'Unrelated Ratio - interaction distribution at {t}')
    ax_unrelated.set_xlabel("Unrelated Ratio")
    ax_unrelated.spines['right'].set_visible(False)
    ax_unrelated.spines['top'].set_visible(False)
    ax_unrelated.set_yticks([])
    ax_unrelated.set_ylabel('probability density function')

    y_unrelated = [this_model.c_model.get_probability_for_dims(np.array([(x)], dtype=[('unrelated', 'float')]), ['unrelated']) for x in x_unrelated]
    ax_unrelated.plot(x_unrelated, y_unrelated, c='C2')
    #ax_unrelated.set_ylim(0, 1)
    # ax_unrelated.xaxis.set_tick_params(rotation=90)
    # ax_unrelated.scatter(data['unrelated'], np.zeros(len(data['unrelated'])) + 0.1, marker='+')


    ax_female = fig.add_subplot(gs[2, 0:2])
    ax_female.set_title(f'Female Ratio - interaction distribution at {t}')
    ax_female.set_xlabel("Female Ratio")
    ax_female.spines['right'].set_visible(False)
    ax_female.spines['top'].set_visible(False)
    ax_female.set_ylabel('probability density function')
    ax_female.set_yticks([])

    y_female = [this_model.c_model.get_probability_for_dims(np.array([(x)], dtype=[('female', 'float')]), ['female']) for x in x_female]
    ax_female.plot(x_female, y_female, c='C0')
    #ax_female.set_ylim(0, 1)
    # ax_female.scatter(data['female'], np.zeros(len(data['female'])) + 0.1, marker='+')

    plt.savefig('figures/%s_%d.png' % (str(t), int(random.random()*1000)), dpi=fig.dpi)
    plt.show()



"""
parameters
"""
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
task_type = 1
ui_data_full = Data.load_ui_data(UI_DATASET, group=task_type)
participant = 1
#if task_type != 'type-based' or participant != 2:
#    continue
this_ui_data = ui_data_full[participant]

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


x_mktcap = np.linspace(min(data['mktcap']), max(data['mktcap']), 100)
x_unrelated = np.linspace(0, 1, 100)
x_female = np.linspace(0, 1, 100)

plot_data()
plot(0)

for interaction_index in range(len(this_ui_data)):
    interaction = this_ui_data[interaction_index]

    print(f'***interaction {interaction_index}')
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

    if interaction_index+1 in [1, 5, 10, 15, 20]:
        plot(interaction_index+1)


