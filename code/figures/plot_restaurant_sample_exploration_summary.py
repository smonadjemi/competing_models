import sys
sys.path.append('../implementation')
# uncommon way to import, but done for easy organization of files

import Data
import MetaModel
import MetaData
import json
import random
import numpy as np
from ncp_utils import predict_interactions_within_top, aggregate_interaction_predictions
import matplotlib.pyplot as plt
from matplotlib import gridspec
import collections
from my_cmap import get_my_cmap
from svgpath2mpl import parse_path

MOUSE_ICON = """M302.189 329.126H196.105l55.831 135.993c3.889 9.428-.555 19.999-9.444 23.999l-49.165 21.427c-9.165 4-19.443-.571-23.332-9.714l-53.053-129.136-86.664 89.138C18.729 472.71 0 463.554 0 447.977V18.299C0 1.899 19.921-6.096 30.277 5.443l284.412 292.542c11.472 11.179 3.007 31.141-12.5 31.141z"""

def plot_data():
    eight_colors = ['e41a1c', '377eb8', '4daf4a', '984ea3', 'd7c298', 'ff7f00', 'a65628', 'f781bf']
    eight_colors = [tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4)) for h in eight_colors]

    colors = eight_colors

    categories = np.unique(data['type'])

    # proportions in full data
    l = len(data['type'])
    c = collections.Counter(data['type'])
    proportions = np.array([c[t] / l for t in categories])

    colors_dict = {t: c for t, c in zip(categories, colors)}
    colors = [list(colors_dict[t]) for t in data['type']]

    fig = plt.figure(figsize=(6, 9))
    gs = fig.add_gridspec(3, 2)

    eps_lat = (max(data['lat']) - min(data['lat'])) / 50
    eps_lng = (max(data['lng']) - min(data['lng'])) / 50
    xs = np.linspace(0, 1, 100)
    ys = np.linspace(0, 1, 100)
    p = [[0 for x in xs] for y in
         ys]

    ax_loc = fig.add_subplot(gs[0:2, 0:2])
    ax_loc.scatter(data['lat'], data['lng'], s=500, c=colors, marker='o')
    ax_loc.imshow(p, extent=(min(xs), max(xs), min(ys), max(ys)), aspect='equal', alpha=0.8, origin='lower',
                  cmap=get_my_cmap())

    ax_loc.set_title(f'location - full data', fontsize=25)
    ax_loc.set_xlabel('latitude', fontsize=20)
    ax_loc.set_ylabel('longitude', fontsize=20)
    ax_loc.set_ylim([0, 1])
    ax_loc.set_xlim([0, 1])
    ax_loc.set_xticks([])
    ax_loc.set_yticks([])
    ax_loc.spines['right'].set_visible(False)
    ax_loc.spines['top'].set_visible(False)
    ax_loc.spines['left'].set_visible(False)
    ax_loc.spines['bottom'].set_visible(False)

    ax_type = fig.add_subplot(gs[2, 0:2])
    ax_type.set_title(f'type - full data', fontsize=25)
    ax_type.spines['right'].set_visible(False)
    ax_type.spines['top'].set_visible(False)
    ax_type.set_yticks([0, 0.5, 1])

    dmodel = models[task_type].d_models['type']
    ax_type.bar(categories, proportions, color=eight_colors)
    ax_type.set_ylim(0, 1)
    ax_type.xaxis.set_tick_params(rotation=90, labelsize=14)

    plt.savefig('figures/%s_%d.png' % ('full_data', int(random.random()*1000)), bbox_inches = 'tight',
    pad_inches = 0.2)
    plt.show()

def plot(t):
    eight_colors = ['e41a1c', '377eb8', '4daf4a', '984ea3', 'd7c298', 'ff7f00', 'a65628', 'f781bf']
    eight_colors = [tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)) for h in eight_colors]

    colors = eight_colors

    categories = np.unique(data['type'])
    # proportions in full data
    l = len(data['type'])
    c = collections.Counter(data['type'])
    proportions = np.array([c[t] / l for t in categories])

    colors_dict = {t: c for t, c in zip(categories, colors)}
    colors = [list(colors_dict[t]) for t in data['type']]

    fig = plt.figure(figsize=(6, 10))
    gs = fig.add_gridspec(3, 2)
    #gs.update(hspace=0.7)

    cmodel = models[task_type].c_model
    eps_lat = (max(data['lat']) - min(data['lat']))/50
    eps_lng = (max(data['lng']) - min(data['lng'])) / 50
    xs = np.linspace(0, 1, 100)
    ys = np.linspace(0, 1, 100)
    p = [[cmodel.get_probability(np.array([(x, y)], dtype=[('lat', 'float'), ('lng', 'float')])) for y in ys] for x in xs]

    ax_loc = fig.add_subplot(gs[0:2, 0:2])
    ax_loc.scatter(data['lng'], data['lat'], s=600, c=colors)

    mouse = parse_path(MOUSE_ICON)
    mouse.vertices[:, 1] *= -1
    # mouse.vertices -= mouse.vertices.mean(axis=0)

    ax_loc.scatter(this_ui_data['lng'][:t], this_ui_data['lat'][:t], marker=mouse, c='black', s=1500)
    ax_loc.set_title(f'location distribution at t={t}', fontsize=25)
    ax_loc.set_xlabel('latitude', fontsize=20)
    ax_loc.set_ylabel('longitude', fontsize=20)
    ax_loc.set_xticks([])
    ax_loc.set_yticks([])
    ax_loc.set_ylim([0, 1])
    ax_loc.set_xlim([0, 1])
    #ax_loc.set_ylim(top=38.74)
    ax_loc.spines['right'].set_visible(False)
    ax_loc.spines['top'].set_visible(False)
    ax_loc.spines['left'].set_visible(False)
    ax_loc.spines['bottom'].set_visible(False)


    #ax_loc.pcolormesh(xs, ys, p, rasterized=True, alpha=0.50, snap=False, shading='gouraud', edgecolor=(1.0, 1.0, 1.0, 0.3), linewidth=0.0015625)




    ax_loc.imshow(p, extent=(min(xs), max(xs), min(ys), max(ys)), aspect='equal', alpha=0.8, origin='lower', cmap=get_my_cmap())

    ax_type = fig.add_subplot(gs[2, 0:2])
    ax_type.set_title(f'type distribution at t={t}', fontsize=18)
    ax_type.spines['right'].set_visible(False)
    ax_type.spines['top'].set_visible(False)
    ax_type.set_yticks([0, 0.5, 1])

    dmodel = models[task_type].d_models['type']
    ax_type.bar(dmodel.categories, dmodel.mu, color=eight_colors)
    ax_type.set_ylim(0, 1)
    ax_type.xaxis.set_tick_params(rotation=45, labelsize=20)

    plt.savefig('figures/%s_%d.png' % (t, int(random.random()*1000)), bbox_inched='tight',
    pad_inches = 0.1)
    plt.show()


"""
This file runs the crime dataset with metamodel involving gaussian and categorical attributes
"""

TASK_TYPES = ['type-based', 'geo-based', 'mixed']
task_type = 'mixed'

#DATA_DIRECTORY = 'outputs/model_posteriors_42(test)'

data = Data.load_data('restaurant_sample')
#ui_data = Data.load_ui_data('crime', task_type='type-based')[10]
cattr = ['lat', 'lng']
dattr = ['type']

model_configs = {'type-based': {'c_dims': [], 'd_dims': ['type']},
                 'geo-based': {'c_dims': ['lat', 'lng'], 'd_dims': []},
                 'mixed': {'c_dims': ['lat', 'lng'], 'd_dims': ['type']},
                 'none': {'c_dims': [], 'd_dims': []}}

withins = [1, 5, 10, 20, 50, 100]

ui_data_full = Data.load_ui_data('restaurant_sample')
participant = 0
#participant 19 and 2
this_ui_data = ui_data_full
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

plot_data()
plot(0)

for interaction_index in range(len(this_ui_data)):
    interaction = this_ui_data[interaction_index]

    print(f'*** interaction {interaction_index}')
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
    print({mname: models[mname].get_log_model_evidence() - max_log_likelihood for mname in models.keys()})
    model_posteriors[interaction_index+1] = mp

    model_log_evidences[interaction_index + 1] = {mname: models[mname].get_log_model_evidence() for mname in
                                                  models.keys()}

    if interaction_index in [0, 1, 2, 3]:
        plot(interaction_index + 1)

print(model_posteriors)

