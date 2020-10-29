import sys
sys.path.append('../implementation')
# uncommon way to import, but done for easy organization of files

import Data
import numpy as np
import met
import scipy.stats as stats
import collections
import MetaModel
import json
import matplotlib.pyplot as plt
from scipy.stats import sem


STORE_DIR = 'outputs/synth_crime_1'
full_data = Data.load_data('crime')

# list of unique categories
categories = np.unique(full_data['type'])


# data_counts is the list of frequencies for each type; to be used for MET package
c_data = collections.Counter(full_data['type'])
data_counts = [c_data[t] for t in categories]

# the proportion of each type in the full data; to be used for scipy.stats
data_proportions = np.array([c_data[t] / len(full_data) for t in categories])


cattr = ['lat', 'lng']
dattr = ['type']

model_configs = {'type-based': {'c_dims': [], 'd_dims': ['type']},
                 'geo-based': {'c_dims': ['lat', 'lng'], 'd_dims': []},
                 'mixed': {'c_dims': ['lat', 'lng'], 'd_dims': ['type']},
                 'none': {'c_dims': [], 'd_dims': []}}


aggregate_results = {k: [] for k in ['ours', 'wall_met_onesided', 'wall_met_twosided', 'wall_scipy_chisq']}

# for each session:
for st in np.unique(full_data['type']):
    session = Data.load_ui_data(dataset='crime_synth', crime_type=st)
    np.save('%s/session_%s'%(STORE_DIR, st), session)

    '''
    initiate the models
    '''
    our_method = {'p_mixed': [], 'p_type': []}
    wall_method = {'p_met_onesided': [], 'p_met_twosided': [], 'p_scipy_chisq':[]}

    models = {model_name: MetaModel.CombinedMetaModel(full_data, model_configs[model_name]['c_dims'],
                                                      model_configs[model_name]['d_dims'], model_name) for
              model_name in model_configs.keys()}

    sum_of_likelihoods = sum([models[m].get_model_evidence() for m in models.keys()])
    mp = {mname: models[mname].get_model_evidence() / sum_of_likelihoods for mname in models.keys()}



    for i in range(len(session)):

        print('Processions session %s, click %d'%(st, i+1))
        '''
        process the click and update models/variebles
        '''
        click = session[i]
        observed_clicks = session[:i+1]

        '''
        wall et al
        '''
        # returns a dictionary of {type: freq} in the clicks
        c_ui = collections.Counter(observed_clicks['type'])

        # convert the dictionary into a frequency list in the same order as categories
        click_counts = [c_ui[t] if t in c_ui.keys() else 0 for t in categories]

        # get p-value for met one-sided:
        met_obj = met.Multinom(data_counts, click_counts)
        p_onesided_met = met_obj.onesided_exact_test()

        # get p-value for met two-sided:
        p_twosided_met = met_obj.twosided_exact_test()

        # for scipy, we need expected frequencies for each category; minimum of one.
        expected_counts = np.array(
            [int(np.ceil(len(observed_clicks) * data_proportions[list(categories).index(t)])) for t in categories])

        # delta dof is -1 so we have dof=number of categories
        chisq, p_scipy = stats.chisquare(click_counts, f_exp=expected_counts, ddof=-1)

        wall_method['p_met_onesided'].append(p_onesided_met)
        wall_method['p_met_twosided'].append(p_twosided_met)
        wall_method['p_scipy_chisq'].append(p_scipy)

        '''
        Our method
        '''
        for model in models:
            models[model].update(click)

        max_log_likelihood = max([models[m].get_log_model_evidence() for m in models.keys()])
        sum_of_likelihoods = sum(
            [np.exp(models[m].get_log_model_evidence() - max_log_likelihood) for m in models.keys()])
        mp = {mname: np.exp(models[mname].get_log_model_evidence() - max_log_likelihood) / sum_of_likelihoods for mname
              in models.keys()}

        our_method['p_mixed'].append(mp['mixed'])
        our_method['p_type'].append(mp['type-based'])


    '''
    save results
    '''
    with open('%s/ours_%s.json'%(STORE_DIR, st), 'w+') as fp:
        json.dump(our_method, fp)
    fp.close()

    with open('%s/wall_%s.json'%(STORE_DIR, st), 'w+') as fp:
        json.dump(wall_method, fp)
    fp.close()

    plt.plot(np.array(range(len(wall_method['p_met_onesided'])))+1, np.array(our_method['p_mixed']) + np.array(our_method['p_type']), label='Our Method: p(type-based)+p(mixed); %s'%(st))
    plt.plot(np.array(range(len(wall_method['p_met_onesided'])))+1, 1 - np.array(wall_method['p_met_onesided']), label='Wall Method: (1-p_type) [p_met_onesided]')
    plt.plot(np.array(range(len(wall_method['p_met_onesided'])))+1, 1 - np.array(wall_method['p_met_twosided']), label='Wall Method: (1-p_type) [p_met_twosided]')
    plt.plot(np.array(range(len(wall_method['p_met_onesided'])))+1, 1 - np.array(wall_method['p_scipy_chisq']), label='Wall Method: (1-p_type) [p_scipy_chisq]')

    plt.title(st)

    plt.legend()
    plt.xlim(left=2)
    plt.xlabel('# of observed clicks')
    plt.savefig('%s/individual_%s.png'%(STORE_DIR, st))
    plt.close()

    aggregate_results['wall_met_onesided'].append(1 - np.array(wall_method['p_met_onesided']))
    aggregate_results['wall_met_twosided'].append(1 - np.array(wall_method['p_met_twosided']))
    aggregate_results['wall_scipy_chisq'].append(1 - np.array(wall_method['p_scipy_chisq']))
    aggregate_results['ours'].append(np.array(our_method['p_mixed']) + np.array(our_method['p_type']))

for k in aggregate_results.keys():
    mean = np.mean(np.array(aggregate_results[k]), axis=0)
    se = sem(np.array(aggregate_results[k]), axis=0)
    plt.plot(np.array(range(len(wall_method['p_met_onesided'])))+1, mean, label=k)
    plt.fill_between(np.array(range(len(wall_method['p_met_onesided'])))+1, mean - se, mean + se, alpha=0.2)

plt.title('Aggregate Results for 8 seesions (one for each type)')

plt.legend()
plt.xlim(left=2)
plt.xlabel('# of observed clicks')
plt.savefig('%s/aggregate.png'%(STORE_DIR))
plt.close()
