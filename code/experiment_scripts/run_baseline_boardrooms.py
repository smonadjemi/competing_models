import sys
sys.path.append('../implementation')
# uncommon way to import, but done for easy organization of files

import numpy as np
from pf import pf_boardrooms
import Data
import json

# parameters (these should be tuned!)
# location_sigma  = 0.1    # p(click | particle) = N(click; particle, σ²I) (how widely should we search around a particle for the next click)
# location_d      = 0.02   # size of diffusion step for location
# mixture_d       = 0.05   # size of diffusion step for mixture weights
# flip_p          = 0.45   # probability of type flipping in diffusion step
num_particles = 1000   # number of particles
params = {'flip_p': 0.23201897342004674, 'location_d': 0.029980607481756582, 'location_sigma': 0.05186453448600459, 'mixture_d': 0.08023702221263218}

location_sigma = params['location_sigma']
location_d = params['location_d']
mixture_d = params['mixture_d']
flip_p = params['flip_p']

SAVE_DIR = 'outputs/boardrooms_hmm_1'

full_data = Data.load_data('boardrooms')
all_d_dimensions = ['industry']
all_c_dimensions = ['mktcap', 'unrelated', 'female', 'age', 'tenure', 'medianpay']
dimensions = ['mktcap', 'unrelated', 'female', 'age', 'tenure', 'medianpay', 'industry']

d_params = {attr: (max(full_data[attr]) - min(full_data[attr]))/50 for attr in all_c_dimensions}
d_mixture = 0.05
p_flip = 0.45
sigma_params = {attr: (max(full_data[attr]) - min(full_data[attr]))/10 for attr in all_c_dimensions}

# tasks (groups of participants in this case
groups = [1, 2, 3]


# k values to predict next click within the top-k clicks
within_nums = [1, 5, 10, 20, 50, 100]

results = np.zeros((len(groups), len(within_nums)))
results_mean = np.zeros((len(groups), len(within_nums)))
results_sd = np.zeros((len(groups), len(within_nums)))

for i, tasknum in enumerate(groups):

    total_len = 0
    withins = np.zeros(len(within_nums))

    results_for_task = {w: [] for w in within_nums}
    all_preds = []

    full_ui_data = Data.load_ui_data('boardrooms', group=tasknum)

    for j in full_ui_data.keys():

        if len(full_ui_data[j]) < 3:
            continue

        print(f'task {tasknum}; participant {j}')

        #pred, inds = pf(dots, location_sigma, location_d, mixture_d, flip_p, num_particles, f'../data/sessionsCorrect/task{i+1}/{j+1}.txt', True)
        pred, inds = pf_boardrooms(full_data, full_ui_data[j], dimensions, all_d_dimensions, all_c_dimensions, sigma_params, d_params, d_mixture, p_flip, num_particles)

        pred = pred[3:]
        all_preds.append(pred)

        total_len += pred.size
        for k, w in enumerate(within_nums):
            withins[k] += np.sum(pred < w)
            temp = np.sum(pred < w)/pred.size
            results_for_task[w].append(temp)

        this_res = {w: results_for_task[w][-1] for w in within_nums}
        print(this_res)

        with open(f'{SAVE_DIR}/ncp_{tasknum}_{j}.json', 'w+') as fp:
            json.dump(this_res, fp)
        fp.close()


    results[i] = withins / total_len
    results_mean[i] = np.array([np.mean(results_for_task[w]) for w in within_nums])
    results_sd[i] = np.array([np.std(results_for_task[w]) for w in within_nums])

print(results)
print(results_mean)
print(results_sd)