"""
This program plots the avg model posterior across sessions for every task
"""

import sys
sys.path.append('../implementation')
# uncommon way to import, but done for easy organization of files

import json
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import sem
import itertools

"""
some constants
"""
DATA_DIRECTORY = 'outputs/model_posteriors_37(test)'

TASK_TO_QUESTION_TO_PARTICIPANTS = {'type-based': {'q1': [7, 8, 9, 10, 11, 12, 13, 15, 16],
                                                   'q2': [1, 2, 3, 4, 5, 17, 18, 19, 20, 21, 22]},
                                    'mixed': {'q3': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                                              'q4': [1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]},
                                    'geo-based': {'q5': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                                                  'q6': [1, 2, 3, 4, 5, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]}}

QUESTION_TO_TASK = {'q1': 'type-based',
                    'q2': 'type-based',
                    'q3': 'mixed',
                    'q4': 'mixed',
                    'q5': 'geo-based',
                    'q6': 'geo-based'}


TASK_TO_MODELCONFIG = {'type-based': (False, True),
                       'geo-based': (True, False),
                       'mixed': (True, True)}

DATASET = 'crime'
if DATASET == 'synthetic':
    TASK_TO_QUESTION_TO_PARTICIPANTS = {'type-based': {'q1': list(range(0, 10))},
                                        'mixed': {'q2': list(range(0, 10))},
                                        'geo-based': {'q3': list(range(0, 10))}}

    QUESTION_TO_TASK = {'q1': 'type-based', 'q2': 'mixed', 'q3': 'geo-based'}

"""
Build interaction/data models for all time steps
"""

top_k_clicks = 1300
#xs = np.array(range(1, top_k_clicks))
posteriors = {}
aggregate_posterior = {}
c_success = {}
bl_stat = {}
number_of_clicks = {q: [] for q in QUESTION_TO_TASK.keys()}
bl_success = {}
bl_success_per_q = {}

for task in TASK_TO_QUESTION_TO_PARTICIPANTS.keys():
    # print("%d participants for task %s" % (len(ui_data), task))
    for question in TASK_TO_QUESTION_TO_PARTICIPANTS[task].keys():
        posteriors[question] = {}
        bl_stat[question] = {}
        bl_success_per_q[question] = {}
        for participant in TASK_TO_QUESTION_TO_PARTICIPANTS[task][question]:
            # print(task, question, participant)

            with open('%s/%s_%s_%d.json' % (DATA_DIRECTORY, task, question, participant), 'r') as file:
                posterior_data = json.load(file)
                number_of_clicks[question].append(len(posterior_data))

                for time in posterior_data.keys():
                    if time not in posteriors[question].keys():
                        posteriors[question][time] = {m: [] for m in posterior_data[time].keys()}

                    if time not in aggregate_posterior.keys():
                        aggregate_posterior[time] = []
                    for m in posterior_data[time].keys():
                        posteriors[question][time][m].append(posterior_data[time][m])

                    aggregate_posterior[time].append(posterior_data[time][QUESTION_TO_TASK[question]])

                    if time not in c_success:
                        c_success[time] = []

                    c_success[time].append(int(task == max(posterior_data[time], key=posterior_data[time].get)))
                    # print(task, max(posterior_data[time], key=posterior_data[time].get))



            with open('%s/ewbaseline_%s_%s_%d.json' % (DATA_DIRECTORY, task, question, participant), 'r') as file:
                bl_data = json.load(file)

                for time in bl_data.keys():
                    if time not in bl_stat[question].keys():
                        bl_stat[question][time] = {m: [] for m in bl_data[time].keys()}

                    if time not in bl_success.keys():
                        bl_success[time] = []

                    if time not in bl_success_per_q[question].keys():
                        bl_success_per_q[question][time] = []

                    for m in bl_data[time].keys():
                        bl_stat[question][time][m].append(bl_data[time][m])


                    one_minus_alpha = 0.95

                    location_matters = ((bl_data[time]['p_lat'] > one_minus_alpha) and (bl_data[time]['p_lng'] > one_minus_alpha))
                    type_matters = (bl_data[time]['p_type'] > one_minus_alpha)

                    bl_success[time].append(int(TASK_TO_MODELCONFIG[task] == tuple((location_matters, type_matters))))
                    bl_success_per_q[question][time].append(int(TASK_TO_MODELCONFIG[task] == tuple((location_matters, type_matters))))




NEEDED_MODELS = {'geo-based': ['geo-based', 'mixed'], 'mixed': ['mixed'], 'type-based': ['type-based', 'mixed']}
NEEDED_PS = {'geo-based': ['p_lat', 'p_lng'], 'mixed': ['p_lat', 'p_lng', 'p_type'], 'type-based': ['p_type']}

"""
for geo-based
"""

for task_type in ['geo-based', 'type-based', 'mixed']:
    this_questions = list(TASK_TO_QUESTION_TO_PARTICIPANTS[task_type].keys())
    this_needed_models = NEEDED_MODELS[task_type]
    this_needed_ps = NEEDED_PS[task_type]

    this_marginals = {str(time): np.append(*(np.sum([posteriors[q][str(time)][model] for model in this_needed_models], axis=0) for q in this_questions)) for time in range(min([len(posteriors[q].keys()) for q in this_questions]))}
    this_bl_p = {str(time+1): np.append(*(np.prod([bl_stat[q][str(time)][p] for p in this_needed_ps], axis=0) for q in this_questions)) for time in range(min([len(bl_stat[q].keys()) for q in this_questions]))}
    plt.figure(figsize=plt.figaspect(0.618))
    ax = plt.axes()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    aggregate_posterior_means = np.array([np.mean(this_marginals[str(t)]) for t in range(len(this_marginals.keys()))])[1:]
    aggregate_posterior_sd = np.array([sem(this_marginals[str(t)]) for t in range(len(this_marginals.keys()))])[1:]

    label = ' + '.join([r'$p(\mathcal{M}_{%s})$'%(m) for m in this_needed_models])
    plt.plot(np.array(range(len(aggregate_posterior_means))) + 1, aggregate_posterior_means, label='Our Method: '+label)
    plt.fill_between(np.array(range(len(aggregate_posterior_means))) + 1, aggregate_posterior_means-aggregate_posterior_sd, aggregate_posterior_means+aggregate_posterior_sd, alpha=0.2)
    plt.ylim([0, 1.1])
    plt.xlabel("# of clicks")
    plt.title("Aggregate Bias Detection for %s Task" % task_type, fontsize=15)

    p_bl_success_mean = np.array([np.mean(this_bl_p[str(t)]) for t in range(1, len(this_bl_p.keys()))])
    p_bl_success_sd = np.array([sem(this_bl_p[str(t)]) for t in range(1, len(this_bl_p.keys()))])

    ps = [p.replace('_', '_{') + '}' for p in this_needed_ps]
    label = r' $\times$ '.join([r'$(1-%s)$' % (m) for m in ps])
    plt.plot(np.array(range(len(p_bl_success_mean)))+1, p_bl_success_mean, label='Wall et al.: '+label)
    plt.fill_between(np.array(range(len(p_bl_success_mean)))+1, p_bl_success_mean-p_bl_success_sd, p_bl_success_mean+p_bl_success_sd, alpha=0.2)

    plt.xlim((2, 12))
    #plt.xticks(list(range(2, 14, 3)))
    plt.xlabel("# of clicks")
    plt.ylabel("avg. probability of ground truth bias")
    #plt.axvline(np.mean([item for question in QUESTION_TO_TASK.keys() for item in number_of_clicks[question]]), c='r', ls='--', lw=0.5)

    plt.legend(loc='lower right', fontsize=13)
    plt.savefig('figures/%s_%s_%d.png' % ("agg", task_type, int(random.random()*1000)), bbox_inches = 'tight',
    pad_inches = 0.1)
    plt.close()


