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

DATASET = 'crime'
if DATASET == 'synthetic':
    TASK_TO_QUESTION_TO_PARTICIPANTS = {'type-based': {'q1': list(range(0, 10))},
                                        'mixed': {'q2': list(range(0, 10))},
                                        'geo-based': {'q3': list(range(0, 10))}}

    QUESTION_TO_TASK = {'q1': 'type-based', 'q2': 'mixed', 'q3': 'geo-based'}

"""
Build interaction/data models for all time steps
"""


posteriors = {}
aggregate_posterior = {}
number_of_clicks = {q: [] for q in QUESTION_TO_TASK.keys()}
for task in TASK_TO_QUESTION_TO_PARTICIPANTS.keys():
    # print("%d participants for task %s" % (len(ui_data), task))
    for question in TASK_TO_QUESTION_TO_PARTICIPANTS[task].keys():
        posteriors[question] = {}
        for participant in TASK_TO_QUESTION_TO_PARTICIPANTS[task][question]:
            # print(task, question, participant)

            with open('%s/ewbaseline_%s_%s_%d.json' % (DATA_DIRECTORY, task, question, participant), 'r') as file:
                posterior_data = json.load(file)
                number_of_clicks[question].append(len(posterior_data))

                for time in posterior_data.keys():
                    if time not in posteriors[question].keys():
                        posteriors[question][time] = {m: [] for m in posterior_data[time].keys()}

                    if time not in aggregate_posterior.keys():
                        aggregate_posterior[time] = []
                    for m in posterior_data[time].keys():
                        posteriors[question][time][m].append(posterior_data[time][m])

                    #aggregate_posterior[time].append(posterior_data[time][QUESTION_TO_TASK[question]])

"""
Aggregated graph
"""
"""
aggregate_posterior_means = np.array([np.mean(aggregate_posterior[str(t)]) for t in range(len(aggregate_posterior.keys()))])
aggregate_posterior_sd = np.array([np.std(aggregate_posterior[str(t)]) for t in range(len(aggregate_posterior.keys()))])

plt.plot(aggregate_posterior_means)
plt.fill_between(list(range(len(aggregate_posterior_means))), aggregate_posterior_means-aggregate_posterior_sd, aggregate_posterior_means+aggregate_posterior_sd, alpha=0.2)
plt.ylim([-0.1, 1.1])
plt.xlabel("# of clicks")
plt.ylabel("Probability of Correct Model")
plt.title("Aggregate Probability of Correct Model across all Tasks")

plt.savefig('figures/%s_%d.png' % ("aggregate", int(random.random()*1000)))
plt.close()
"""


posterior_means = {question: {model: [np.mean(posteriors[question][str(time)][model]) for time in range(1, len(posteriors[question]))] for model in posteriors[question][str(1)]} for question in posteriors.keys()}
posterior_sd = {question: {model: [np.std(posteriors[question][str(time)][model]) for time in range(1, len(posteriors[question]))] for model in posteriors[question][str(1)]} for question in posteriors.keys()}


for question in QUESTION_TO_TASK.keys():

    task_type = QUESTION_TO_TASK[question]
    #mean = np.array(posterior_means[question][task_type])
    #sd = np.array(posterior_sd[question][task_type])
    """

    plt.plot(mean, label=task_type)
    plt.fill_between(list(range(len(mean))), mean-sd, mean+sd, alpha=0.2)
    plt.ylim([-0.1, 1.1])
    plt.xlabel("# of clicks")
    plt.ylabel(r'$p(\mathcal{M})$')
    plt.title("Question %s - Ground Truth: %s" % (question, task_type))

    # plt.savefig('figures/%s_%d.png' % (question, time_lib.time_ns()))
    plt.close()
    """
    for t in ['p_type', 'p_lat', 'p_lng']:
        mean = np.array(posterior_means[question][t])
        sd = np.array(posterior_sd[question][t])
        plt.plot(mean, label=t)
        plt.fill_between(list(range(len(mean))), mean-sd, mean+sd, alpha=0.2)
    plt.ylim([-0.1, 1.1])
    plt.xlabel("# of clicks")
    plt.ylabel(r'$p(\mathcal{M})$')
    plt.plot([0.95 for t in range(len(mean))], c='r', lw=0.5, ls='--')
    plt.title("Question %s - Ground Truth: %s" % (question, task_type))
    plt.legend()
    plt.axvline(np.mean([item for item in number_of_clicks[question]]), c='r',
                ls='--', lw=0.5)

    plt.savefig('figures/%s_%d.png' % (question, int(random.random()*1000)))
    plt.close()
