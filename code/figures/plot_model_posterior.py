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

"""
some constants
"""
DATA_DIRECTORY = 'outputs/sample_restaurant'

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

top_k_clicks = 16
xs = np.array(range(top_k_clicks))
posteriors = {}
aggregate_posterior = {}
c_success = {}
totals = {}
number_of_clicks = {q: [] for q in QUESTION_TO_TASK.keys()}
for task in TASK_TO_QUESTION_TO_PARTICIPANTS.keys():
    # print("%d participants for task %s" % (len(ui_data), task))
    for question in TASK_TO_QUESTION_TO_PARTICIPANTS[task].keys():
        posteriors[question] = {}
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
                        c_success[time] = 0
                        totals[time] = 0

                    totals[time] += 1
                    if task == max(posterior_data[time], key=posterior_data[time].get):
                        c_success[time] += 1
                    # print(task, max(posterior_data[time], key=posterior_data[time].get))


"""
Aggregated graph
"""

aggregate_posterior_means = np.array([np.mean(aggregate_posterior[str(t)]) for t in range(2, len(aggregate_posterior.keys()))])[:top_k_clicks]
aggregate_posterior_sd = np.array([sem(aggregate_posterior[str(t)]) for t in range(2, len(aggregate_posterior.keys()))])[:top_k_clicks]

plt.figure(figsize=plt.figaspect(0.618))
plt.plot(aggregate_posterior_means)
plt.fill_between(list(range(len(aggregate_posterior_means))), aggregate_posterior_means-aggregate_posterior_sd, aggregate_posterior_means+aggregate_posterior_sd, alpha=0.2)
plt.ylim([-0.1, 1.1])
plt.xlabel("# of clicks")
plt.ylabel("Probability of Correct Model")
plt.title("Aggregate Probability of Correct Model across all Tasks")
plt.xticks(xs)
#plt.axvline(np.mean([item for question in QUESTION_TO_TASK.keys() for item in number_of_clicks[question]]), c='r', ls='--', lw=0.5)

plt.savefig('figures/%s_%d.png' % ("aggregate", int(random.random()*1000)))
plt.close()


p_success = [c_success[str(t)]/totals[str(t)] for t in range(2, len(c_success.keys()))]
plt.figure(figsize=plt.figaspect(0.618))
plt.plot(p_success)
plt.ylim([-0.1, 1.1])
plt.xlabel("# of clicks")
plt.ylabel("Probability of Picking Correct Model")
plt.title("Aggregate Probability of Picking Correct Model across all Tasks")
#plt.axvline(np.mean([item for question in QUESTION_TO_TASK.keys() for item in number_of_clicks[question]]), c='r', ls='--', lw=0.5)

#plt.savefig('figures/%s_%d.png' % ("aggregate_picking", int(random.random()*1000)))
plt.close()


posterior_means = {question: {model: [np.mean(posteriors[question][str(time)][model]) for time in range(2, len(posteriors[question]))] for model in posteriors[question]["2"]} for question in posteriors.keys()}
posterior_sd = {question: {model: [sem(posteriors[question][str(time)][model]) for time in range(2, len(posteriors[question]))] for model in posteriors[question]["2"]} for question in posteriors.keys()}

for question in QUESTION_TO_TASK.keys():
    task_type = QUESTION_TO_TASK[question]
    mean = np.array(posterior_means[question][task_type])[:top_k_clicks]
    sd = np.array(posterior_sd[question][task_type])[:top_k_clicks]

    plt.figure(figsize=plt.figaspect(0.618))

    plt.plot(mean, label=task_type)
    plt.fill_between(list(range(len(mean))), mean-sd, mean+sd, alpha=0.2)
    plt.ylim([-0.1, 1.1])
    plt.xlabel("# of clicks")
    plt.ylabel(r'$p(\mathcal{M})$')
    plt.title("Question %s - Ground Truth: %s" % (question, task_type))

    # plt.savefig('figures/%s_%d.png' % (question, time_lib.time_ns()))
    plt.close()

    plt.figure(figsize=plt.figaspect(0.618))
    for t in ['none', 'geo-based', 'type-based', 'mixed']:
        mean = np.array(posterior_means[question][t])[:top_k_clicks]
        sd = np.array(posterior_sd[question][t])[:top_k_clicks]
        plt.plot(mean, label=t)
        plt.fill_between(list(range(len(mean))), mean-sd, mean+sd, alpha=0.2)
    plt.ylim([-0.1, 1.1])
    plt.xlabel("# of clicks")
    plt.ylabel(r'$p(\mathcal{M})$')
    plt.title("Question %s - Ground Truth: %s" % (question, task_type))
    plt.legend(loc='center right')
    plt.xticks(xs)

    #plt.axvline(np.mean([number_of_clicks[question]]), c='r', ls='--', lw=0.5)

    plt.savefig('figures/%s_%d.png' % (question, int(random.random()*1000)))
    plt.close()




"""
Marginal probability of attributes being important
"""

posterior_means = {question: {model: [np.mean(posteriors[question][str(time)][model]) for time in range(2, len(posteriors[question]))] for model in posteriors[question]["2"]} for question in posteriors.keys()}
posterior_sd = {question: {model: [sem(posteriors[question][str(time)][model]) for time in range(2, len(posteriors[question]))] for model in posteriors[question]["2"]} for question in posteriors.keys()}



posterior_means = {question: {'type': [np.mean(np.array(posteriors[question][str(time)]['type-based']) + np.array(posteriors[question][str(time)]['mixed'])) for time in range(2, len(posteriors[question]))],
                              'location': [np.mean(np.array(posteriors[question][str(time)]['geo-based']) + np.array(posteriors[question][str(time)]['mixed'])) for time in range(2, len(posteriors[question]))]}

                   for question in posteriors.keys()}


posterior_sd = {question: {'type': [sem(np.array(posteriors[question][str(time)]['type-based']) + np.array(posteriors[question][str(time)]['mixed'])) for time in range(2, len(posteriors[question]))],
                           'location': [sem(np.array(posteriors[question][str(time)]['geo-based']) + np.array(posteriors[question][str(time)]['mixed'])) for time in range(2, len(posteriors[question]))]}

                for question in posteriors.keys()}


for question in QUESTION_TO_TASK.keys():
    task_type = QUESTION_TO_TASK[question]

    plt.figure(figsize=plt.figaspect(0.618))
    for t in ['type', 'location']:
        mean = np.array(posterior_means[question][t])[:top_k_clicks]
        sd = np.array(posterior_sd[question][t])[:top_k_clicks]
        plt.plot(mean, label=t)
        plt.fill_between(list(range(len(mean))), mean-sd, mean+sd, alpha=0.2)
    plt.ylim([-0.1, 1.1])
    plt.xlabel("# of clicks")
    plt.ylabel(r'$p(attribute)$')
    plt.title("Marginal Posterior - Question %s - Ground Truth: %s" % (question, task_type))
    plt.xticks(xs)
    plt.legend(loc='center right')

    #plt.axvline(np.mean([number_of_clicks[question]]), c='r', ls='--', lw=0.5)

    plt.savefig('figures/posterior_marginal_%s_%d.png' % (question, int(random.random()*1000)))
    plt.close()