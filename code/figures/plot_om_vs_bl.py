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
DATA_DIRECTORY = 'outputs/model_posteriors_42(test)'

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




"""
Aggregated graph picking model
"""
plt.figure(figsize=plt.figaspect(0.618))
p_success_mean = np.array([np.mean(c_success[str(t)]) for t in range(len(c_success.keys()))])[1:top_k_clicks]
p_success_sd = np.array([sem(c_success[str(t)]) for t in range(len(c_success.keys()))])[1:top_k_clicks]

p_bl_success_mean = np.array([np.mean(bl_success[str(t)]) for t in range(len(bl_success.keys()))])[1:top_k_clicks]
p_bl_success_sd = np.array([sem(bl_success[str(t)]) for t in range(len(bl_success.keys()))])[1:top_k_clicks]

plt.plot(p_success_mean, label='Our Method\'s Success Rate')
plt.fill_between(p_success_mean-p_success_sd, p_success_mean+p_success_sd, alpha=0.2)

plt.plot(p_bl_success_mean, label='Wall et al. Method\'s Success Rate')
plt.fill_between(p_bl_success_mean-p_bl_success_sd, p_bl_success_mean+p_bl_success_sd, alpha=0.2)

plt.ylim([-0.1, 1.1])
plt.xlabel("# of clicks")
plt.title("Aggregate Probability of Picking Correct Model across all Tasks")
#plt.axvline(np.mean([item for question in QUESTION_TO_TASK.keys() for item in number_of_clicks[question]]), c='r', ls='--', lw=0.5)

plt.legend()

#plt.savefig('figures/%s_%d.png' % ("aggregate_picking", int(random.random()*1000)))
plt.close()



"""
aggregate graph p(model)
"""
plt.figure(figsize=plt.figaspect(0.618))
aggregate_posterior_means = np.array([np.mean(aggregate_posterior[str(t)]) for t in range(len(aggregate_posterior.keys()))])[1:top_k_clicks]
aggregate_posterior_sd = np.array([sem(aggregate_posterior[str(t)]) for t in range(len(aggregate_posterior.keys()))])[1:top_k_clicks]

plt.plot(aggregate_posterior_means, label=r'$p(\mathcal{M})$')
plt.fill_between(aggregate_posterior_means-aggregate_posterior_sd, aggregate_posterior_means+aggregate_posterior_sd, alpha=0.2)
plt.ylim([-0.1, 1.1])
plt.xlabel("# of clicks")
plt.title("Aggregate Results across all Tasks")

p_bl_success_mean = np.array([np.mean(bl_success[str(t)]) for t in range(len(bl_success.keys()))])[1:top_k_clicks]
p_bl_success_sd = np.array([sem(bl_success[str(t)]) for t in range(len(bl_success.keys()))])[1:top_k_clicks]


plt.plot(p_bl_success_mean, label='Wall et al. Success Rate')
plt.fill_between(p_bl_success_mean-p_bl_success_sd, p_bl_success_mean+p_bl_success_sd, alpha=0.2)

plt.ylim([-0.1, 1.1])
plt.xlabel("# of clicks")
#plt.axvline(np.mean([item for question in QUESTION_TO_TASK.keys() for item in number_of_clicks[question]]), c='r', ls='--', lw=0.5)

plt.legend(loc='upper left')
plt.savefig('figures/%s_%d.png' % ("aggregate_comparison", int(random.random()*1000)))
plt.close()





posterior_means = {question: {model: [np.mean(posteriors[question][str(time)][model]) for time in range(len(posteriors[question]))] for model in posteriors[question]["2"]} for question in posteriors.keys()}
posterior_sd = {question: {model: [sem(posteriors[question][str(time)][model]) for time in range(len(posteriors[question]))] for model in posteriors[question]["2"]} for question in posteriors.keys()}


bl_per_q_means = {question: [np.mean(bl_success_per_q[question][str(time)]) for time in range(len(bl_success_per_q[question]))] for question in bl_success_per_q.keys()}
bl_per_q_sd = {question: [sem(bl_success_per_q[question][str(time)]) for time in range(len(bl_success_per_q[question]))] for question in bl_success_per_q.keys()}


for question in QUESTION_TO_TASK.keys():
    plt.figure(figsize=plt.figaspect(0.618))
    task_type = QUESTION_TO_TASK[question]
    mean = np.array(posterior_means[question][task_type])[1:top_k_clicks]
    sd = np.array(posterior_sd[question][task_type])[1:top_k_clicks]

    bl_mean = np.array(bl_per_q_means[question])[1:top_k_clicks]
    bl_sd = np.array(bl_per_q_sd[question])[1:top_k_clicks]


    plt.plot(mean, label=r'$p(\mathcal{M}$=%s)'%(task_type))
    plt.fill_between(mean-sd, mean+sd, alpha=0.2)

    plt.plot(bl_mean, label='Wall et al. Success Rate')
    plt.fill_between(bl_mean - bl_sd, bl_mean + bl_sd, alpha=0.2)


    plt.ylim([-0.1, 1.1])
    plt.xlabel("# of clicks")
    plt.title("Question %s - Ground Truth: %s" % (question, task_type))
    #plt.axvline(np.mean([number_of_clicks[question]]), c='r', ls='--', lw=0.5)
    plt.legend(loc='upper left')
    plt.savefig('figures/%s_%d.png' % (question, int(random.random()*1000)))
    plt.close()

    for t in ['none', 'geo-based', 'type-based', 'mixed']:
        mean = np.array(posterior_means[question][t])[1:top_k_clicks]
        sd = np.array(posterior_sd[question][t])[1:top_k_clicks]
        plt.plot(mean, label=t)
        plt.fill_between(mean-sd, mean+sd, alpha=0.2)
    plt.ylim([-0.1, 1.1])
    plt.xlabel("# of clicks")
    plt.title("Question %s - Ground Truth: %s" % (question, task_type))
    plt.legend(loc='upper left')

    #plt.axvline(np.mean([number_of_clicks[question]]), c='r', ls='--', lw=0.5)

    # plt.savefig('figures/%s_%d.png' % (question, int(random.random()*1000)))
    plt.close()








posterior_means = {task: {model: [np.mean(list(itertools.chain.from_iterable([posteriors[question][str(time)][model] for question in TASK_TO_QUESTION_TO_PARTICIPANTS[task].keys()]))) for time in range(min([len(posteriors[q]) for q in TASK_TO_QUESTION_TO_PARTICIPANTS[task].keys()]))] for model in TASK_TO_MODELCONFIG.keys()} for task in TASK_TO_MODELCONFIG.keys()}
posterior_sd = {task: {model: [sem(list(itertools.chain.from_iterable([posteriors[question][str(time)][model] for question in TASK_TO_QUESTION_TO_PARTICIPANTS[task].keys()]))) for time in range(min([len(posteriors[q]) for q in TASK_TO_QUESTION_TO_PARTICIPANTS[task].keys()]))] for model in TASK_TO_MODELCONFIG.keys()} for task in TASK_TO_MODELCONFIG.keys()}
#posterior_sd = {question: {model: [sem(posteriors[question][str(time)][model]) for time in range(len(posteriors[question]))] for model in posteriors[question]["2"]} for question in posteriors.keys()}


#bl_per_q_means = {question: [np.mean(bl_success_per_q[question][str(time)]) for time in range(len(bl_success_per_q[question]))] for question in bl_success_per_q.keys()}
bl_per_q_means = {task: [np.mean(list(itertools.chain.from_iterable([bl_success_per_q[question][str(time)] for question in TASK_TO_QUESTION_TO_PARTICIPANTS[task].keys()]))) for time in range(min([len(bl_success_per_q[q]) for q in TASK_TO_QUESTION_TO_PARTICIPANTS[task].keys()]))]  for task in TASK_TO_MODELCONFIG.keys()}

#bl_per_q_sd = {question: [sem(bl_success_per_q[question][str(time)]) for time in range(len(bl_success_per_q[question]))] for question in bl_success_per_q.keys()}
bl_per_q_sd = {task: [sem(list(itertools.chain.from_iterable([bl_success_per_q[question][str(time)] for question in TASK_TO_QUESTION_TO_PARTICIPANTS[task].keys()]))) for time in range(min([len(bl_success_per_q[q]) for q in TASK_TO_QUESTION_TO_PARTICIPANTS[task].keys()]))]  for task in TASK_TO_MODELCONFIG.keys()}


for task in TASK_TO_QUESTION_TO_PARTICIPANTS.keys():
    plt.figure(figsize=plt.figaspect(0.618))
    mean = np.array(posterior_means[task][task])[1:top_k_clicks]
    sd = np.array(posterior_sd[task][task])[1:top_k_clicks]

    bl_mean = np.array(bl_per_q_means[task])[1:top_k_clicks]
    bl_sd = np.array(bl_per_q_sd[task])[1:top_k_clicks]


    plt.plot(mean, label=r'$p(\mathcal{M}$=%s)'%(task))
    plt.fill_between(mean-sd, mean+sd, alpha=0.2)

    plt.plot(bl_mean, label='Wall et al. Success Rate')
    plt.fill_between(bl_mean - bl_sd, bl_mean + bl_sd, alpha=0.2)


    plt.ylim([-0.1, 1.1])
    plt.xlabel("# of clicks")
    plt.title("Question %s - Ground Truth: %s" % (task, task))
    #plt.axvline(np.mean([number_of_clicks[question]]), c='r', ls='--', lw=0.5)
    plt.legend(loc='upper left')
    plt.savefig('figures/%s_%d.png' % (task, int(random.random()*1000)))
    plt.close()

    for t in ['geo-based', 'type-based', 'mixed']:
        mean = np.array(posterior_means[task][t])[1:top_k_clicks]
        sd = np.array(posterior_sd[task][t])[1:top_k_clicks]
        plt.plot(mean, label=t)
        plt.fill_between(mean-sd, mean+sd, alpha=0.2)
    plt.ylim([0, 1.1])
    plt.xlim(left=2)
    plt.xlabel("# of clicks")
    plt.title("Question %s - Ground Truth: %s" % (task, task))
    plt.legend(loc='upper left')

    #plt.axvline(np.mean([number_of_clicks[question]]), c='r', ls='--', lw=0.5)

    # plt.savefig('figures/%s_%d.png' % (question, int(random.random()*1000)))
    plt.close()
