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
DATA_DIRECTORY = 'outputs/model_posteriors_40(test)'

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

ncp_results = {}
metrics = {}
aggregate_posterior = {}
c_success = {}
totals = {}
within_nums = [1, 5, 10, 20, 50, 100]
number_of_clicks = {q: [] for q in QUESTION_TO_TASK.keys()}
for task in TASK_TO_QUESTION_TO_PARTICIPANTS.keys():
    # print("%d participants for task %s" % (len(ui_data), task))
    ncp_results[task] = {w: [] for w in within_nums}
    for question in TASK_TO_QUESTION_TO_PARTICIPANTS[task].keys():
        for participant in TASK_TO_QUESTION_TO_PARTICIPANTS[task][question]:


            with open('%s/ncp_%s_%s_%d.json' % (DATA_DIRECTORY, task, question, participant), 'r') as file:
                target_data = json.load(file)
                for w in within_nums:
                    ncp_results[task][w].append(target_data[str(w)])


means = {task: [np.mean(ncp_results[task][w]) for w in within_nums] for task in TASK_TO_QUESTION_TO_PARTICIPANTS.keys()}
sds = {task: [sem(ncp_results[task][w]) for w in within_nums] for task in TASK_TO_QUESTION_TO_PARTICIPANTS.keys()}

print(means)
print(sds)

"""
Aggregated graph
"""

n_groups = len(within_nums)


hmm_mean = {'type-based': [0.07709171, 0.43443653, 0.59648642, 0.86257764, 0.89518634, 0.90062112],
            'mixed': [0.01871017, 0.11278508, 0.23069401, 0.44272895, 0.77588247, 0.93279414],
            'geo-based': [0.02348794, 0.17731705, 0.36416161, 0.59606553, 0.89673465, 0.99777335]}


hmm_sd = {'type-based': [0.16033362, 0.37736384, 0.30593988, 0.26888698, 0.2360989, 0.23412473],
            'mixed': [0.01899957, 0.0469673, 0.0668153, 0.13346782, 0.1363253, 0.05332704],
            'geo-based': [0.02574665, 0.14405341, 0.284131, 0.31268264, 0.093551, 0.00644303]}

plt.rcParams['axes.axisbelow'] = True

for task in TASK_TO_QUESTION_TO_PARTICIPANTS.keys():
    m = means[task]
    sd = sds[task]

    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 1
    #plt.grid()
    plt.figure(figsize=plt.figaspect(0.618))
    ax = plt.axes()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #plt.rcParams.update({'font.size': 12})
    plt.rcParams['axes.axisbelow'] = True
    rects1 = plt.bar(index+bar_width/2, tuple(hmm_mean[task]), bar_width,
                     alpha=opacity,
                     color='orange',
                     label='Ottley et al.',
                     yerr=tuple(np.array(hmm_sd[task])/len(ncp_results[task][1])),
                     capsize=3
                     )

    rects2 = plt.bar(index + 3*bar_width/2, tuple(m), bar_width,
                     alpha=opacity,
                     #color='#7cc496',
                     label='Our Method',
                     yerr=tuple(sd),
                     capsize=3
                     )

    print('Us; ', task)
    print(m)

    plt.xlabel('k', fontsize=13)
    plt.ylabel('avg accuracy', fontsize=13)
    plt.xticks(index + bar_width, ('1', '5', '10', '20', '50', '100'))
    plt.yticks((0, 0.2, 0.4, 0.6, 0.8, 1.0))
    plt.ylim([0, 1])



    plt.title("Aggregate Next Click Prediction for %s Task" % (task), fontsize=15)
    plt.legend(loc='upper left', fontsize=13)

    plt.savefig('figures/ncp_%s_%d.png' % (task, int(random.random()*1000)), bbox_inches = 'tight',
    pad_inches = 0.1)
    plt.close()
