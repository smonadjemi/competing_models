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
import Data

"""
some constants
"""
hmm_res_directory = 'outputs/adams_stl_hmm'
our_res_directory = 'outputs/adams_stl_metamodel'

all_tasks = ['full-open-ended']

full_data = Data.load_data('crime')

"""
Build interaction/data models for all time steps
"""

ncp_results_hmm = {}
ncp_results_our = {}
within_nums = [1, 5, 10, 20, 50, 100]
for task in all_tasks:
    # print("%d participants for task %s" % (len(ui_data), task))
    #ui_data = Data.load_ui_data('boardrooms', group=task)
    ui_data = Data.load_ui_data('crime', task_type='full-open-ended')
    ncp_results_hmm[task] = {w: [] for w in within_nums}
    ncp_results_our[task] = {w: [] for w in within_nums}
    for participant in ui_data.keys():

        if len(ui_data[participant]) < 3:
            continue

        print(f'task {task}; participant {participant}')

        with open('%s/ncp_%s_%d.json' % (hmm_res_directory, '', participant), 'r') as file:
            target_data = json.load(file)
            for w in within_nums:
                ncp_results_hmm[task][w].append(target_data[str(w)])

        with open('%s/ncp_%s_q_%d.json' % (our_res_directory, task, participant), 'r') as file:
            target_data = json.load(file)
            for w in within_nums:
                ncp_results_our[task][w].append(target_data[str(w)])


means = {task: [np.mean(np.array(ncp_results_our[task][w])[np.logical_not(np.isnan(ncp_results_hmm[task][w]))]) for w in within_nums] for task in all_tasks}
sds = {task: [sem(np.array(ncp_results_our[task][w])[np.logical_not(np.isnan(ncp_results_hmm[task][w]))]) for w in within_nums] for task in all_tasks}

hmm_mean = {task: [np.mean(np.array(ncp_results_hmm[task][w])[np.logical_not(np.isnan(ncp_results_hmm[task][w]))]) for w in within_nums] for task in all_tasks}
hmm_sd = {task: [sem(np.array(ncp_results_hmm[task][w])[np.logical_not(np.isnan(ncp_results_hmm[task][w]))]) for w in within_nums] for task in all_tasks}

#print(hmm_sd)
#print(sds)

"""
Aggregated graph
"""

n_groups = len(within_nums)


plt.rcParams['axes.axisbelow'] = True

for task in all_tasks:
    m = means[task]
    sd = sds[task]

    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 1
    #plt.grid()
    plt.rcParams['axes.axisbelow'] = True
    plt.figure(figsize=plt.figaspect(0.618))
    ax = plt.axes()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #plt.rcParams.update({'font.size': 12})

    rects1 = plt.bar(index+bar_width/2, tuple(hmm_mean[task]), bar_width,
                     alpha=opacity,
                     color='orange',
                     label='Ottley et al.',
                     yerr=tuple(np.array(hmm_sd[task])),
                     capsize=3
                     )

    rects2 = plt.bar(index + 3*bar_width/2, tuple(m), bar_width,
                     alpha=opacity,
                     #color='#7cc496',
                     label='Our Method',
                     yerr=tuple(sd),
                     capsize=3
                     )

    plt.xlabel('k', fontsize=13)
    plt.ylabel('avg accuracy', fontsize=13)
    plt.xticks(index + bar_width, ('1', '5', '10', '20', '50', '100'))
    plt.yticks((0, 0.2, 0.4, 0.6, 0.8, 1.0))
    plt.ylim([0, 1])



    plt.title("Next Interaction Prediction for S&P 500 Boardrooms Data", fontsize=15)
    plt.legend(loc='upper left', fontsize=13)



    plt.savefig('figures/ncp_%s_%d.png' % (task, int(random.random()*1000)), bbox_inches = 'tight',
    pad_inches = 0.1)

    plt.show()
    plt.close()
