import sys
sys.path.append('../implementation')
# uncommon way to import, but done for easy organization of files

import matplotlib.pyplot as plt
import numpy as np
import random

our_method = {"1": 0.0, "5": 0.0, "10": 0.125, "20": 0.25, "50": 0.5, "100": 0.625}
hmm = {1: 0.0, 5: 0.0, 10: 0.14285714285714285, 20: 0.14285714285714285, 50: 0.5714285714285714, 100: 0.7142857142857143}


within_nums = [1, 5, 10, 20, 50, 100]
n_groups = len(within_nums)

hmm_mean = [hmm[x] for x in within_nums]
m = [our_method[str(x)] for x in within_nums]

index = np.arange(n_groups)
bar_width = 0.35
opacity = 1
#plt.grid()
plt.figure(figsize=plt.figaspect(0.618))
#plt.rcParams.update({'font.size': 12})
plt.rcParams['axes.axisbelow'] = True
rects1 = plt.bar(index+bar_width/2, tuple(hmm_mean), bar_width,
                 alpha=opacity,
                 color='orange',
                 label='Ottley et al.',
                 #yerr=tuple(np.array(hmm_sd[task])/len(ncp_results[task][1])),
                 capsize=3
                 )

rects2 = plt.bar(index + 3*bar_width/2, tuple(m), bar_width,
                 alpha=opacity,
                 #color='#7cc496',
                 label='Our Method',
                 #yerr=tuple(sd),
                 capsize=3
                 )

plt.xlabel('k')
plt.ylabel('accuracy')
plt.xticks(index + bar_width, ('1', '5', '10', '20', '50', '100'))
plt.yticks((0, 0.2, 0.4, 0.6, 0.8, 1.0))
plt.ylim([0, 1])


plt.title("Next Interaction Prediction for Open Ended Task w/ Crime Dataset")
plt.legend(loc='upper left')
plt.savefig('figures/ncp_%s_%d.png' % ('openended', int(random.random()*1000)))
plt.close()