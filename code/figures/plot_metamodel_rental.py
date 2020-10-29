import sys
sys.path.append('../implementation')
# uncommon way to import, but done for easy organization of files

import matplotlib.pyplot as plt
from itertools import compress
import json


data_path = './outputs/rental_metamodel_3/0_0.json'

with open(data_path, 'r') as fp:
    temp_res = json.load(fp)

res = {int(key): temp_res[key] for key in temp_res.keys()}
print(res)


dimensions = 0
k = 3
save_path = None
title = ''

#dimensions = ['location', 'numBedroom', 'numBathroom', 'numPeople', 'price']

#for full crime data:
#dimensions = ['location', 'category', 'Description', 'District', 'Neighborhood', 'DateOccur']



res_array = {m: [res[t][m] for t in res.keys()] for m in res[0].keys()}

top_n = set()
for t in range(3, len(res.keys())):
    for item in [k for k, v in sorted(res[t].items(), key=lambda item: item[1])][-k:]:
        top_n.add(item)
    print({k:v for k, v in sorted(res[t].items(), key=lambda item: item[1])})

for m in res_array.keys():
    a = res_array[m]
    if m in top_n:
        plt.plot(a, label=m, alpha=0.6, lw=2)
    else:
        continue
        plt.plot(a)



plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=2)
#plt.title('Palm Springs Vacation Rentals Dataset')
plt.title(title)
plt.xlabel('clicks')

if save_path is not None:
    plt.savefig(save_path, dpi=400)

plt.show()

