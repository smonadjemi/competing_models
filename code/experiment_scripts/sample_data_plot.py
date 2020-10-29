import sys
sys.path.append('../implementation')
# uncommon way to import, but done for easy organization of files

import matplotlib.pyplot as plt

data = [('x_1', 0.35, 0.85, 'Italian'),
        ('x_2', 0.95, 0.45, 'Mexican'),
        ('x_3', 0.85, 0.1, 'Persian'),
        ('x_4', 0.75, 0.25, 'Italian'),
        ('x_5', 0.15, 0.75, 'Mexican')]

colors = {'Italian': '#1b9e77', 'Persian': '#d95f02', 'Mexican': '#7570b3'}
plt.rcParams['mathtext.fontset'] = 'stix'
fig, ax = plt.subplots()
fig.set_size_inches(4.3, 2.9)
for t in colors.keys():
        x = [d[1] for d in data if d[3] == t]
        y = [d[2] for d in data if d[3] == t]
        c = [colors[d[3]] for d in data if d[3] == t]
        l = [d[3] for d in data if d[3] == t]
        txts = [d[0] for d in data if d[3] == t]




        ax.scatter(x, y, edgecolors='none', c=c, s=1000, label=t)


        for i, txt in enumerate(txts):
                tt = r'$%s$'%txt
                ax.annotate(tt, (x[i]-0.02, y[i]-0.02), color='white', size=20)

plt.ylim([0,1])
#plt.axis('off')
plt.yticks([])
plt.xticks([])
lgnd = plt.legend(fontsize=8)
for handle in lgnd.legendHandles:
    handle.set_sizes([50])
plt.tight_layout()
plt.savefig('./outputs/example_d.png', dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.show()
