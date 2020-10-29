import sys
sys.path.append('../implementation')
# uncommon way to import, but done for easy organization of files

import Data
from simulation_rental import batch_animated_visualization
from simulation_rental import simulate
from simulation_with_libs import simulate_with_libs
import res
import json
import matplotlib.pyplot as plt
import numpy as np

"""
parameters
"""
DIRECTORY = './outputs/boardrooms_metamodel_1'
DIRECTORY_BASELINE = './outputs/boardrooms_hmm_1'
#synthetic or crime
DATASET = 'boardrooms'
UI_DATASET = 'boardrooms'
task_type = 1
# participant = 150

overall_accuracy = {k: [] for k in [1, 5, 10, 20, 50, 100]}
overall_accuracy_bl = {k: [] for k in [1, 5, 10, 20, 50, 100]}
full_ui_data = ui_data = Data.load_ui_data(UI_DATASET, group=1)

LOCATION_VAR_NAMES_FOR_DATASET = {"synthetic": ('x', 'y'),
                                  "crime": ('lat', 'lng'),
                                  'rental': ('lat', 'lng'),
                                  'full-crime': ('latitude', 'longitude'),
                                  'boardrooms': None}

location_variable_names = LOCATION_VAR_NAMES_FOR_DATASET[DATASET]
dimensions = ['mktcap', 'unrelated', 'female', 'age', 'tenure', 'medianpay', 'industry']
d_dimensions = ['industry']
c_dimensions = ['mktcap', 'unrelated', 'female', 'age', 'tenure', 'medianpay']

print('processing ', len(full_ui_data.keys()), ' participants')

xs = []
ys = []

xs_bl = []
ys_bl = []
for participant in full_ui_data.keys():
    """
    Load the data
    """
    #data = Data.load_data(DATASET)
    #ui_data = Data.load_ui_data(UI_DATASET)[participant]

    if len(full_ui_data[participant]) < 3:
        continue

    """
    Build interaction/data models for all time steps
    """
    # data_models, interaction_models, model_belief_dicts, pmf_models, target_analysis_dict, ncp_results_dict, sorted_points = simulate(data, ui_data, ui_data, location_variable_names, dimensions, d_dimensions, c_dimensions)
    title = ' '.join([DATASET, str(task_type), str(participant)])

    #plot_path = DIRECTORY + "/fig_boardrooms_"+str(participant)+".png"
    #res.plot_results(model_belief_dicts, title, dimensions, k=2, save_path=plot_path)

    ncp_path = DIRECTORY + "/ncp_1_" + str(participant) + ".json"
    with open(ncp_path, 'r') as fp:
        ncp_results_dict = json.load(fp)


    for tk in overall_accuracy.keys():
        overall_accuracy[tk].append(float(ncp_results_dict[str(tk)]))
        xs.append(tk)
        ys.append(ncp_results_dict[str(tk)])

    # print('participant:', str(participant), '-- ', ncp_results_dict)

    ncp_path = DIRECTORY_BASELINE + "/ncp_1_" + str(participant) + ".json"
    with open(ncp_path, 'r') as fp:
        ncp_results_dict = json.load(fp)

    for tk in overall_accuracy.keys():
        overall_accuracy_bl[tk].append(float(ncp_results_dict[str(tk)]))
        xs_bl.append(tk)
        ys_bl.append(ncp_results_dict[str(tk)])

m = [np.mean(overall_accuracy[tk]) for tk in overall_accuracy.keys()]
sd = [np.std(overall_accuracy[tk]) for tk in overall_accuracy.keys()]

m_bl = [np.mean(overall_accuracy_bl[tk]) for tk in overall_accuracy_bl.keys()]
sd_bl = [np.std(overall_accuracy_bl[tk]) for tk in overall_accuracy_bl.keys()]



title = 'Next Interaction Prediction for Boardrooms (' + str(len(ui_data)) + ' Participants - Group ' + str(task_type) + ')'

index = np.arange(6)
bar_width = 0.35
opacity = 1
plt.rcParams['axes.axisbelow'] = True


rects1 = plt.bar(index+bar_width/1.9, tuple(m_bl), bar_width,
                 alpha=opacity,
                 label='Logistic Regression',
                 yerr=tuple(sd_bl),
                 capsize=3
                 )

rects2 = plt.bar(index-bar_width/1.9, tuple(m), bar_width,
                 alpha=opacity,
                 label='Our Method',
                 yerr=tuple(sd),
                 capsize=3
                 )

plt.legend()
plt.xlabel('k')
plt.ylabel('avg accuracy')
plt.title(title)
plt.xticks(index, ('1', '5', '10', '20', '50', '100'))
plt.yticks((0, 0.2, 0.4, 0.6, 0.8, 1.0))
plt.ylim([-0.2, 1.2])
plt.grid()
ncp_path = DIRECTORY + "/ncp_overall_boardrooms.png"
plt.savefig(ncp_path, dpi=400)
plt.show()


plt.scatter(xs, ys)
plt.title(title)
plt.xlabel('k')
plt.ylabel('accuracy for each participant')
plt.xticks([1, 5, 10, 20, 50, 100])
plt.yticks((0, 0.2, 0.4, 0.6, 0.8, 1.0))
plt.ylim([-0.2, 1.2])

plt.show()


index = np.arange(7)
bar_width = 0.35
opacity = 1
#plt.grid()
plt.rcParams['axes.axisbelow'] = True


rects1 = plt.boxplot([overall_accuracy[k] for k in overall_accuracy.keys()])
rects2 = plt.boxplot([overall_accuracy_bl[k] for k in overall_accuracy.keys()])

plt.xlabel('k')
plt.ylabel('accuracy')
plt.legend()
plt.title(title)
plt.xticks(index, ('', '1', '5', '10', '20', '50', '100'))
plt.yticks((0, 0.2, 0.4, 0.6, 0.8, 1.0))
plt.ylim([-0.2, 1.2])
ncp_path = DIRECTORY + "/ncp_overall_boardrooms_box.png"
plt.savefig(ncp_path, dpi=400)
plt.show()




"""
Send for animation visualization
"""
"""
experiment_title = "Data: " + DATASET
batch_animated_visualization(data, ui_data, data_models=data_models, interaction_models=interaction_models, pmfs=pmf_models,
                             x_dim=location_x_var_name, y_dim=location_y_var_name,
                             model_belief_dicts=model_belief_dicts, experiment_title=experiment_title, sorted_predictions=sorted_points, save=SAVE_ANIMATION,
                             include_time=False)
"""