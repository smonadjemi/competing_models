'''
Import all the needed packages
'''
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import special as sp
from scipy import stats
import math
import errno
import time
import collections
from matplotlib.animation import FuncAnimation
import matplotlib.animation as anim_pack
import Models
import itertools
from colour import Color


'''
this block contains functions for visualizations
'''


def batch_animated_visualization(data, ui_data, interaction_models, data_models, model_belief_dicts, pmfs,
                                 x_dim, y_dim, color_dim='type', loc_dim='location', cont_dims=[], disc_dims=[],
                                 include_time=False, experiment_title="not specified", sorted_predictions=[], save=False, view=True):

    cont_domains, disc_domains = get_domains(data)

    # things to visualize
    vis_data_x = data[x_dim]
    vis_data_y = data[y_dim]

    vis_clicks_x = []
    vis_clicks_y = []

    all_x = np.linspace(cont_domains[x_dim][0], cont_domains[x_dim][1], 50)
    all_y = np.linspace(cont_domains[y_dim][0], cont_domains[y_dim][1], 50)

    table_color_gradient = list(Color('red').range_to(Color("green"), 1001))

    # 8colors
    eight_colors = [(117 / 255, 112 / 255, 179 / 255, 1), (231 / 255, 41 / 255, 138 / 255, 1),
                    (102 / 255, 166 / 255, 30 / 255, 1), (27 / 255, 158 / 255, 119 / 255, 1),
                    (217 / 255, 95 / 255, 2 / 255, 1), (230 / 255, 171 / 255, 2 / 255, 1),
                    (166 / 255, 118 / 255, 29 / 255, 1), (102 / 255, 102 / 255, 102 / 255, 1)]

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(9, 8)
    # gs = gridspec.GridSpec(7, 7)
    gs.update(hspace=0.7, wspace=0.7)

    gs_none = gridspec.GridSpecFromSubplotSpec(9, 9, subplot_spec=gs[5:, 0:2])
    gs_loc = gridspec.GridSpecFromSubplotSpec(9, 9, subplot_spec=gs[5:, 2:4])
    gs_type = gridspec.GridSpecFromSubplotSpec(9, 9, subplot_spec=gs[5:, 4:6])
    gs_mixed = gridspec.GridSpecFromSubplotSpec(9, 9, subplot_spec=gs[5:, 6:8])

    fig.suptitle("Experiment Details: " + experiment_title, fontweight="bold", fontsize=14)

    mc_to_ax = {}
    mc_to_name = {(False, False): 'None', (True, False): 'Geo-Based', (False, True): 'Type-Based', (True, True): 'Mixed'}

    outerax = fig.add_subplot(gs_none[:, :])
    outerax.tick_params(axis='both', which='both', bottom=0, left=0,
                        labelbottom=0, labelleft=0)
    plt.setp(outerax.spines.values(), color='green', linewidth=2)
    outerax.set_title("None\n%.2f"%model_belief_dicts[0][(False, False)])
    mc_to_ax[(False, False)] = outerax

    outerax = fig.add_subplot(gs_loc[:, :])
    outerax.tick_params(axis='both', which='both', bottom=0, left=0,
                        labelbottom=0, labelleft=0)
    outerax.set_title("Geo-Based\n%.2f" % model_belief_dicts[0][(True, False)])
    mc_to_ax[(True, False)] = outerax

    outerax = fig.add_subplot(gs_type[:, :])
    outerax.tick_params(axis='both', which='both', bottom=0, left=0,
                        labelbottom=0, labelleft=0)
    outerax.set_title("Type-Based\n%.2f" % model_belief_dicts[0][(False, True)])
    mc_to_ax[(False, True)] = outerax

    outerax = fig.add_subplot(gs_mixed[:, :])
    outerax.tick_params(axis='both', which='both', bottom=0, left=0,
                        labelbottom=0, labelleft=0)
    outerax.set_title("Mixed\n%.2f" % model_belief_dicts[0][(True, True)])
    mc_to_ax[(True, True)] = outerax

    mc_to_gs = {(False, False): gs_none, (True, False): gs_loc, (False, True): gs_type, (True, True): gs_mixed}

    # data/click plot
    ax1 = fig.add_subplot(gs[0:4, 0:4])
    colors = eight_colors
    colors_dict = {t: c for t, c in zip(disc_domains[color_dim], colors)}
    colors = [list(colors_dict[t]) for t in data[color_dim]]
    scatter_data = ax1.scatter(vis_data_x, vis_data_y, s=2, marker='o', c=colors)

    # plot the user interaction data
    scatter_clicks = ax1.scatter(vis_clicks_x, vis_clicks_y, marker='^', c='black')

    # title = "t=" + str(len(ui_data))
    # ax1.set_title(title, fontsize=10)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax1.axis('off')
    ax1.tick_params(axis='both', which='both', bottom=0, left=0,
                        labelbottom=0, labelleft=0)
    ax1.set_title("Data and Clicks")

    # data/click plot
    ax_pred = fig.add_subplot(gs[0:4, 4:])
    colors = eight_colors
    colors_dict = {t: c for t, c in zip(disc_domains[color_dim], colors)}
    colors = [list(colors_dict[t]) for t in data[color_dim]]
    scatter_data = ax_pred.scatter(vis_data_x, vis_data_y, s=3, marker='o', c=colors)
    ax_pred.set_xticks([])
    ax_pred.set_yticks([])

    ax_pred.axis('off')
    ax_pred.tick_params(axis='both', which='both', bottom=0, left=0,
                    labelbottom=0, labelleft=0)
    ax_pred.set_title("Top-100 Predictions")
    pred_x = [d[1] for d in sorted_predictions[0][-100:]]
    pred_y = [d[2] for d in sorted_predictions[0][-100:]]
    scatter_pred = ax_pred.scatter(pred_x, pred_y, marker='o', s=1,  c='black')


    mc_to_ax_loc = {}
    mc_to_ax_type = {}
    mc_to_type_bar = {}
    mc_to_loc_scatter = {}
    for mc in mc_to_gs.keys():
        ax2 = fig.add_subplot(mc_to_gs[mc][1:4, 1:8])
        ax2.axis('off')
        ax2.set_title('location', fontsize=8)
        mc_to_ax_loc[mc] = ax2

        ax3 = fig.add_subplot(mc_to_gs[mc][5:8, 1:8])
        ax3.set_ylim([0, 1])
        ax3.set_yticks([0, 0.5, 1])
        ax3.yaxis.set_tick_params(labelsize=3)
        ax3.xaxis.set_tick_params(rotation=45, labelsize=3)
        ax3.set_title("type", fontsize=8)
        # ax3.set_xlabel("categories")
        type_bar = ax3.bar([], [])
        mc_to_ax_type[mc] = ax3


        plt.setp(mc_to_ax[mc].spines.values(), color=table_color_gradient[int(model_belief_dicts[0][mc] * 1000)].get_hex_l(), linewidth=2)








    xs = []
    ys = []
    ps_null = []
    for point in data:
        xs.append(point[x_dim])
        ys.append(point[y_dim])

        # for uniform null
        #ps_null.append(1)

        # for gaussian null
        #ps_null.append(d_loc.get_probability([point[x_dim], point[y_dim]]))

    #ax2_d.scatter(xs, ys, c=ps_null, cmap="copper", marker='o', s=2)
    # type model bar chart


    #categories, proportions = data_models[color_dim]
    #ax3_d.bar(categories, proportions, color=eight_colors)

    """
    Add the table of models
    """
    """
    # for now, say n is the number of models
    n = len(interaction_models)
    columns = ['location', 'type', r'$p(\mathcal{M}_n)$']
    rows = [r'$\mathcal{M}_%d$' % (d + 1) for d in range(2 ** n)]


    model_dict = model_belief_dicts[0]
    table_data = [[r'$\checkmark$' if isinstance(j, bool) and j == True else '' if isinstance(j, bool) and j == False
                    else j for j in [*i, model_dict[i]]] for i in model_dict]
    table_colors = [['#D3D3D3' if isinstance(j, bool) and j == True else 'w' if isinstance(j, bool) and j == False
                    else table_color_gradient[int(j*1000)].get_hex_l() for j in [*i, model_dict[i]]] for i in model_dict]

    ax_table = fig.add_subplot(gs[0:5, 5:])
    ax_table.axis('off')
    table = ax_table.table(table_data, cellColours=table_colors, loc='upper center', cellLoc='center', colLabels=columns, rowLabels=rows)
    ax_table.set_title("Belief over Models")
    """
    # print(pmfs)
    # plot pmfs

    location_colors = list(Color('blue').range_to(Color("yellow"), 1001))

    for mc in mc_to_ax_loc.keys():
        ax_loc = mc_to_ax_loc[mc]
        ax_type = mc_to_ax_type[mc]
        loc_pmf = pmfs[0][mc]['location']
        type_pmf = pmfs[0][mc]['type']

        xs = []
        ys = []
        ps = []
        for dot in loc_pmf.keys():
            xs.append(dot[0])
            ys.append(dot[1])
            ps.append(loc_pmf[dot])

        scatter = ax_loc.scatter(xs, ys, c=ps, cmap="copper", marker='o', s=2)
        mc_to_loc_scatter[mc] = scatter

        if mc in mc_to_type_bar.keys():
            mc_to_type_bar[mc].remove()

        categories = [c for c in type_pmf.keys()]
        proportions = [type_pmf[c] for c in categories]
        type_bar = ax_type.bar(categories, proportions, color=eight_colors)
        ax3.set_xticklabels(list(categories))
        mc_to_type_bar[mc] = type_bar

        for rect, p in zip(type_bar, proportions):
            rect.set_height(p)


    def init():
        nonlocal mc_to_ax_loc, mc_to_ax_type, mc_to_loc_scatter, mc_to_type_bar

        print("Frame %d/%d  " % (0, len(ui_data)), end="\r", flush=True)

        for mc in mc_to_gs.keys():
            ax_loc = mc_to_ax_loc[mc]
            ax_type = mc_to_ax_type[mc]
            loc_pmf = pmfs[0][mc]['location']
            type_pmf = pmfs[0][mc]['type']



            xs = []
            ys = []
            ps = []
            for dot in loc_pmf.keys():
                xs.append(dot[0])
                ys.append(dot[1])
                ps.append(loc_pmf[dot])

            scatter = ax_loc.scatter(xs, ys, c=ps, cmap="copper", marker='o', s=2)
            mc_to_loc_scatter[mc] = scatter


            type_bar = mc_to_type_bar[mc]
            categories = [c for c in type_pmf.keys()]
            proportions = [type_pmf[c] for c in categories]

            for rect, p in zip(type_bar, proportions):
                rect.set_height(p)
        """
        table_data = [
            [r'$\checkmark$' if isinstance(j, bool) and j == True else '' if isinstance(j, bool) and j == False
            else j for j in [*i, model_dict[i]]] for i in model_dict]
        table_colors = [['#D3D3D3' if isinstance(j, bool) and j == True else 'w' if isinstance(j, bool) and j == False
                        else table_color_gradient[int(j * 1000)].get_hex_l() for j in [*i, model_dict[i]]] for i in model_dict]
        table.remove()
        table = ax_table.table(table_data, cellColours=table_colors, loc='upper center', cellLoc='center', colLabels=columns,
                          rowLabels=rows)
        """
        plt.draw()

    def update(frame):
        nonlocal mc_to_ax_loc, mc_to_ax_type, mc_to_loc_scatter, mc_to_type_bar

        print("Frame %d/%d" % (frame+1, len(ui_data)), end="\r", flush=True)

        # update clicks
        vis_clicks_x = ui_data[x_dim][:frame + 1]
        vis_clicks_y = ui_data[y_dim][:frame + 1]

        scatter_clicks.set_offsets(list(zip(vis_clicks_x[:frame + 1], vis_clicks_y[:frame + 1])))

        pred_x = [d[1] for d in sorted_predictions[frame][-100:]]
        pred_y = [d[2] for d in sorted_predictions[frame][-100:]]
        scatter_pred.set_offsets(list(zip(pred_x, pred_y)))

        for mc in mc_to_gs.keys():
            ax_loc = mc_to_ax_loc[mc]
            ax_type = mc_to_ax_type[mc]
            loc_pmf = pmfs[frame][mc]['location']
            type_pmf = pmfs[frame][mc]['type']

            plt.setp(mc_to_ax[mc].spines.values(),
                     color=table_color_gradient[int(model_belief_dicts[frame][mc] * 1000)].get_hex_l(), linewidth=2)

            mc_to_ax[mc].set_title("%s\n%.2f" % (mc_to_name[mc], model_belief_dicts[frame][mc]))


            xs = []
            ys = []
            ps = []
            for dot in loc_pmf.keys():
                xs.append(dot[0])
                ys.append(dot[1])
                ps.append(loc_pmf[dot])

            mc_to_loc_scatter[mc].remove()
            scatter = ax_loc.scatter(xs, ys, c=ps, cmap="copper", marker='o', s=2)
            mc_to_loc_scatter[mc] = scatter

            type_bar = mc_to_type_bar[mc]
            categories = [c for c in type_pmf.keys()]
            proportions = [type_pmf[c] for c in categories]

            for rect, p in zip(type_bar, proportions):
                rect.set_height(p)

        """
        # update table
        table.remove()
        model_dict = model_belief_dicts[frame+1]
        table_data = [
            [r'$\checkmark$' if isinstance(j, bool) and j == True else '' if isinstance(j, bool) and j == False
            else j for j in [*i, model_dict[i]]] for i in model_dict]
        table_colors = [['#D3D3D3' if isinstance(j, bool) and j == True else 'w' if isinstance(j, bool) and j == False
                        else table_color_gradient[int(j * 1000)].get_hex_l() for j in [*i, model_dict[i]]] for i in model_dict]

        table = ax_table.table(table_data, cellColours=table_colors, loc='upper center', cellLoc='center', colLabels=columns, rowLabels=rows)
        """

    animation = FuncAnimation(fig, update, frames=len(ui_data), init_func=init, interval=1000, repeat_delay=1000)

    # Set up formatting for the movie files
    Writer = anim_pack.writers['ffmpeg']
    writer = Writer(fps=0.5, metadata=dict(artist='Me'), bitrate=1800)

    if save:
        print("saving animation...")
        animation.save('figures/example_animation'+ str(int(random.random()*10000)) +'.mp4', writer=writer, dpi=200)

    # ax3 = fig.add_subplot(133)
    # ax3.set_title('final predictive distribution')
    # ax3.set_xticks([])
    # ax3.set_yticks([])

    # fig.autofmt_xdate()

    if view:
        plt.show()

    print()



def incremental_animated_visualization(data, ui_data=[], cmodel=None, dmodel=None, x_dim='x', y_dim='y', color_dim='type', include_time=False,
              savefilepath=None):
    cont_domains, disc_domains = get_domains(data)

    # things to visualize
    vis_data_x = data[x_dim]
    vis_data_y = data[y_dim]

    vis_clicks_x = []
    vis_clicks_y = []

    all_x = np.linspace(cont_domains[x_dim][0], cont_domains[x_dim][1], 10)
    all_y = np.linspace(cont_domains[y_dim][0], cont_domains[y_dim][1], 10)


    # models
    location_model = None
    type_model = None

    # 8colors
    eight_colors = [(117 / 255, 112 / 255, 179 / 255, 1), (231 / 255, 41 / 255, 138 / 255, 1),
                    (102 / 255, 166 / 255, 30 / 255, 1), (27 / 255, 158 / 255, 119 / 255, 1),
                    (217 / 255, 95 / 255, 2 / 255, 1), (230 / 255, 171 / 255, 2 / 255, 1),
                    (166 / 255, 118 / 255, 29 / 255, 1), (102 / 255, 102 / 255, 102 / 255, 1)]

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(10, 10)
    #gs = gridspec.GridSpec(7, 7)
    gs.update(hspace=0.7, wspace=0.7)

    gs_clicks = gridspec.GridSpecFromSubplotSpec(9, 9, subplot_spec=gs[5:, 0:5])
    gs_data = gridspec.GridSpecFromSubplotSpec(9, 9, subplot_spec=gs[5:, 5:])

    info_ax = plt.subplot(gs[9, :])
    info_ax.axis("off")
    info_ax.text(0, 0, "Experiment Details: type-based")

    """
    outergs = fig.add_gridspec(10, 10)
    #outergs.update(bottom=.50, left=0.07, right=0.50, top=0.93)
    outerax = fig.add_subplot(outergs[5:9, 0:4])
    outerax.tick_params(axis='both', which='both', bottom=0, left=0,
                        labelbottom=0, labelleft=0)


    
    """

    outerax = fig.add_subplot(gs_clicks[:, :])
    outerax.tick_params(axis='both', which='both', bottom=0, left=0,
                        labelbottom=0, labelleft=0)
    outerax = fig.add_subplot(gs_data[:, :])
    outerax.tick_params(axis='both', which='both', bottom=0, left=0,
                        labelbottom=0, labelleft=0)

    # data/click plot
    ax1 = fig.add_subplot(gs[0:4, 0:4])
    colors = eight_colors
    colors_dict = {t: c for t, c in zip(disc_domains[color_dim], colors)}
    colors = [list(colors_dict[t]) for t in data[color_dim]]
    scatter_data = ax1.scatter(vis_data_x, vis_data_y, s=2, marker='o', c=colors)


    # plot the user interaction data
    scatter_clicks = ax1.scatter(vis_clicks_x, vis_clicks_y, marker='^', c='black')

    # title = "t=" + str(len(ui_data))
    # ax1.set_title(title, fontsize=10)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax1.axis('off')
    ax1.set_title("Data and Clicks")

    # location model plot
    ax2 = fig.add_subplot(gs_clicks[1:4, 1:4])
    ax2.axis('off')
    ax2.set_title("Location Model - C")

    ax2_d = fig.add_subplot(gs_data[1:4, 1:4])
    ax2_d.set_title("Location Model - D")
    ax2_d.axis("off")
    d_reshaped = np.array([list(item) for item in data[[x_dim, y_dim]]])
    d_loc_cov = np.cov(np.transpose(d_reshaped))
    d_loc_mu = np.mean(d_reshaped, axis=0)
    p_d = np.array(
        [[stats.multivariate_normal.pdf([x, y], d_loc_mu, d_loc_cov) for x in all_x] for y in all_y])
    ax2_d.pcolormesh(all_x, all_y, p_d, shading='gouraud')






    # type model bar chart
    ax3 = fig.add_subplot(gs_clicks[1:4, 5:8])

    ax3.set_ylim([0, 1])
    ax3.set_yticks([0, 0.5, 1])
    ax3.xaxis.set_tick_params(rotation=45)
    ax3.set_title("Type Model - C")
    ax3.set_xlabel("categories")
    type_bar = ax3.bar([], [])


    ax3_d = fig.add_subplot(gs_data[1:4, 5:8])
    ax3_d.set_title("Type Model - D")
    ax3_d.set_ylim([0, 1])
    ax3_d.set_yticks([0, 0.5, 1])
    ax3_d.xaxis.set_tick_params(rotation=45)
    ax3_d.set_xlabel("categories")

    categories = np.unique(data[color_dim])
    l = len(data[color_dim])
    c = collections.Counter(data[color_dim])
    proportions = np.array([c[t] / l for t in categories])
    ax3_d.bar(categories, proportions, color=eight_colors)




    """
    Add the table of models
    """

    # for now, say n is the number of models
    n = 2
    columns = ['location', 'type', r'$p(\mathcal{M}_n)$']
    rows = [r'$\mathcal{M}_%d$' % (d+1) for d in range(2**n)]

    table_data = np.zeros((2**n, n+1))
    ax_table = fig.add_subplot(gs[0:5, 5:])
    ax_table.axis('off')
    table = plt.table(table_data, loc='upper center', cellLoc='center', colLabels=columns, rowLabels=rows)
    ax_table.set_title("Belief over Models")






    def init():
        nonlocal location_model, type_model, type_bar

        print("init")

        """
        create all models and set parameters
        """
        location_x_var_name = x_dim
        location_y_var_name = y_dim
        mu_0 = np.array([(min(data[location_x_var_name] + max(data[location_x_var_name]))) / 2,
                         (min(data[location_y_var_name] + max(data[location_y_var_name]))) / 2,
                         0])
        T_0 = np.array([[(max(data[location_x_var_name]) - min(data[location_x_var_name])) / 10, 0, 0],
                        [0, (max(data[location_y_var_name]) - min(data[location_y_var_name])) / 10, 0],
                        [0, 0, 10]])
        k_0 = 1
        v_0 = 3
        location_model = Models.MultivariateGaussianModel("location", mu_0, T_0, k_0, v_0)

        type_var_name = 'type'
        type_categories = np.unique(data[type_var_name])
        alpha = np.ones(type_categories.shape)
        type_model = Models.DirichletModel(type_var_name, type_categories, alpha)

        scatter_clicks.set_offsets([[None, None]])

        p = np.array(
            [[location_model.get_probability(np.array([(x, y, 0)], dtype=[(x_dim, 'float'), (y_dim, "float"), ('timstamp', "float")]), include_time=include_time) for x in all_x] for y in
             all_y])
        # print(min(map(min, p)), max(map(max, p)))
        ax2.pcolormesh(all_x, all_y, p, shading='gouraud')

        type_bar.remove()
        type_bar = ax3.bar(type_model.categories, type_model.mu, color=eight_colors)
        ax3.set_xticklabels(list(type_model.categories))
        ax2.set_xlabel("categories")

        for rect, p in zip(type_bar, type_model.mu):
            rect.set_height(p)

        plt.draw()



    def update(frame):
        nonlocal location_model, type_model

        print("updating", frame)

        # update clicks
        vis_clicks_x = ui_data[x_dim][:frame+1]
        vis_clicks_y = ui_data[y_dim][:frame+1]

        # update location model
        loc_time_click = np.array([(ui_data[x_dim][frame], ui_data[y_dim][frame], frame)], dtype=[(x_dim, 'float'), (y_dim, "float"), ('timstamp', "float")])


        type_click = ui_data[color_dim][frame]
        type_model.update_model(type_click)

        p = np.array(
            [[location_model.get_probability(
                np.array([(x, y, 0)], dtype=[(x_dim, 'float'), (y_dim, "float"), ('timstamp', "float")]),
                include_time=include_time) for x in all_x] for y in
             all_y])
        # print(min(map(min, p)), max(map(max, p)))

        ax2.pcolormesh(all_x, all_y, p, shading='gouraud')

        scatter_clicks.set_offsets(list(zip(vis_clicks_x[:frame+1], vis_clicks_y[:frame+1])))


        # update type model
        location_model.update_model(loc_time_click)

        for rect, p in zip(type_bar, type_model.mu):
            rect.set_height(p)








    animation = FuncAnimation(fig, update, frames=len(ui_data), init_func=init, interval=1000, repeat_delay=1000)

    # Set up formatting for the movie files
    Writer = anim_pack.writers['ffmpeg']
    writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)

    #animation.save('figures/example_animation'+ str(time.time_ns()) +'.mp4', writer=writer)

    # ax3 = fig.add_subplot(133)
    # ax3.set_title('final predictive distribution')
    # ax3.set_xticks([])
    # ax3.set_yticks([])

    # fig.autofmt_xdate()

    plt.show()


def generate_colors(n, a=1):
    '''
    generates n random colors in RGBA format
    '''
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 255
        ret.append((r / 255, g / 255, b / 255, a))
    return ret




def visualize(data, ui_data = [], cmodel=None, dmodel=None, x_dim='x', y_dim='y', color_dim='type', include_time=False,
              savefilepath=None):
    cont_domains, disc_domains = get_domains(data)

    # 8colors
    eight_colors = [(117 / 255, 112 / 255, 179 / 255, 1), (231 / 255, 41 / 255, 138 / 255, 1),
                    (102 / 255, 166 / 255, 30 / 255, 1), (27 / 255, 158 / 255, 119 / 255, 1),
                    (217 / 255, 95 / 255, 2 / 255, 1), (230 / 255, 171 / 255, 2 / 255, 1),
                    (166 / 255, 118 / 255, 29 / 255, 1), (102 / 255, 102 / 255, 102 / 255, 1)]

    if cmodel is None:

        fig = plt.figure(figsize=(6, 7))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 2])

        gs.update(hspace=0.05)

        ax1 = plt.subplot(gs[0])
        ax3 = plt.subplot(gs[1])

        # ax2 = fig.add_subplot(122)

        # plot the data on x,y with colors as type
        colors = generate_colors(len(disc_domains[color_dim]))
        colors = eight_colors
        colors_dict = {t: c for t, c in zip(disc_domains[color_dim], colors)}
        colors = [list(colors_dict[t]) for t in data[color_dim]]
        ax1.scatter(data[x_dim], data[y_dim], s=2, marker='o', c=colors)

        # plot the user interaction data
        ax1.scatter(ui_data[x_dim], ui_data[y_dim], marker='^', c='black')

        # title = "t=" + str(len(ui_data))
        # ax1.set_title(title, fontsize=10)
        ax1.set_xticks([])
        ax1.set_yticks([])


        # ax3.bar(dummy_dm.domains, dummy_dm.mu_1, color=eight_colors)
        # ax3.bar(types, dummy_dm.mu_1, color=eight_colors)

        # ax3.set_xlabel(dm.names)
        ax3.set_ylim([0, 1])
        ax3.set_yticks([0, 0.5, 1])
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax1.axis('off')


    else:

        fig = plt.figure(figsize=(6, 10))
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 3, 2])

        gs.update(hspace=0.05)

        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax3 = plt.subplot(gs[2])

        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        # ax2 = fig.add_subplot(122)


        # plot the color map
        all_x = np.linspace(cont_domains[x_dim][0], cont_domains[x_dim][1], 10)
        all_y = np.linspace(cont_domains[y_dim][0], cont_domains[y_dim][1], 10)

        p = np.array(
            [[cmodel.get_probability({x_dim: x, y_dim: y}, include_time=include_time) for x in all_x] for y in all_y])
        print(min(map(min, p)), max(map(max, p)))
        ax2.pcolormesh(all_x, all_y, p, shading='gouraud')

        # plot the data on x,y with colors as type

        colors = eight_colors
        colors_dict = {t: c for t, c in zip(disc_domains[color_dim], colors)}
        colors = [list(colors_dict[t]) for t in data[color_dim]]
        ax1.scatter(data[x_dim], data[y_dim], s=6, marker='.', c=colors)

        ax1.axis('off')
        ax2.axis('off')

        # plot the user interaction data
        if len(cmodel.ui_data) > 0:
            # print(cm.ui_data)
            ax1.scatter(cmodel.ui_data[:, 0], cmodel.ui_data[:, 1], marker='^', c='black')
            ax2.scatter(cmodel.ui_data[:, 0], cmodel.ui_data[:, 1], marker='^', c='black')

        title = "t=" + str(len(cmodel.ui_data))
        # ax1.set_title(title, fontsize=10)
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2.set_xticks([])
        ax2.set_yticks([])

    if dmodel is not None:
        types = ['Homicide', 'Theft-Related', 'Assault', 'Arson', 'Fraud', 'Vandalism', 'Weapons', 'Vagrancy']
        ax3 = plt.subplot(gs[2])

        ax3.bar(list(types), dmodel.mu, color=eight_colors)

        # ax3.set_xlabel(dm.names)
        ax3.set_ylim([0, 1])
        ax3.set_yticks([0, 0.5, 1])
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)

    # ax3 = fig.add_subplot(133)
    # ax3.set_title('final predictive distribution')
    # ax3.set_xticks([])
    # ax3.set_yticks([])

    fig.autofmt_xdate()

    # ax1.set_title(title, fontsize=10)

    # categorical barchart

    if savefilepath is not None:
        if not os.path.exists(os.path.dirname(savefilepath)):
            try:
                os.makedirs(os.path.dirname(savefilepath))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        #plt.savefig(savefilepath, dpi=100, transparent=True, bbox_inches=0)

        # fig.autofmt_xdate()
    plt.show()




def get_domains(data):
    '''
    given the structured numpy array (data), this function find the domain
    of each dimention

    @return two dictionaries with {'continuous_dim_name': [min, max] }, {'dicrete_dim_name': list_of_values}

    '''
    continuous_domains = {}
    discrete_domains = {}
    for dim_name in data.dtype.names:
        if data.dtype[dim_name] is np.dtype('float'):
            # print(dim_name, "is continuous")
            continuous_domains[dim_name] = [np.min(data[dim_name]), np.max(data[dim_name])]
        else:
            # print(dim_name, "is discrete")
            discrete_domains[dim_name] = list(np.unique(data[dim_name]))

    return continuous_domains, discrete_domains


'''
this block contains all the functions needed for building and maintaining models
'''


def t_pdf(x, df, mu, sigma):
    d = len(x)
    '''
    print('x: ', x)
    print('df: ', df)
    print('mu: ', mu)
    print('sigma: ', sigma)
    '''

    # final formula is (a/b)*c
    a = sp.gamma((df + d) / 2.0)
    b = sp.gamma(df / 2.0) * df ** (d / 2.0) * math.pi ** (d / 2.0) * np.linalg.det(sigma) ** (1 / 2.0)
    c = (1 + (1.0 / df) * np.dot(np.transpose(x - mu), np.linalg.solve(sigma, (x - mu)))) ** (-(df + d) / 2.0)

    ans = (a / b) * c

    return ans


def mahalanobis_distance(x, mu, S):
    x = np.array([float(xx) for xx in x])
    ans = np.dot(np.transpose(x - mu), np.linalg.solve(S, (x - mu)))
    return ans


def get_recall_precision(inferred_set, ground_truth):
    # returns a tuple (recall, precision)
    recall = -1
    precision = -1

    # recall: wanna cover as much as possible of the ground truth
    c = 0
    for x in ground_truth:
        if x in inferred_set:
            c = c + 1

    recall = c / len(ground_truth)

    # recall: wanna maximize the true positives
    c = 0
    for x in inferred_set:
        if x in ground_truth:
            c = c + 1

    precision = c / len(inferred_set)

    return recall, precision


def get_intersection_of_objectives(full_data, all_objectives):
    intersection_set = np.array([], dtype=full_data.dtype)
    for data_point in full_data:
        if all([data_point in s for s in all_objectives]):
            intersection_set = np.append(intersection_set, data_point)

    return intersection_set


def compute_bic(num_observations, k_num_parameters, log_likelihood):
    return k_num_parameters * np.log(num_observations) - 2 * log_likelihood


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
