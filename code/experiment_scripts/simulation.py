
"""
Contains simulation function
"""

import sys
sys.path.append('../implementation')
# uncommon way to import, but done for easy organization of files

import numpy as np
import Models
import copy
import itertools
from scipy import stats
import collections
from utils import printProgressBar


def simulate(data, ui_data, gt_data, location_variable_names, dimensions):

    location_x_var_name, location_y_var_name = location_variable_names

    """
    Build interaction/data models for all time steps
    """
    interaction_models = {dim: {t: None for t in range(len(ui_data) + 1)} for dim in dimensions}
    data_models = {dim: None for dim in dimensions}

    # the interaction model at t=0
    location_var_name = 'location'
    mu_0 = np.array([(min(data[location_x_var_name] + max(data[location_x_var_name]))) / 2,
                     (min(data[location_y_var_name] + max(data[location_y_var_name]))) / 2,
                     0])
    T_0 = np.array([[(max(data[location_x_var_name]) - min(data[location_x_var_name])) / 100, 0, 0],
                    [0, (max(data[location_y_var_name]) - min(data[location_y_var_name])) / 100, 0],
                    [0, 0, 1]])
    k_0 = 1
    v_0 = 3
    #location_model = Models.MultivariateGaussianModel(location_var_name, mu_0, T_0, k_0, v_0)

    location_model = Models.NWModel(data, k_0, v_0)

    type_var_name = 'numBedroom'
    type_categories = np.unique(data[type_var_name])
    l = len(data[type_var_name])
    c = collections.Counter(data[type_var_name])
    proportions = np.array([c[t] / l for t in type_categories])
    alpha = np.ones(type_categories.shape)
    #alpha =  8 * proportions
    type_model = Models.DirichletModel(type_var_name, type_categories, alpha)

    interaction_models[location_var_name][0] = copy.deepcopy(location_model)
    interaction_models[type_var_name][0] = copy.deepcopy(type_model)

    # the static data model (does not change over time)
    d_reshaped = np.array([list(item) for item in data[[location_x_var_name, location_y_var_name]]])
    d_loc_cov = np.cov(np.transpose(d_reshaped))
    d_loc_mu = np.mean(d_reshaped, axis=0)
    data_models[location_var_name] = (d_loc_mu, d_loc_cov)


    """
    building the kde here
    """

    m1 = data[location_x_var_name]
    m2 = data[location_y_var_name]

    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    kde_loc_pdf = {(point[location_x_var_name], point[location_y_var_name]):
                                     kernel.pdf([point[location_x_var_name], point[location_y_var_name]])[0]
                                 for point in data}

    total_locs = sum([kde_loc_pdf[t] for t in kde_loc_pdf.keys()])

    for k in kde_loc_pdf.keys():
        kde_loc_pdf[k] /= total_locs

    categories = np.unique(data[type_var_name])
    l = len(data[type_var_name])
    c = collections.Counter(data[type_var_name])
    proportions = np.array([c[t] / l for t in categories])
    data_models[type_var_name] = (categories, proportions)

    printProgressBar(0, len(ui_data), prefix="Updating Models: ")
    for click_index in range(len(ui_data)):
        click = ui_data[click_index]

        loc_time_click = np.array([(click[location_x_var_name], click[location_y_var_name], click_index)],
                                  dtype=[(location_x_var_name, 'float'), (location_y_var_name, "float"),
                                         ('timstamp', "float")])
        type_click = click[type_var_name]

        type_model.update_model(type_click)
        location_model.update_model(loc_time_click)

        interaction_models[location_var_name][click_index + 1] = copy.deepcopy(location_model)
        interaction_models[type_var_name][click_index + 1] = copy.deepcopy(type_model)

        printProgressBar(click_index+1, len(ui_data), prefix="Updating Models: ")

    """
    Find model evidence for all time steps
    """



    # the number of dimensions
    N = len(dimensions)

    # True means click models, False means data models. Embedding this into a dictionary for easy access.
    MODELS_DICTIONARY = {True: interaction_models,
                         False: data_models}

    INTERACTION_MATTERS = True
    INTERACTION_DOES_NOT_MATTER = False

    # model configurations, we have 2**n of them.
    model_configs = list(itertools.product([False, True], repeat=N))

    # maps time steps to a dictionary of model evidences. this will be passed to the visualization function
    model_belief_dicts = {t: {} for t in range(len(ui_data) + 1)}
    pmf_models = {t: {c: {} for c in model_configs} for t in range(len(ui_data) + 1)}

    # in the beginning, we have uniform belief over all models.
    model_evidence_dict = {m: 1 / 2 ** N for m in model_configs}
    model_belief_dicts[0] = model_evidence_dict

    # loop over each time step to calculate model
    printProgressBar(0, len(ui_data), prefix="Computing Model Evidences:")
    for time_step in range(0, len(ui_data)):
        # loop over each of the 2^n models for this time step

        p_type_given_location_important = {t: sum(MODELS_DICTIONARY[True][location_var_name][time_step].get_probability(np.array([(d[location_x_var_name], d[location_y_var_name], time_step+1)],
                                         dtype=[(location_x_var_name, 'float'), (location_y_var_name, "float"),
                                                ('timstamp', "float")]), include_time=False) for d in data[tuple([data[type_var_name] == t])]) for t in type_categories}

        observed_clicks = ui_data[:time_step+1]

        total_evidence = 0
        for model_config in model_configs:
            # find pmf
            location_config = model_config[0]
            type_config = model_config[1]

            # location model
            m_location = MODELS_DICTIONARY[location_config][location_var_name]
            pmf_location = {}

            if location_config == INTERACTION_MATTERS:
                m_location = m_location[time_step]
                total_prob = 0
                """
                total_p_loc_given_type = {point_type: sum([m_location.get_probability(np.array([(p[location_x_var_name], p[location_y_var_name], time_step)],
                                         dtype=[(location_x_var_name, 'float'), (location_y_var_name, "float"),
                                          ('timstamp', "float")])) for p in data[tuple([data[type_var_name]==point_type])]]) for point_type in categories}
                """

                for point in data:
                    loc_point = np.array([(point[location_x_var_name], point[location_y_var_name], time_step+1)],
                                         dtype=[(location_x_var_name, 'float'), (location_y_var_name, "float"),
                                                ('timstamp', "float")])

                    if type_config == INTERACTION_MATTERS:
                        # loc model - mixed
                        #m_type = MODELS_DICTIONARY[type_config][type_var_name][time_step]
                        #prob = ((m_location.get_probability(loc_point))/(total_p_loc_given_type[point[type_var_name]])) * m_type.mu[list(m_type.categories).index(point[type_var_name])]
                        prob = m_location.get_probability(loc_point, include_time=True)
                        #m_type = MODELS_DICTIONARY[type_config][type_var_name][time_step]
                        #prob = (m_location.get_probability(loc_point, include_time=False) / p_type_given_location_important[point[type_var_name]]) * m_type.mu[
                        #    list(m_type.categories).index(point[type_var_name])]
                    else:
                        # loc model - geo-based
                        prob = m_location.get_probability(loc_point, include_time=True)


                    total_prob += prob
                    pmf_location[(point[location_x_var_name], point[location_y_var_name])] = prob
                # normalize pmf to get probabilities
                for key in pmf_location:
                    pmf_location[key] = np.exp(np.log(pmf_location[key]) - np.log(total_prob))
            else:
                mu_location, cov_location = m_location
                total_prob = 0
                for point in data:

                    if type_config == INTERACTION_MATTERS:
                        # loc model - type-based
                        m_type = MODELS_DICTIONARY[type_config][type_var_name][time_step]
                        prob = (1/len(data[tuple([data[type_var_name]==point[type_var_name]])])) #* m_type.mu[list(m_type.categories).index(point[type_var_name])]
                    else:
                        # loc model - none
                        # prob = stats.multivariate_normal.pdf([point[location_x_var_name], point[location_y_var_name]], mu_location, cov_location)
                        # uncomment for uniform null-location
                        prob = 1/len(data)

                        # uncomment for kde
                        # prob = kernel.pdf([point[location_x_var_name], point[location_y_var_name]])[0]



                    total_prob += prob
                    pmf_location[(point[location_x_var_name], point[location_y_var_name])] = prob

                # normalize pmf to get probabilities
                for key in pmf_location:
                    pmf_location[key] = np.exp(np.log(pmf_location[key]) - np.log(total_prob))




            # type model
            m_type = MODELS_DICTIONARY[type_config][type_var_name]
            pmf_type = {}

            if type_config == INTERACTION_MATTERS:
                if location_config == INTERACTION_MATTERS:
                    # type model - mixed
                    m_type = m_type[time_step]
                    pmf_type = {k: v for (k, v) in zip(m_type.categories, m_type.mu)}
                else:
                    # type model - type-based
                    m_type = m_type[time_step]
                    pmf_type = {k: v for (k, v) in zip(m_type.categories, m_type.mu)}
            else:

                # if location does not matter, just use null type model. otherwise multiply null-type model by location model
                if location_config == INTERACTION_DOES_NOT_MATTER:
                    # type model - none
                    pmf_type = {k: v for (k, v) in zip(*m_type)}
                else:
                    # type model - geo-based
                    # if location matters but type does not matter (geo-based):
                    # pmf_type = {k: (np.sum([pmf_location[(x[location_x_var_name], x[location_y_var_name])] for x in data[data[type_var_name]==k]])) for (k, v) in zip(*m_type)}
                    # pmf_type = {k: 1 for (k, v) in zip(*m_type)}
                    pmf_type = {k: v for (k, v) in zip(*m_type)}
                    total = np.sum([v for v in pmf_type.values()])


            #print((location_config, type_config))

            pmf_models[time_step][model_config][location_var_name] = pmf_location
            pmf_models[time_step][model_config][type_var_name] = pmf_type


            """
            if (location_config, type_config) == (True, False):
                #print("type|location")
                pmf_models[time_step][(True, False)] = pmf_type
            elif (location_config, type_config) == (False, True):
                #print("location|type")
                pmf_models[time_step][(False, True)] = pmf_location
            """


            # calculate p(C|M)
            log_model_evidence = 0
            click = observed_clicks[-1]
            # location attribute
            log_model_evidence += np.log(pmf_location[(click[location_x_var_name], click[location_y_var_name])])
            # type attribute
            log_model_evidence += np.log(pmf_type[click[type_var_name]])

            # the prior (previous time step)
            log_model_evidence += np.log(model_belief_dicts[time_step][model_config])
            model_evidence = np.exp(log_model_evidence)
            model_belief_dicts[time_step+1][model_config] = model_evidence

            # keep track of total for normalization purposes
            total_evidence = total_evidence + model_evidence

        # normalize
        for model_config in model_configs:
            model_belief_dicts[time_step+1][model_config] = np.exp(np.log(model_belief_dicts[time_step+1][model_config]) - np.log(total_evidence))

        printProgressBar(time_step+1, len(ui_data), prefix="Computing Model Evidences:")

    # Next click prediction and Target Recognition
    target_analysis = {}

    within_nums = [1, 5, 10, 20, 50, 100]
    ncp_showed_up = {w:0 for w in within_nums}

    printProgressBar(0, len(ui_data), prefix="Next click prediction: ")
    sorted_predictions_per_time = {}
    for time_step in range(len(ui_data)):

        observed_point = ui_data[time_step]

        probability_of_points = {(point[type_var_name], point[location_x_var_name], point[location_y_var_name]):
                                     sum([np.exp(np.log(model_belief_dicts[time_step][mc]) + int(mc[0]) * np.log(pmf_models[time_step][mc][location_var_name][(point[location_x_var_name], point[location_y_var_name])]) + int(mc[1]) * np.log(pmf_models[time_step][mc][type_var_name][point[type_var_name]])) for mc in model_configs])
                                 for point in data}


        sorted_points = [tuple(k) for k, v in sorted(probability_of_points.items(), key=lambda item: item[1])]
        sorted_predictions_per_time[time_step] = sorted_points
        target = [tuple(gt_data[[type_var_name, location_x_var_name, location_y_var_name]][k]) for k in range(len(gt_data))]

        multiple_of_gt_size = 3

        target_hat = sorted_points[-1 * multiple_of_gt_size * len(gt_data):]

        recall_vector = [p in target for p in target_hat]
        recall = sum(recall_vector)/len(target)

        precision_vector = [p in target for p in target_hat]
        precision = sum(precision_vector)/len(target_hat)

        coverage_vector = [p in target_hat for p in target]
        coverage = sum(coverage_vector)/len(target)

        printProgressBar(time_step+1, len(ui_data), prefix="Next Click Prediction: ")
        target_analysis[time_step] = {'recall': recall, 'precision': precision, 'coverage': coverage}

        for w in within_nums:
            ncp_showed_up[w] += int((observed_point[type_var_name], observed_point[location_x_var_name], observed_point[location_y_var_name]) in sorted_points[-1*w:])

    for w in within_nums:
        ncp_showed_up[w] /= len(ui_data)


    # print(pmf_models)
    return data_models, interaction_models, model_belief_dicts, pmf_models, target_analysis, ncp_showed_up, sorted_predictions_per_time



