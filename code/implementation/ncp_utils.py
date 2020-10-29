import numpy as np


def predict_interactions_within_top(ks, models, observation):
    """
    does this interaction belong to the top-k set of points, using model averaging?
    :param ks: list of k values to consider
    :param models: list of models to average over
    :return: a dictionary {k: True/False}
    """
    showed_up = {k: None for k in ks}

    max_log_likelihood = max([models[m].get_log_model_evidence() for m in models.keys()])
    sum_of_likelihoods = sum([np.exp(models[m].get_log_model_evidence() - max_log_likelihood) for m in models.keys()])
    mp = {mname: np.exp(models[mname].get_log_model_evidence() - max_log_likelihood) / sum_of_likelihoods for mname in
          models.keys()}

    aggregate_probabilities = {point: sum([np.exp(np.log(mp[model]) + np.log(models[model].get_probability(point))) for model in models]) for point in models[list(models.keys())[0]].pdf.keys()}

    sorted_points = [tuple(k) for k, v in sorted(aggregate_probabilities.items(), key=lambda item: item[1])]

    print('sorted next clicks')
    print({ke: aggregate_probabilities[ke] for ke in sorted_points})

    # print('observed:', observed_point, '; sorted:', sorted_points[0])
    for k in ks:
        showed_up[k] = int(tuple(observation) in sorted_points[-1 * k:])

    return showed_up


def aggregate_interaction_predictions(individual_results, ks):
    """
    aggregates the results over all time stamps (>3) and returns the success rate
    :param individual_results: {t: {k: True/False}}
    :param ks: list of k values
    :return: {k: accuracy}
    """
    aggregated_ncp = {k: 0 for k in ks}
    total_clicks = 0

    for time_step in individual_results.keys():
        if time_step > 3:
            total_clicks += 1
            for k in ks:
                aggregated_ncp[k] += int(individual_results[time_step][k])

    for k in ks:
        aggregated_ncp[k] /= total_clicks

    return aggregated_ncp
