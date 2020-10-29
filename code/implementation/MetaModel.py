"""
Having individual models over attributes causes us to have overhead calculations to
get the probability of each point.

The purpose of a meta model class is to take in all attributes and only output the overall probability of a point

Requires all points available, and needs an update function for interaction models.

"""


import Models
import numpy as np


class CombinedMetaModel:
    """
    The case where we put all important continuous variables in one gaussian, and ignore unimportant attributes
    """
    def __init__(self, data, continuous_attributes, discrete_attributes, name='untitled'):
        self.name = name
        self.data = data
        self.c_attr_list = continuous_attributes
        self.d_attr_list = discrete_attributes

        self.number_of_observations = 0

        self.pdf = {tuple(point): 1/len(self.data) for point in self.data}

        print('%s model created' % (self.name))

        if continuous_attributes:
            #self.k = 0.01
            self.k = len(self.c_attr_list)
            self.v = len(self.c_attr_list)
            self.c_model = Models.NWModel(self.data[self.c_attr_list], self.k, self.v, '%s c_model'%self.name)
        if discrete_attributes:
            constant = 0.0001
            self.d_models = {attr: Models.DirichletModel('%s for %s' % (attr, self.name), np.unique(self.data[attr]), constant * np.ones(np.unique(self.data[attr]).shape)) for attr in discrete_attributes}

        # penalize having too many attributes
        #self.log_model_likelihood = -np.log(len(continuous_attributes) + len(discrete_attributes) + 2)
        self.log_model_likelihood = 0
    def update(self, observation):
        """
        This function updates the underlying models first,
        then it updates the pdf by computing probabilities and normalizing them.
        :param observation:
        :return:
        """

        # print('updating ', observation)
        if self.c_attr_list:
            self.c_model.update_model(observation[self.c_attr_list])

        if self.d_attr_list:
            for d_attr in self.d_attr_list:
                self.d_models[d_attr].update_model(observation[d_attr])

        #if self.number_of_observations > 0:
        """
        updating the pdf
        """
        updated_pdf = {}
        for point in self.data:
                log_prob = 0

                if self.c_attr_list:
                    log_prob += np.log(self.c_model.get_normalized_pmf(point[self.c_attr_list], include_time=False))

                if self.d_attr_list:
                    for d_attr in self.d_attr_list:
                        log_prob += np.log(self.d_models[d_attr].get_probability(point[d_attr]) / sum([self.d_models[d_attr].get_probability(cat) * len(self.data[tuple([self.data[d_attr] == cat])]) for cat in self.d_models[d_attr].categories]))
                updated_pdf[tuple(point)] = log_prob



        """
        normalize and replace pdf
        """
        max_log_pdf = max([updated_pdf[point] for point in updated_pdf.keys()])
        updated_pdf = {point: np.exp(updated_pdf[point] - max_log_pdf) for point in updated_pdf.keys()}
        sum_probs = sum([updated_pdf[point] for point in updated_pdf.keys()])
        normalized_updated_pdf = {point: updated_pdf[point]/sum_probs for point in updated_pdf.keys()}
        self.pdf = normalized_updated_pdf

        print(self.name, self.number_of_observations, self.pdf)

        self.number_of_observations += 1
        self.log_model_likelihood += np.log(self.get_probability(observation))

    def get_probability(self, point):
        print('get prob called ', self.pdf[tuple(point)])
        return self.pdf[tuple(point)]

    def get_model_evidence(self):
        return np.exp(self.log_model_likelihood)

    def get_log_model_evidence(self):
        return self.log_model_likelihood


