import numpy as np
import collections
import math
from scipy.stats import multivariate_normal
import scipy.special as sp
from sklearn.preprocessing import QuantileTransformer
import operator
'''
Let this block handle all the code related to Model classes and functions
'''


class Model:
    """
    Model is an abstract class which other models will extend. The purpose is to unify function definitions.
    """

    def __init__(self):
        raise Exception("calling abstract class, Model. Need to extent it first and define functions")

    def get_probability(self):
        raise NotImplementedError("calling abstract class, Model. Need to extend it first and define functions")


class MultivariateUniformModel(Model):
    def __init__(self, min_vec, max_vec, n=10):
        self.x_min = min_vec[0]
        self.y_min = min_vec[1]

        self.x_max = max_vec[0]
        self.y_max = max_vec[1]

        self.n = n

        self.delta_x = (self.x_max - self.x_min) / self.n
        self.delta_y = (self.y_max - self.y_min) / self.n

    def get_probability(self, observation):
        h = 1 / ((self.x_max - self.x_min) * (self.y_max - self.y_min))

        # pdf:
        # return h

        # cdf:
        return self.delta_x * self.delta_y * h

class MultivariateNormalLibModel(Model):
    """
    this model is a wrapper for scipy.stats.multivariate_normal
    """
    def __init__(self, mvn_model, min_vec, max_vec, n=10):
        self.model = mvn_model

        self.x_min = min_vec[0]
        self.y_min = min_vec[1]

        self.x_max = max_vec[0]
        self.y_max = max_vec[1]

        self.n = n

        self.delta_x = (self.x_max - self.x_min) / self.n
        self.delta_y = (self.y_max - self.y_min) / self.n
        #print("Multivariate Gaussian LIBRARY model created")

    def get_probability(self, observation):
        # print(observation)
        observation = list(observation[0])
        x = observation[0]
        y = observation[1]
        x_index = math.floor((x - self.x_min) / self.delta_x)
        y_index = math.floor((y - self.y_min) / self.delta_y)

        bound_1 = [self.x_min + x_index * self.delta_x, self.y_min + y_index * self.delta_y]
        bound_2 = [self.x_min + (x_index + 1) * self.delta_x, self.y_min + (y_index + 1) * self.delta_y]

        return self.model.cdf(bound_2) - self.model.cdf(bound_1)



class KDELibModel(Model):
    """
    this model is a wrapper for kde_gaussian
    """
    def __init__(self, kde_model, min_vec, max_vec, n=10):
        self.model = kde_model

        self.x_min = min_vec[0]
        self.y_min = min_vec[1]

        self.x_max = max_vec[0]
        self.y_max = max_vec[1]

        self.n = n

        self.delta_x = (self.x_max - self.x_min) / self.n
        self.delta_y = (self.y_max - self.y_min) / self.n
        #print("KDE Gaussian LIBRARY model created")

    def get_probability(self, observation):
        observation = list(observation)
        x = observation[0]
        y = observation[1]
        x_index = math.floor((x - self.x_min)/self.delta_x)
        y_index = math.floor((y - self.y_min)/self.delta_y)

        bound_1 = [self.x_min + x_index * self.delta_x, self.y_min + y_index * self.delta_y]
        bound_2 = [self.x_min + (x_index + 1) * self.delta_x, self.y_min + (y_index + 1) * self.delta_y]
        return self.model.integrate_box(bound_1, bound_2)


class TransformedNullContinuous:

    def __init__(self, data):
        self.d = [list(i) for i in data]
        self.transformation = QuantileTransformer(n_quantiles=len(self.d), output_distribution='normal').fit(self.d)
        self.transformed_d = np.array([list(self.transformation.transform(np.array(i).reshape(1, -1))[0]) for i in self.d])
        self.mu = np.mean(self.transformed_d, axis=0)
        self.cov = np.cov(np.transpose(self.transformed_d))

    def get_probability(self, observation):
        transformed_observation = self.transformation.transform(np.array(observation).reshape(1, -1))[0]
        return multivariate_normal.pdf(transformed_observation, self.mu, self.cov)

class MultivariateGaussianModel(Model):
    """
    Gaussian model with Normal-Wishart prior for continuous dimensions.
    k and v are hyper-parameters
    """

    def __init__(self, var_name, mu_0, T_0, k_0, v_0):

        self.var_name = var_name

        self.observations = None

        self.mu_0 = mu_0
        self.mu_n = mu_0

        self.T_0 = T_0
        self.T_n = T_0

        self.k_0 = k_0
        self.k_n = k_0

        self.v_0 = v_0
        self.v_n = v_0

        print("Multivariate Gaussian model created for:", var_name)

    def update_model(self, observation, debug=False):
        """
        This function is based on section 8.3 of the paper "Conjugate Bayesian analysis of the Gaussian distribution"
        by Kevin P. Murphy
        @:param observation is a structured list
        """

        # If this is the first observation, instantiate the observation list; otherwise update the list
        if self.observations is None:
            self.observations = np.array(observation)
        else:
            self.observations = np.append(self.observations, observation)

        # turn the observation structured array into a numpy array for arithmetic operations
        observations_array = self.observations.view(np.float).reshape(self.observations.shape + (-1,))

        # update the model if more than one observation has arrived
        if len(self.observations) > 1:
            n = len(observations_array)
            x_bar = np.mean(observations_array, axis=0)

            S = (n - 1) * np.cov(np.transpose(observations_array))

            self.T_n = self.T_0 + S + ((self.k_n * n) / (self.k_n + n)) * np.dot(np.transpose(np.matrix(self.mu_0 - x_bar)),
                                                                             np.matrix(self.mu_0 - x_bar))


            #self.T_n = self.T_0 + S + ((self.k_0 * n) / (self.k_0 + n)) * np.dot(np.transpose(np.matrix(self.mu_0 - x_bar)),
            #    np.matrix(self.mu_0 - x_bar))



            #self.mu_n = (self.k_n * self.mu_0 + n * x_bar) / (self.k_n + n)

            self.mu_n = (self.k_0 * self.mu_0 + n * x_bar) / (self.k_0 + n)

            self.v_n = self.v_0 + n
            self.k_n = self.k_0 + n

        if debug:
            print(self.var_name, "model updated; number of observations: ", len(self.observations))

    def get_probability(self, x, include_time=False):
        """
        According to Eq. 232 of the paper "Conjugate Bayesian analysis of the Gaussian distribution" by Kevin P. Murphy

        :param x: numpy structured array, the value of interest
        :param include_time:
        :return:
        """

        # convert x from structured array to a numpy array
        x_array = x.view(np.float).reshape(x.shape + (-1,))[0]

        # dimension of x
        d = len(x)

        if include_time:
            v = self.v_n - d + 1
            lamb = ((self.k_n + 1) / (self.k_n * (self.v_n - d + 1))) * self.T_n
            return multivariate_t_pdf(x_array, v, self.mu_n, lamb)
        else:
            d = d - 1
            v = self.v_n - d + 1
            lamb = ((self.k_n + 1) / (self.k_n * (self.v_n - d + 1))) * self.T_n[:-1, :-1]
            return multivariate_t_pdf(x_array[:-1], v, self.mu_n[:-1], lamb)

    def get_model_evidence(self):
        a = 1 / (np.pi ** (len(self.ui_data) * (len(self.mu_0) - 1)) / 2)
        b = sp.gamma(self.v_n / 2) * (np.linalg.det(self.T_0) ** (self.v / 2))
        c = sp.gamma(self.v / 2) * (np.linalg.det(self.T_n) ** (self.v_n / 2))
        d = (self.k / self.k_n) ** ((len(self.mu_0) - 1) / 2)

        res = a * (b / c) * d
        # print("cm get model evidence returns ", res)

        return res

    def get_log_maximum_likelihood(self):
        logp = 0
        mu_map, sigma_map = self.get_mode()
        distribution = multivariate_normal(mean=mu_map, cov=sigma_map)

        for observation_point in self.ui_data[:, :-1]:
            probability = distribution.pdf(observation_point)
            logp = logp + np.log(probability)

        return logp

    def get_log_null_likelihood(self):
        # pperpoint = 1 / len(self.data)
        # logp = len(self.ui_data) * np.log(pperpoint)

        logp = 0
        mu_null = self.mu_0[:-1]
        sigma_null = self.T_0[:-1, :-1]
        distribution = multivariate_normal(mean=mu_null, cov=sigma_null)

        for observation_point in self.ui_data[:, :-1]:
            probability = distribution.pdf(observation_point)
            logp = logp + np.log(probability)

        return logp


class NWModel(Model):
    """
    The old class from chi paper results

    Normal-Wishart model for continuous dimensions.
    k and v are parameters
    """

    v = 0
    k = 0

    df = 0

    mu_0 = None
    T_0 = None

    mu = None
    T = None

    df = 0

    data = None
    ui_data = None
    domains = None

    def __init__(self, data, k, v, name='unspecified name'):
        self.k = k
        self.v = v
        self.df = v - len(data.dtype.names) + 1
        self.data = data

        # domains of continuous data
        self.domains = get_domains(data)[0]

        self.ui_data = np.empty((0, len(self.domains) + 1))

        # find the mean of the domain for continuous dimentions
        self.mu_0 = np.array([np.mean(self.domains[dname]) for dname in self.domains.keys()])

        # add time dimension
        self.mu_0 = np.append(self.mu_0, 0)

        # for the starting covariance, we make a n+1 x n+1 matrix of zeros (extra dimension for time)
        self.T_0 = np.zeros((len(self.domains) + 1, len(self.domains) + 1))

        # time covariance
        self.T_0[len(self.domains), len(self.domains)] = 1

        d_cov = np.cov(np.transpose(np.array([list(point) for point in self.data]))) / self.v

        if len(self.domains) > 1:

            for i in range(len(d_cov)):
                for j in range(len(d_cov[i])):
                    # d is domain of ith dimension
                    d = self.domains[list(self.domains.keys())[i]]
                    # self.T_0[i, i] = (d[1] - d[0]) / 10
                    self.T_0[i, j] = d_cov[i][j]

        else:
            self.T_0[0, 0] = d_cov

        # self.T_0 = np.linalg.inv(self.T_0)

        # print(self.T_0)
        self.mu = self.mu_0
        self.T = self.T_0

        self.df = v - len(self.mu_0) + 1

        #print(self.T_0, 'prior_t\n')

        self.pmf_with_time = {tuple(point): 0 for point in self.data}
        self.pmf_without_time = {tuple(point): 0 for point in self.data}

        print("NW model created for ", name)
        # print(self.mu_0)
        # print(self.T_0)

    def update_model(self, observation):
        """

        @param observation is a dictionary
        """
        # add the observation to list
        #print(observation.dtype, self.domains.keys())
        new_observation_vector = [float(observation[k]) for k in self.domains.keys()]
        new_observation_vector.append(len(self.ui_data))
        self.ui_data = np.vstack([self.ui_data, new_observation_vector])

        # update the model if more than one observation has arrived

        d = len(self.mu)
        df = self.v - d + 1
        n = len(self.ui_data)


        #print('xbar  ', x_bar, '\n')



        #self.T_n = self.T_0 + S + ((self.k * n) / (self.k + n)) * np.dot(np.transpose(np.matrix(self.mu_0 - x_bar)),
        #                                                                 np.matrix(self.mu_0 - x_bar))
        if len(self.ui_data) > 1:

            x_bar = sum(self.ui_data) / n
            S = (n - 1) * np.cov(np.transpose(self.ui_data))
            self.T_n = self.T_0 + S + ((self.k * n) / (self.k + n)) * np.dot((self.mu_0 - x_bar).reshape(d, 1), (self.mu_0 - x_bar).reshape(1, d))
            new_loc = (self.k * self.mu_0 + n * x_bar) / (self.k + n)
        else:
            self.T_n = self.T_0 + ((self.k * n) / (self.k + n)) * np.dot((self.mu_0 - self.ui_data[0]).reshape(d, 1),
                                                                             (self.mu_0 - self.ui_data[0]).reshape(1, d))
            new_loc = (self.k * self.mu_0 + n * self.ui_data[0]) / (self.k + n)
        # print(self.T_n, '\n')
        self.v_n = self.v + n
        self.k_n = self.k + n

        new_scale = ((self.k_n + 1) / (self.k_n * (self.v_n - d + 1))) * self.T_n
        new_df = self.v_n - d + 1

        self.df = new_df
        self.mu = new_loc
        self.T = new_scale


        log_pdf_without_time = {tuple(point[list(self.domains.keys())]): self.get_log_pdf(point, include_time=False) for point in self.data}
        maxpoint = max(log_pdf_without_time.items(), key=operator.itemgetter(1))[1]

        for point in log_pdf_without_time.keys():
            log_pdf_without_time[point] -= maxpoint

        pdf_without_time = {point: np.exp(log_pdf_without_time[point]) for point in log_pdf_without_time.keys()}
        sum_of_pdf_without_time = sum([pdf_without_time[point] for point in pdf_without_time.keys()])
        self.pmf_without_time = {point: pdf_without_time[point]/sum_of_pdf_without_time for point in pdf_without_time.keys()}

        #maxp = max(self.pmf_without_time.items(), key=operator.itemgetter(1))
        #minp = min(self.pmf_without_time.items(), key=operator.itemgetter(1))
        #print('max is', maxp)
        #print('min is', minp)

        log_pdf_with_time = {tuple(point[list(self.domains.keys())]): self.get_log_pdf(point, include_time=True)
                                for point in self.data}
        maxpoint = max(log_pdf_with_time.items(), key=operator.itemgetter(1))[1]

        for point in log_pdf_with_time.keys():
            log_pdf_with_time[point] -= maxpoint

        pdf_with_time = {point: np.exp(log_pdf_with_time[point]) for point in log_pdf_with_time.keys()}
        sum_of_pdf_with_time = sum([pdf_with_time[point] for point in pdf_with_time.keys()])
        self.pmf_with_time = {point: pdf_with_time[point] / sum_of_pdf_with_time for point in
                                 pdf_with_time.keys()}

        maxp = max(self.pmf_without_time.items(), key=operator.itemgetter(1))
        minp = min(self.pmf_without_time.items(), key=operator.itemgetter(1))
        meanp = np.mean([self.pmf_without_time[point] for point in self.pmf_without_time.keys()])
        #print('max is', maxp)
        #print('min is', minp)
        #print('mean is', meanp)


        #print(self.mu)
        #print(self.T_n)
        #print(self.df)


        #print('test probability: %d' % self.pmf_without_time[tuple([-90.2549712400000, 38.7052049300000])])
        #print("model updated; number of observations: ", len(self.ui_data))

    def get_probability(self, x, include_time=False):
        # x is a dictionary
        new_x_vector = [float(x[k]) for k in self.domains.keys()]
        new_x_vector.append(len(self.ui_data))

        if include_time:
            #return t_pdf(new_x_vector, self.df, self.mu, self.T)
            precision_matrix = np.linalg.solve(self.T, np.identity(len(self.T)))
            return multivariate_t_pdf(new_x_vector, self.df, self.mu, precision_matrix)

        else:
            #return t_pdf(new_x_vector[:-1], self.df, self.mu[:-1], self.T[:-1, :-1])
            precision_matrix = np.linalg.solve(self.T[:-1, :-1], np.identity(len(self.T) - 1))
            return multivariate_t_pdf(new_x_vector[:-1], self.df+2, self.mu[:-1], precision_matrix)

    def get_probability_for_dims(self, x, dims):
        # x is a dictionary
        new_x_vector = [float(x[k]) for k in dims]

        indeces = [list(self.domains.keys()).index(d) for d in dims]

        #return t_pdf(new_x_vector, self.df, self.mu, self.T)
        precision_matrix = np.linalg.solve(self.T, np.identity(len(self.T)))[np.ix_(indeces, indeces)]
        mu = self.mu[np.ix_(indeces)]
        return multivariate_t_pdf(new_x_vector, self.df,mu, precision_matrix)




    def get_log_pdf(self, x, include_time=False):
        # x is a dictionary
        new_x_vector = [float(x[k]) for k in self.domains.keys()]
        new_x_vector.append(len(self.ui_data))

        if include_time:
            #return t_pdf(new_x_vector, self.df, self.mu, self.T)
            precision_matrix = np.linalg.solve(self.T, np.identity(len(self.T)))
            return multivariate_t_log_pdf(new_x_vector, self.df, self.mu, precision_matrix)

        else:
            #return t_pdf(new_x_vector[:-1], self.df, self.mu[:-1], self.T[:-1, :-1])
            precision_matrix = np.linalg.solve(self.T[:-1, :-1], np.identity(len(self.T) - 1))
            return multivariate_t_log_pdf(new_x_vector[:-1], self.df, self.mu[:-1], precision_matrix)



    def get_normalized_pmf(self, x, include_time=False):
        # x is a dictionary
        new_x_vector = [float(x[k]) for k in self.domains.keys()]
        #new_x_vector.append(len(self.ui_data))

        if include_time:
            #return t_pdf(new_x_vector, self.df, self.mu, self.T)
            return self.pmf_with_time[tuple(new_x_vector)]
        else:
            #return t_pdf(new_x_vector[:-1], self.df, self.mu[:-1], self.T[:-1, :-1])
            return self.pmf_without_time[tuple(new_x_vector)]



class DirichletModel(Model):
    """
    Dirichlet model for discrete dimensions
    alpha is the model parameter
    """

    def __init__(self, var_name, categories, alpha):

        self.var_name = var_name
        self.observations = None

        # hyper-parameter of categorical distribution
        self.alpha = alpha

        self.categories = categories

        self.m = np.zeros(len(self.categories))

        self.mu = (self.alpha + self.m) / (np.sum(self.alpha + self.m))
        # print(self.domains)
        # print(self.m)

        print('Dirichlet model created for:', var_name)

    def update_model(self, observation, debug=False):
        if self.observations is None:
            self.observations = np.array([observation])
        else:
            self.observations = np.append(self.observations, observation)

        self.m[list(self.categories).index(observation)] += 1
        self.mu = (self.alpha + self.m) / (np.sum(self.alpha + self.m))

        if debug:
            print(self.var_name, "model updated; number of observations: ", len(self.observations))



    def get_probability(self, x):
        return self.mu[list(self.categories).index(x)]

    def get_mode(self):
        return np.array([(mu2val - 1) / (sum(self.mu_2) - len(self.mu_2)) for mu2val in self.mu_2])

    def get_model_evidence(self):

        observation_counts = collections.Counter(self.ui_data)
        sorted_observation_counts = np.array([observation_counts[t] for t in dm.domains])

        a = (sp.gamma(sum(self.alpha))) / (np.prod(np.array([sp.gamma(a) for a in self.alpha])))
        b = np.prod(np.array([sp.gamma(sorted_observation_counts[j] + self.alpha[j]) for j in range(len(self.alpha))]))
        c = sp.gamma(len(self.ui_data) + np.sum(self.alpha))

        evidence = a * b / c
        return evidence

    def select_model(self):

        observation_counts = collections.Counter(self.ui_data)
        sorted_observation_counts = np.array([observation_counts[t] for t in dm.domains])
        # for m_1:
        m_1_evidence = np.prod(self.mu_1 ** sorted_observation_counts)
        m_1_evidence = np.sum(np.log(self.mu_1) * sorted_observation_counts)

        # for m_2:
        m_2_evidence = self.get_model_evidence()

        normalized_evidence = np.array([np.exp(m_1_evidence), m_2_evidence])
        normalized_evidence = normalized_evidence / np.sum(normalized_evidence)

        log_bayes_factor = np.log(m_1_evidence) - np.log(m_2_evidence)

        # if bayes_factor > 1, m_1 i.e. the data distribution is the better one, so the user
        # is probably not searching based on types.

        if log_bayes_factor > 0:
            print("selected model 1; exploration NOT type-based; bayes_factor=", log_bayes_factor)
            self.mu = self.mu_1

        else:
            print("selected model 2; exploration type-based; bayes_factor=", log_bayes_factor)
            self.mu = self.mu_2

        return log_bayes_factor, normalized_evidence

    def get_inferred_set(self, top_n_categories):
        if top_n_categories > len(self.domains):
            raise Exception("trying to get top k, where k > n")

        mu = self.mu
        top_cats = dm.domains[dm.mu.argsort()[-1 * top_n_categories:]]
        points = np.array([], dtype=self.full_data.dtype)

        for t in top_cats:
            pp = self.full_data[self.full_data[dm.names] == t]
            points = np.append(points, pp)

        return points

    def get_log_maximum_likelihood(self):
        mu_map = self.get_mode()

        observation_counts = collections.Counter(self.ui_data)
        sorted_observation_counts = np.array([observation_counts[t] for t in dm.domains])

        logmaxlikelihood = np.sum(np.log(mu_map) * sorted_observation_counts)
        return logmaxlikelihood

    def get_log_null_likelihood(self):
        observation_counts = collections.Counter(self.ui_data)
        sorted_observation_counts = np.array([observation_counts[t] for t in dm.domains])
        # for m_1:
        lognulllikelihood = np.sum(np.log(self.mu_1) * sorted_observation_counts)
        return lognulllikelihood


def multivariate_t_pdf(x, v, mu, lamb):
    """
    Multivariate t distribution, according to Eq. 2.162 in Bishop's PRML
    :param x: value of interest
    :param v: hyper-parameter
    :param mu: mean, center parameter
    :param lamb: Lambda, precision matrix; spread parameter
    :return: pdf value St(x | mu, lambda, nu)
    """
    # number of dimensions in x
    d = len(x)
    x = np.array(x)
    mu = np.array(mu)
    lamb = np.array(lamb)

    if v==0:
        print('warning: v=0; changed to v=1')
        v = 1

    # lamb = lamb.tolist()

    # final formula is (a/b)*c
    a = sp.gamma((v + d) / 2.0) * np.linalg.det(lamb) ** (1 / 2.0)
    b = sp.gamma(v / 2.0) * (v ** (d / 2.0)) * (math.pi ** (d / 2.0))
    c = (1 + ((1.0 / v) * float(np.dot((x - mu), np.dot(lamb, (x - mu)))))) ** (-(v + d) / 2.0)


    ans = (a / b) * c
    return ans


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


def multivariate_t_log_pdf(x, v, mu, lamb):
    """
    Multivariate t distribution, according to Eq. 2.162 in Bishop's PRML
    :param x: value of interest
    :param v: hyper-parameter
    :param mu: mean, center parameter
    :param lamb: Lambda, precision matrix; spread parameter
    :return: pdf value St(x | mu, lambda, nu)
    """
    # number of dimensions in x
    d = len(x)

    return sp.gammaln((d+v)/2) - sp.gammaln(v/2) + (1/2.0) * np.log(np.linalg.det(lamb)) - (d/2.0) * np.log(math.pi * v) + ((-d-v)/2) * np.log(1 + (1/v)*(float(np.dot((x - mu), np.dot(lamb, (x - mu))))))
