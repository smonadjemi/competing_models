import random
import numpy as np
import os
import pandas as pd
from datetime import datetime
import json


'''
Functions to load or generate data
'''


def load_data(dataset, **kwargs):
    if dataset == "synthetic":
        return _generate_synthetic_data(**kwargs)
    elif dataset == "crime":
        return _load_crime_data(**kwargs)
    elif dataset == "boardrooms":
        return _load_boardrooms_data(**kwargs)
    else:
        raise Exception("invalid dataset " + dataset)


def load_ui_data(dataset, task_type=None, group=None, **kwargs):
    if dataset == "synthetic":
        return _generate_synthetic_ui_data(task_type, **kwargs)
    elif dataset == "crime":
        return _load_crime_ui_data(task_type)
    elif dataset == "boardrooms":
        return _load_boardrooms_ui_data(group)
    elif dataset == "crime_gt":
        return _load_crime_ground_truth_data(task_type, **kwargs)
    else:
        raise Exception("invalid dataset " + dataset)




def _generate_synthetic_data(subdiv=40):
    '''
    Generate synthetic data in this function;
    We will have two columns of continuous variables, and one column of categorical.

    @return a structured numpy array
    '''

    types = ['a', 'b', 'c', 'd']
    xs = np.linspace(0, 1, subdiv)
    ys = np.linspace(3, 4, subdiv)

    '''
    This is the definition of our columns with their names and data types
    '''
    data = np.array([], dtype=[('x', 'float'), ('y', 'float'), ('type', 'U10')])

    '''
    populate the data array
    '''

    for xx in xs:
        for yy in ys:
            element = (xx, yy, random.choices(types, [0.1, 0.1, 0.7, 0.1])[0])
            data = np.append(data, np.array(element, dtype=data.dtype))

    return data


def _generate_synthetic_ui_data(task_type, data=None, number_of_clicks=40):
    """
    based on the task type (geo-based, type-based, mix), we assign a value to cont_propotions
    we generate random interaction data that simulate a certain type of task.

    @return an array of structured numpy array
    """
    ui_data = np.array([], dtype=[('time_stamp', 'i4'), ('x', 'float'), ('y', 'float'), ('type', 'U10')])

    if task_type == 'geo-based':
        # how likely the data will be geo based
        cont_proportion = 1

        # data pool from a certain geo area
        data_pool = data[(data['x'] < 0.2) & (data['x'] > 0) & (data['y'] < 4) & (data['y'] > 3.6)]

        # populate the user interaction clicks
        for k in range(number_of_clicks):
            if random.random() < cont_proportion:
                # choose a geo based point
                element = np.random.choice(data_pool)
            else:
                # choose a random point from the population
                element = np.random.choice(data)

            ui_data = np.append(ui_data, np.array([(k, *element)], dtype=ui_data.dtype))


    elif task_type == 'type-based':
        # how likely the data will be type based
        disc_proportion = 0.8

        # data pool from a certain geo area
        data_pool = data[data['type'] == 'c']

        # populate the user interaction clicks
        for k in range(number_of_clicks):
            if random.random() < disc_proportion:
                # choose a type based point
                element = np.random.choice(data_pool)
            else:
                # choose a random point from the population
                element = np.random.choice(data)

            ui_data = np.append(ui_data, np.array([(k, *element)], dtype=ui_data.dtype))

    elif task_type == 'mixed':
        # how likely the data will be geo based
        noise = 0.1

        # data pool from a certain geo area
        data_pool = data[
            (data['x'] < 0.2) & (data['x'] > 0) & (data['y'] < 4) & (data['y'] > 3.6) & (data['type'] == 'c')]

        # populate the user interaction clicks
        for k in range(number_of_clicks):
            if random.random() > noise:
                # choose a mixed - based point
                element = np.random.choice(data_pool)
            else:
                # choose a random point from the population as noise
                element = np.random.choice(data)

            ui_data = np.append(ui_data, np.array([(k, *element)], dtype=ui_data.dtype))

    elif task_type == 'diagonal':
        k = 0  # for time-stamp
        for element in data[np.round(data['y'], 3) == np.round(data['x'] + 3, 3)]:
            ui_data = np.append(ui_data, np.array([(k, *element)], dtype=ui_data.dtype))
            k += 1

    elif task_type == 'off-diagonal':
        k = 0  # for time-stamp
        for element in data[np.round(data['y'], 3) == np.round(4 - data['x'], 3)]:
            ui_data = np.append(ui_data, np.array([(k, *element)], dtype=ui_data.dtype))
            k += 1

    else:
        raise Exception("Invalid task type: " + task_type)

    return [ui_data]


def _load_crime_data():
    '''
    Load the crime data in this function

    @return structured numpy array
    '''

    # read in the data
    crime_data_path = "../../data/stl_crime/data.csv"
    crime_data = np.genfromtxt(crime_data_path, delimiter=',', skip_header=1)
    types = ['Homicide', 'Theft-Related', 'Assault', 'Arson', 'Fraud', 'Vandalism', 'Weapons', 'Vagrancy']

    data = np.array([(types[int(c[1])-1], c[3], c[2]) for c in crime_data],
                    dtype=[('type', 'U100'), ('lng', 'float'), ('lat', 'float')])

    return data


def _load_crime_ui_data(task):
    '''
    Load the interaction data from crime map

    @return an array of structured numpy arrays, each corresponding to one user.
    '''

    # since interaction data is references to actual data points (by crime id),
    # we need to load the crime data dictionary
    types = ['Homicide', 'Theft-Related', 'Assault', 'Arson', 'Fraud', 'Vandalism', 'Weapons', 'Vagrancy']
    crime_data_path = "../../data/stl_crime/data.csv"
    crime_data = np.genfromtxt(crime_data_path, delimiter=',', skip_header=1)
    # format: {id : (type, lat, lng)}
    crime_data_dictionary = {int(c[0]): (types[int(c[1])-1], c[3], c[2]) for c in crime_data}

    # read in the data
    interaction_data_path = "../../data/stl_crimes/ottley_experiment_interaction_data"
    user_interaction_data = {task_name: {int(participant_id.replace('.txt', '')): np.array([crime_data_dictionary[int(c)] for c in np.genfromtxt(
        interaction_data_path + '/' + task_name + '/' + participant_id)],
                                                  dtype=[('type', 'U100'), ('lng', 'float'), ('lat', 'float')])
                                         for participant_id in os.listdir(interaction_data_path + '/' + task_name)} for
                             task_name in os.listdir(interaction_data_path)}

    return user_interaction_data[task]


def _load_synthetic_crime_ui_data(num_clicks=15, crime_type=None):
    # generates synthetic type-based ui data
    full_data = _load_crime_data()
    if crime_type is None:
        crime_type = random.choice(np.unique(full_data['type']))
    sub_data = full_data[full_data['type']==crime_type]
    session = np.array(random.choices(sub_data, k=num_clicks), dtype=full_data.dtype)
    return session


def _load_crime_ground_truth_data(question):
    '''
        Load the interaction data from crime map

        @return an array of structured numpy arrays, each corresponding to one user.
        '''

    # since interaction data is references to actual data points (by crime id),
    # we need to load the crime data dictionary
    types = ['Homicide', 'Theft-Related', 'Assault', 'Arson', 'Fraud', 'Vandalism', 'Weapons', 'Vagrancy']
    crime_data_path = "../../data/stl_crime/data.csv"
    crime_data = np.genfromtxt(crime_data_path, delimiter=',', skip_header=1)
    # format: {id : (type, lat, lng)}
    crime_data_dictionary = {c[0]: (types[int(c[1]) - 1], c[3], c[2]) for c in crime_data}

    # read in the data
    interaction_data_path = "./data/targetPoints"
    user_interaction_data = np.array([crime_data_dictionary[c] for c in np.genfromtxt(
        interaction_data_path + '/' + question + '.txt')],
                                                  dtype=[('type', 'U10'), ('lng', 'float'), ('lat', 'float')])

    return [user_interaction_data]


def _load_boardrooms_data():
    '''
    Load the rental data in this function

    @return structured numpy array
    '''

    # read in the data
    data_path = "../../data/boardrooms/data.json"
    attributes_of_interest = ['idcompany', 'companyname', 'mktcap', 'unrelated', 'female', 'age', 'tenure', 'medianpay', 'industry']
    with open(data_path, 'r') as data_file:
        data = json.loads(data_file.read())
    data_file.close()

    data_array = np.array([tuple(row[attr] for attr in attributes_of_interest) for row in data],
                          dtype=[('idcompany', 'int'), ('companyname', 'U40'), ('mktcap', 'float'), ('unrelated', 'float'), ('female', 'float'), ('age', 'float'), ('tenure', 'float'), ('medianpay', 'float'), ('industry', 'U40')])

    return data_array

def _load_boardrooms_ui_data(group):
    '''
    Load the rental data in this function

    group can be:
    (1) those who had access to a search box and used it during exploration,
    (2) those who had access to the search box, but did not use it, and
    (3) those who did not have access to the search box.

    @return structured numpy array
    '''

    # read in the data
    data_path = "../../data/boardrooms/data.json"
    attributes_of_interest = ['idcompany', 'companyname', 'mktcap', 'unrelated', 'female', 'age', 'tenure', 'medianpay', 'industry']
    with open(data_path, 'r') as data_file:
        data = json.loads(data_file.read())
    data_file.close()

    data_array = np.array([tuple(row[attr] for attr in attributes_of_interest) for row in data],
                          dtype=[('idcompany', 'int'), ('companyname', 'U40'), ('mktcap', 'float'), ('unrelated', 'float'), ('female', 'float'), ('age', 'float'), ('tenure', 'float'), ('medianpay', 'float'), ('industry', 'U40')])

    data_dictionary = {row['idcompany']: row for row in data_array}

    # read in the user interaction data
    interaction_data_path = "../../data/boardrooms/feng_experiment_interaction_data/searchinvis-boardrooms-per-visit.csv"
    user_interaction_data = pd.read_csv(interaction_data_path)
    user_interaction_data = user_interaction_data.dropna(subset=['id', 'code', 'duration'])
    user_interaction_data = user_interaction_data[user_interaction_data['duration'] > 1000]

    # print(user_interaction_data)

    if group == 1:
        user_interaction_data = user_interaction_data[np.logical_and(user_interaction_data['condition'] == 'foresight', user_interaction_data['if_search_factor'] == 'search')]
    elif group == 2:
        user_interaction_data = user_interaction_data[np.logical_and(user_interaction_data['condition'] == 'foresight', user_interaction_data['if_search_factor'] == 'non-search')]
    elif group == 3:
        user_interaction_data = user_interaction_data[user_interaction_data['condition'] == 'control']
    else:
        raise Exception('group argument missing')

    interactions_by_participant = {}
    for index, row in user_interaction_data.iterrows():
        id = int(row['id'])
        if id not in interactions_by_participant.keys():
            interactions_by_participant[id] = []
        interactions_by_participant[id].append(int(row['code']))

    full_interactions_by_participant = {id: np.array([data_dictionary[cid] for cid in interactions_by_participant[id]], dtype=data_array.dtype) for id in interactions_by_participant.keys()}

    return full_interactions_by_participant
