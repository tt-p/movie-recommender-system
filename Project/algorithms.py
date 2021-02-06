"""
====================  ====================================================
Function              Description
====================  ====================================================
read_dataset_user     Reads data set from path returns a user dict.
read_dataset_item     Reads data set from path returns a movie dict.
user_based_sim        Returns similarity using Pearson correlation.
user_based_predict    Returns a prediction from parameters for a movie.
item_based_sim        Returns similarity using adjusted cosine similarity.
item_based_predict    Returns a prediction from parameters for a movie.
calculate_mae         Returns Mean Absolute Error from parameters.
====================  ====================================================
"""
import math
import statistics

from sklearn.metrics import mean_absolute_error as mae


def read_dataset(path, model):
    rating_dict = {}
    if model == "user":
        rating_dict = __read_dataset_user(path)
    elif model == "item":
        rating_dict = __read_dataset_item(path)
    else:
        raise Exception("Invalid model={0}. Use user or item".format(model))
    return rating_dict


def __read_dataset_user(path):
    """Reads data set from path returns user a dict.

        Parameters
        ----------
        path : str
            Absolute path of the MovieLens data set(u.data).

        Returns
        -------
        rating_dict : dict
            Returns a dict of users, movies and ratings.
            { string:user = { string:movie = int:rating } }
    """
    rating_dict = {}
    with open(path, "r") as reader:
        lines = reader.readlines()
        for line in lines:
            x = line.split(sep="\t")
            user = x[0]
            movie = x[1]
            rating = int(x[2])
            if user not in rating_dict:
                rating_dict[user] = dict()
                rating_dict[user][movie] = rating
            else:
                rating_dict[user][movie] = rating

    return rating_dict


def __read_dataset_item(path):
    """Reads data set from path returns a movie dict.

        Parameters
        ----------
        path : str
            Absolute path of the MovieLens data set(u.data).

        Returns
        -------
        rating_dict : dict
            Returns a dict of users, movies and ratings.
            { string:movie = { string:user = int:rating } }
    """
    rating_dict = {}
    with open(path, "r") as reader:
        lines = reader.readlines()
        for line in lines:
            x = line.split(sep="\t")
            user = x[0]
            movie = x[1]
            rating = int(x[2])
            if movie not in rating_dict:
                rating_dict[movie] = dict()
                rating_dict[movie][user] = rating
            else:
                rating_dict[movie][user] = rating

    return rating_dict


def user_based_sim(mean_data, user_data, u1, u2):
    """Returns a similarity using Pearson correlation.

        Parameters
        ----------
        mean_data : dict
            Dictionary of the users mean of ratings

        user_data : dict
            Dictionary of the data set.

        u1 : str
            Id of user to be used for similarity measure.

        u2 : str
            Id of user to be used for similarity measure.

        Returns
        -------
        similarity value : float
            Returns Pearson correlation of u1 and u2.
    """
    u1_movies = set([m1 for m1 in user_data[u1]])
    u2_movies = set([m2 for m2 in user_data[u2]])

    joint = u1_movies & u2_movies

    if len(joint) == 0:
        return 0

    mean1 = mean_data[u1]
    mean2 = mean_data[u2]

    numer = 0
    denom1 = 0
    denom2 = 0

    for m in joint:
        ra = user_data[u1][m]
        rb = user_data[u2][m]

        numer += (ra - mean1) * (rb - mean2)
        denom1 += math.pow(ra - mean1, 2)
        denom2 += math.pow(rb - mean2, 2)

    if denom1 == 0 or denom2 == 0:
        return 0

    return round(numer / (math.sqrt(denom1) * math.sqrt(denom2)), 15)


def item_based_sim(mean_data, movie_data, m1, m2):
    """Returns a similarity using adjusted cosine similarity.

        Parameters
        ----------
        mean_data : dict
            Dictionary of the users mean of ratings

        movie_data : dict
            Dictionary of the data set.

        m1 : str
            Id of movie to be used for similarity measure.

        m2 : str
            Id of movie to be used for similarity measure.

        Returns
        -------
        similarity value : float
            Returns adjusted cosine similarity of m1 and m2.
    """
    joint = set([u1 for u1 in movie_data[m1]]) & set([u2 for u2 in movie_data[m2]])

    if len(joint) == 0:
        return 0

    numer = 0
    denom1 = 0
    denom2 = 0

    for u in joint:
        ra = movie_data[m1][u]
        rb = movie_data[m2][u]

        mean = mean_data[u]

        numer += (ra - mean) * (rb - mean)
        denom1 += math.pow(ra - mean, 2)
        denom2 += math.pow(rb - mean, 2)

    if denom1 == 0 or denom2 == 0:
        return 0

    return round(numer / (math.sqrt(denom1) * math.sqrt(denom2)), 15)


def user_based_predict(mean_data, user_data, u1, m, nn_list):
    """Returns a prediction from parameters for a movie.

        Parameters
        ----------
        mean_data : dict
            Dictionary of the users mean of ratings

        user_data : dict
            Dictionary of the data set.

        u1 : str
            Id of user to be used for prediction.

        m : str
            Id of movie to be used for prediction.

        nn_list: list
            List of k-nearest-neighbors of u1.
            [ (user, similarity) ]

        Returns
        -------
        prediction : float
            Returns a rating.
    """
    numer = 0
    denom = 0
    mean1 = mean_data[u1]

    for tup in nn_list:
        u2 = tup[0]
        sim = tup[1]

        if m in user_data[u2]:
            r2 = user_data[u2][m]
        else:
            continue

        mean2 = mean_data[u2]

        numer += sim * (r2 - mean2)
        denom += sim

    if denom == 0:
        return round(mean1, 15)

    return round(mean1 + (numer / denom), 15)


def item_based_predict(movie_data, u, m1, nn_list):
    """Returns a prediction from parameters for a movie.

        Parameters
        ----------
        movie_data : dict
            Dictionary of the data set.

        u : str
            Id of user to be used for prediction.

        m1 : str
            Id of movie to be used for prediction.

        nn_list: list
            List of k-nearest-neighbors of m1.
            [ (movie, similarity) ]

        Returns
        -------
        prediction : float
            Returns a rating.
    """
    numer = 0
    denom = 0

    for tup in nn_list:
        m2 = tup[0]
        sim = tup[1]
        r_u = 0

        if u in movie_data[m2]:
            r_u = movie_data[m2][u]
        else:
            continue

        numer += sim * r_u
        denom += sim

    if denom == 0:
        return -1

    return round(numer / denom, 16)


def calculate_mae(rating_data, predict_data):
    """Returns Mean Absolute Error from parameters.

        Parameters
        ----------
        rating_data : dict
            Dictionary of the ratings.

        predict_data : dict
            Dictionary of the predictions.

        Returns
        -------
        prediction : float
            Returns Mean Absolute Error of rating_data and predict_data.
    """
    rating_list = []
    pred_list = []

    for user in predict_data:
        for movie in predict_data[user]:
            if predict_data[user][movie] == -1:
                continue
            else:
                rating_list.append(rating_data[user][movie])
                pred_list.append(predict_data[user][movie])

    return mae(rating_list, pred_list)