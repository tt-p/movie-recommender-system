import argparse

from algorithms import *

from sklearn.model_selection import KFold


parser = argparse.ArgumentParser()
parser.add_argument("path", help="Absolute path of the dataset")
parser.add_argument("kfold", choices=[5, 10], type=int, help="k in k-fold cross validation")
parser.add_argument("knear", choices=[10, 20, 30, 40, 50, 60, 70, 80], type=int, help="k in k-nearest neighbors")
parser.add_argument("model", choices=["user", "item"], help="user or item")

user_sim = {}
movie_sim = {}


def user_based_experiment(mean_data, user_data, kf, knn):
    """Returns a prediction dict.

        Parameters
        ----------
        mean_data : dict
            Dictionary of the users mean of ratings

        user_data : dict
            Dictionary of the data set.
            { string:user = { string:movie = float:prediction } }

        kf : int
            k in k-fold cross validation.

        knn : int
            k in k-nearest neighbors.

        Returns
        -------
        pred_data : dict
            Returns a dict of users, movies and predictions.
            { string:user = { string:movie = float:prediction } }
    """
    pred_data = {}
    users = [u for u in user_data]
    kf = KFold(n_splits=kf)
    splits = kf.split(users)

    for train_index, test_index in splits:

        for a in test_index:
            ua = users[a]
            n_list = []
            for b in train_index:
                ub = users[b]

                if (ua, ub) in user_sim:
                    n_list.append((ub, user_sim[(ua, ub)]))
                elif (ub, ua) in user_sim:
                    n_list.append((ub, user_sim[(ub, ua)]))
                else:
                    res = user_based_sim(mean_data, user_data, ua, ub)
                    user_sim[(ua, ub)] = res
                    n_list.append((ub, res))

            n_list.sort(reverse=True, key=lambda tup: tup[1])
            knn_list = n_list[:int(knn)]

            m_pred = {}
            for m in user_data[ua]:
                m_pred[m] = user_based_predict(mean_data, user_data, ua, m, knn_list)

            pred_data[ua] = m_pred

    return pred_data


def item_based_experiment(mean_data, movie_data, kf, knn):
    """Returns a prediction dict.

            Parameters
            ----------
            mean_data : dict
                Dictionary of the users mean of ratings
                { string:movie = { string:user = float:rating } }
            movie_data : dict
                Dictionary of the data set.

            kf : int
                k in k-fold cross validation.

            knn : int
                k in k-nearest neighbors.

            Returns
            -------
            pred_data : dict
                Returns a dict of users, movies and predictions.
                { string:movie = { string:user = float:prediction } }
    """
    pred_data = {}
    movies = [m for m in movie_data]
    kf = KFold(n_splits=kf)
    splits = kf.split(movies)

    for train_index, test_index in splits:

        for a in test_index:
            ma = movies[a]
            n_list = []
            for b in train_index:
                mb = movies[b]

                if (ma, mb) in movie_sim:
                    n_list.append((mb, movie_sim[(ma, mb)]))
                elif (mb, ma) in movie_sim:
                    n_list.append((mb, movie_sim[(mb, ma)]))
                else:
                    res = item_based_sim(mean_data, movie_data, ma, mb)
                    movie_sim[(ma, mb)] = res
                    n_list.append((mb, res))

            n_list.sort(reverse=True, key=lambda tup: tup[1])
            knn_list = n_list[:int(knn)]

            u_pred = {}
            for u in movie_data[ma]:
                u_pred[u] = item_based_predict(movie_data, u, ma, knn_list)

            pred_data[ma] = u_pred

    return pred_data


if __name__ == '__main__':
    args = parser.parse_args()
    model = args.model
    user_rating_dict = read_dataset(args.path, "user")
    movie_rating_dict = read_dataset(args.path, "item")

    means = {}
    for user in user_rating_dict:
        means[user] = statistics.mean([user_rating_dict[user][movie] for movie in user_rating_dict[user]])

    if model == "user":
        pred_dict = user_based_experiment(means, user_rating_dict, args.kfold, args.knear)
        print(f"model = {args.model} | k-fold = {args.kfold} | k-near = {args.knear} | "
              f"mae = {round(calculate_mae(user_rating_dict, pred_dict), 4)}")
    elif model == "item":
        pred_dict = item_based_experiment(means, movie_rating_dict, args.kfold, args.knear)
        print(f"model = {args.model} | k-fold = {args.kfold} | k-near = {args.knear} | "
              f"mae = {round(calculate_mae(movie_rating_dict, pred_dict), 4)}")
    else:
        print("Error!")
