import argparse
import statistics

from algorithms import read_dataset, user_based_sim, item_based_sim, user_based_predict, item_based_predict

parser = argparse.ArgumentParser()
parser.add_argument("path", help="Absolute path of the dataset")
parser.add_argument("model", choices=["user", "item"], help="user or item")
parser.add_argument("knear", choices=[10, 20, 30, 40, 50, 60, 70, 80], type=int, help="k in k-nearest neighbors")
parser.add_argument("recom", type=int, help="number of recommendation")

user_rating_dict = {}
movie_rating_dict = {}


def get_unrated_movies(user):
    return set(movie_rating_dict.keys()).difference(set(user_rating_dict[user]))


def get_predictions_user(mean_data, user_data, k, u1, n):
    predictions = []
    unrated_movies = get_unrated_movies(u1)

    n_list = []

    for u2 in user_data:
        if u2 != u1:
            n_list.append((u2, user_based_sim(mean_data, user_data, u1, u2)))

    n_list.sort(reverse=True, key=lambda tup: tup[1])

    for m in unrated_movies:
        predictions.append((m, user_based_predict(mean_data, user_data, u1, m, n_list[:k])))

    predictions.sort(reverse=True, key=lambda tup: tup[1])

    return [p[0] for p in predictions[:n]]


def get_predictions_item(mean_data, movie_data, k, user, n):
    predictions = []
    unrated_movies = get_unrated_movies(user)

    sim = {}

    for m1 in unrated_movies:
        n_list = []
        for m2 in movie_data:

            if m1 == m2:
                continue
            else:
                if (m1, m2) in sim:
                    n_list.append((m2, sim[(m1, m2)]))
                elif (m2, m1) in sim:
                    n_list.append((m2, sim[(m2, m1)]))
                else:
                    res = item_based_sim(mean_data, movie_data, m1, m2)
                    sim[(m1, m2)] = res
                    n_list.append((m2, res))

        n_list.sort(reverse=True, key=lambda tup: tup[1])
        predictions.append((m1, item_based_predict(movie_data, user, m1, n_list[:k])))

    predictions.sort(reverse=True, key=lambda tup: tup[1])

    return [p[0] for p in predictions[:n]]


if __name__ == '__main__':
    args = parser.parse_args()
    user_rating_dict = read_dataset(args.path, "user")
    movie_rating_dict = read_dataset(args.path, "item")
    pred_list = []

    inp = input("Please enter user id :")

    means = {}

    for u in user_rating_dict:
        means[u] = statistics.mean([user_rating_dict[u][m] for m in user_rating_dict[u]])

    if args.model == "user":
        pred_list = get_predictions_user(means, user_rating_dict, int(args.knear), inp, int(args.recom))
    elif args.model == "item":
        pred_list = get_predictions_item(means, movie_rating_dict, int(args.knear), inp, int(args.recom))
    else:
        raise Exception(f"Invalid model={args.model}. Use user or item")

    print("Recommendations :")
    for i in range(1, len(pred_list) + 1):
        print("{0:2}. {1}".format(i, pred_list[i - 1]))
