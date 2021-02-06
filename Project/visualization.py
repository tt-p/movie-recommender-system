import argparse
import statistics
import matplotlib.pyplot as plt

from functools import reduce
from tabulate import tabulate
from algorithms import read_dataset

parser = argparse.ArgumentParser()
parser.add_argument("path", help="Absolute path of the dataset")


def dataset_summary(data):
    movies = set()
    for user in data:
        for movie in data[user]:
            movies.add(movie)

    table = [
        ["Total number of Users",
         len(data)],
        ["Total number of Movies",
         len(movies)],
        ["Total number of Ratings",
         reduce(lambda x, y: x + y, map(lambda x: len(x), data.values()))],
    ]
    print(tabulate(table, headers=["Summary of Dataset", ""]))
    print()


def rate_count_summary(data):
    rating_counts = list(map(lambda x: len(x), data.values()))

    table = [
        ["Mean of the Rating Counts", statistics.mean(rating_counts)],
        ["Median of the Rating Counts", statistics.median(rating_counts)],
        ["Mode of the Rating Counts", statistics.mode(rating_counts)],
        ["Minimum of the Rating Counts", min(rating_counts)],
        ["Maximum of the Rating Counts", max(rating_counts)]
    ]
    print(tabulate(table, headers=["Summary of Rating Counts", ""]))
    print()

    plt.figure(figsize=[8, 6])
    plt.title("Histogram of the Rating Counts")

    plt.hist(bins=200, x=rating_counts)
    plt.show()


def rate_value_summary(data):
    ratings = []
    for user in data:
        for movie in data[user]:
            ratings.append(data[user][movie])

    table = [
        ["Mean of the Ratings", statistics.mean(ratings)],
        ["Median of the Ratings", statistics.median(ratings)],
        ["Mode of the Ratings", statistics.mode(ratings)],
        ["Minimum of the Ratings", min(ratings)],
        ["Maximum of the Ratings", max(ratings)]
    ]
    print(tabulate(table, headers=["Summary of Ratings", ""]))
    print()

    plt.title("Box plot of the Ratings")

    plt.boxplot(ratings)
    plt.show()


def experiment_results():
    knn = [10, 20, 30, 40, 50, 60, 70, 80]
    user_mae = [0.7832, 0.7514, 0.7226, 0.6980, 0.6757, 0.6572, 0.6404, 0.6269]
    item_mae = [0.3711, 0.3721, 0.3739, 0.3754, 0.3797, 0.3827, 0.3881, 0.3968]

    plt.plot(knn, user_mae, c="green")
    plt.plot(knn, item_mae, c="blue")
    plt.plot(knn, user_mae, "g.", label="User Based")
    plt.plot(knn, item_mae, "b.", label="Item Based")
    plt.title("Experiment Results")
    plt.xlabel("k-NN")
    plt.ylabel("MAE")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    args = parser.parse_args()
    rating_dict = read_dataset(args.path, "user")
    dataset_summary(rating_dict)
    rate_count_summary(rating_dict)
    rate_value_summary(rating_dict)
    experiment_results()
