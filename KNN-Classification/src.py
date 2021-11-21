import random
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

# Finding majority votes in a given sequence or array
def majority_vote(votes):
    """ Returns the element with most no. of votes. 
    Returns element at random in case of a tie."""
    vote_count = {}
    for vote in votes:
        if vote in vote_count:
            vote_count[vote] += 1
        else:
            vote_count[vote] = 1
            
    winner_element = []
    max_vote = max(vote_count.values())
    for element, count in vote_count.items():
        if count == max_vote:
            winner_element.append(element)
    return random.choice(winner_element)

# Distance between two points
def distance(p1, p2):
    """Calculate the distance between two points and returns it. """
    return np.sqrt(np.sum(np.square(p1 - p2)))

def find_nearest_neighbours(p, points, k=5):
    """Returns the k nearest neighbours to a point p in a set of points """
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p, points[i])
    ind = np.argsort(distances)
    return ind[:k]

def knn_predict(p, points, outcomes, k=5):
    """ Returns the class of point p given.
    returns a random value in case of a tie."""
    ind = find_nearest_neighbours(p, points, k=5)
    return majority_vote(outcomes[ind])

# Generating Synthetic Data

def synthetic_data(n=50):
    """ Create two sets of points from two bivariate distributions"""
    points = np.concatenate((ss.norm(0,1).rvs((n,2)), ss.norm(1,1).rvs((n,2))), axis = 0)
    outcomes = np.concatenate((np.repeat(0, n), np.repeat(1, n)), axis = 0)
    return (points, outcomes)

def make_prediction_grid(points, outcomes, limits, h, k):
    """Function will run through all the points in the prediction grid and predict the class for 
        each point and return them correspondingly."""
    
    (x_min, x_max, y_min, y_max) = limits
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xs, ys)
    prediction_grid = np.zeros(xx.shape, dtype = int)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            p = np.array([x,y])
            prediction_grid[j,i] = knn_predict(p, points, outcomes, k)
    return (xx, yy, prediction_grid)


def plot_prediction_grid (xx, yy, prediction_grid, filename):
    """ Plot KNN predictions for every point on the grid."""
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
    plt.scatter(points[:,0], points [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    plt.savefig(filename)
    
points, outcomes = synthetic_data()
limits = (-3, 4, -3, 4); h = 0.1; k = 5
xx, yy, prediction_grid = make_prediction_grid(points, outcomes, limits, h, k)

filename = "knn_prediction50.pdf"
plot_prediction_grid(xx, yy, prediction_grid, filename)

from sklearn import datasets
iris = datasets.load_iris()
outcomes = iris.target
