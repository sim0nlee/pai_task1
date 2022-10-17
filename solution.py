import os
import typing

from sklearn.cluster import KMeans
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm


# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = True
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation


# Cost function constants
COST_W_UNDERPREDICT = 25.0
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 10.0


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)
        self.kernel = None
        self.gp = None
        # TODO: Add custom initialization for your model here if necessary

    def make_predictions(self, test_features: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of locations.
        :param test_features: Locations as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        # TODO: Use your GP to estimate the posterior mean and stddev for each location here
        gp_mean, gp_std = self.gp.predict(test_features, return_std=True)

        # TODO: Use the GP posterior to form your predictions here
        predictions = gp_mean

        return predictions, gp_mean, gp_std

    def fitting_model(self, train_GT: np.ndarray, train_features: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_features: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_GT: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """
        # TODO: Fit your model here

        train_x, train_y = samp_reduce(train_features, train_GT, 2000)

        kernels = [
            RBF(1.0, (1e-5, 1e3)) + WhiteKernel(1.0, (1e-4, 1e2)),
            Matern(1.0, (1e-4, 1e4), nu=0.5) + WhiteKernel(1.0, (1e-4, 1e2)),
            Matern(1.0, (1e-4, 1e4), nu=1.5) + WhiteKernel(1.0, (1e-4, 1e2)),
            Matern(1.0, (1e-4, 1e4), nu=2.5) + WhiteKernel(1.0, (1e-4, 1e2)),
            RationalQuadratic(1.0, 1.0, (1e-5, 1e3), (1e-5, 1e3)) + WhiteKernel(1.0, (1e-4, 1e2)),
            # ExpSineSquared() + WhiteKernel(1.0, (1e-4, 1e2))
        ]

        self.kernel = self.cross_val(train_x, train_y, kernels, 3)

        self.gp = GaussianProcessRegressor(kernel=self.kernel,
                                           normalize_y=True,
                                           n_restarts_optimizer=5,
                                           copy_X_train=False,
                                           random_state=22)

        self.gp.fit(train_x, train_y)
        print('Kernel:', self.gp.kernel_)


    def cross_val(self, X, y, kernels, nfolds):
        """
        Custom 3-fold CV that uses one fold for training
        and two folds for validation due to GP inference complexity.
        CV on a set of potential kernel choices.
        :returns: CV results
        """

        fold_ids = np.arange(len(X))
        self.rng.shuffle(fold_ids)
        folds = np.array_split(fold_ids, nfolds)

        cv_scores = np.zeros((len(kernels), nfolds + 1))

        for k in range(0, len(kernels)):
            print()
            print('Testing kernel:', kernels[k])
            gp = GaussianProcessRegressor(kernel=kernels[k],
                                          normalize_y=True,
                                          n_restarts_optimizer=5,
                                          copy_X_train=False,
                                          random_state=99)
            fold_scores = np.zeros(nfolds)

            for i in range(0, nfolds):
                x_train = np.delete(X, folds[i], 0)
                y_train = np.delete(y, folds[i])

                x_val = X[folds[i], :]
                y_val = y[folds[i]]

                print('Training...')
                gp.fit(x_train, y_train)
                print('Kernel optimized params:', gp.kernel_)
                gp_mean, gp_std = gp.predict(x_val, return_std=True)
                y_pred = gp_mean

                cost = cost_function(y_val, y_pred)
                fold_scores[i] = cost
                print('Cost of Fold %i : %f' % (i + 1, cost))

            cv_scores[k] = np.append(fold_scores, np.mean(fold_scores))
        print('CV done for all kernels.\n', cv_scores)
        return kernels[np.argmin(cv_scores[:, nfolds-1])]


def samp_reduce(train_x, train_y, k=100):
    """
    Reduces the sample size via k-means clustering
    to deal with GP inference complexity.
    :k: desired sample size
    """

    cluster = KMeans(n_clusters=k)
    cluster.fit(train_x, train_y)

    train_y_new = np.empty((k,))
    for i in range(0, k):
        idx = (cluster.labels_ == i)
        train_y_new[i] = train_y[idx].mean()
    train_x_new = cluster.cluster_centers_
    return train_x_new, train_y_new





def cost_function(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1
    assert predictions.ndim == 1
    assert ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask_1 = predictions < ground_truth
    weights[mask_1] = COST_W_UNDERPREDICT

    # Case ii): significant overprediction
    mask_2 = (predictions >= 1.2*ground_truth)
    weights[mask_2] = COST_W_OVERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)



def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle('Extended visualization of task 1')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.make_predictions(visualization_xs)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_stddev = np.reshape(gp_stddev, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0
    vmax_stddev = 35.5

    # Plot the actual predictions
    ax_predictions = fig.add_subplot(1, 3, 1)
    predictions_plot = ax_predictions.imshow(predictions, vmin=vmin, vmax=vmax)
    ax_predictions.set_title('Predictions')
    fig.colorbar(predictions_plot)

    # Plot the raw GP predictions with their stddeviations
    ax_gp = fig.add_subplot(1, 3, 2, projection='3d')
    ax_gp.plot_surface(
        X=grid_lon,
        Y=grid_lat,
        Z=gp_mean,
        facecolors=cm.get_cmap()(gp_stddev / vmax_stddev),
        rcount=EVALUATION_GRID_POINTS_3D,
        ccount=EVALUATION_GRID_POINTS_3D,
        linewidth=0,
        antialiased=False
    )
    ax_gp.set_zlim(vmin, vmax)
    ax_gp.set_title('GP means, colors are GP stddev')

    # Plot the standard deviations
    ax_stddev = fig.add_subplot(1, 3, 3)
    stddev_plot = ax_stddev.imshow(gp_stddev, vmin=vmin, vmax=vmax_stddev)
    ax_stddev.set_title('GP estimated stddev')
    fig.colorbar(stddev_plot)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def main():
    # Load the training dateset and test features
    train_features = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_GT = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_features = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Fit the model
    print('Fitting model')
    model = Model()
    model.fitting_model(train_GT, train_features)

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.make_predictions(test_features)
    print('Predictions:', predictions[0])
    print('Mean:', predictions[1])
    print('Stddev:', predictions[2])

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
