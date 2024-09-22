import numpy as np
from scipy.stats import qmc


class DataTransformer:
    def __init__(self, bounds, settings):
        self.bounds = bounds
        self.settings = settings
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    def apply_transformation(self, train_x, train_y):
        train_x_transformed = (
            self.normalize(train_x)
            if self.settings.get("normalize", False)
            else train_x
        )
        train_y_transformed = (
            self.standardize(train_y)
            if self.settings.get("standardize", False)
            else train_y
        )
        return train_x_transformed, train_y_transformed

    def inverse_transformation(self, train_x_transformed, train_y_transformed):
        train_x_original = (
            self.inverse_normalize(train_x_transformed)
            if self.settings.get("normalize", False)
            else train_x_transformed
        )
        train_y_original = (
            self.inverse_standardize(train_y_transformed)
            if self.settings.get("standardize", False)
            else train_y_transformed
        )
        return train_x_original, train_y_original

    def normalize(self, train_x):
        self.x_mean = self.bounds[0]
        self.x_std = self.bounds[1] - self.bounds[0]
        return (train_x - self.x_mean) / self.x_std

    def inverse_normalize(self, train_x_normalized):
        return train_x_normalized * self.x_std + self.x_mean

    def standardize(self, train_y):
        self.y_mean = np.mean(train_y)
        self.y_std = np.std(train_y)
        return (train_y - self.y_mean) / self.y_std

    def inverse_standardize(self, train_y_standardized):
        return train_y_standardized * self.y_std + self.y_mean



# Generate initial data based on Sobol samples
def generate_initial_data(objective, bounds, n=5, seed=None):
    # Generate Sobol sequences with a given seed
    sobol = qmc.Sobol(d=len(bounds[0]), seed=seed)  # Pass the seed to the Sobol constructor
    initial_x = sobol.random_base2(m=int(np.log2(n)))  # Generate n Sobol points
    
    # Scale Sobol points to the bounds
    initial_x = qmc.scale(initial_x, bounds[0], bounds[1])

    # Evaluate the objective function
    initial_y = objective(initial_x)

    return initial_x, initial_y
