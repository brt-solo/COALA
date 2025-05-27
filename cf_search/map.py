import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import random
import torch
device='cpu'
import numpy as np
import random
import statsmodels.api as sm
from sklearn.base import BaseEstimator

class mapcf_trend_intracell:
    """MAP-Elites with crossover and mutation - crossover only occurs between parents of the same cell"""

    def __init__(self, dim_map, dim_x, n_bins, max_evals, params, X_train_df, behavior_indices, wrapper, use_crossover=True):
        self.dim_map = dim_map
        self.dim_x = dim_x
        self.n_bins = n_bins
        self.max_evals = max_evals
        self.params = params
        self.use_crossover = use_crossover
        self.archive = {}
        self.X_train_df = X_train_df
        self.wrapper = wrapper
        self.behavior_indices = behavior_indices

        # Define min/max feature bounds
        self.feature_mins = np.array(params["min"]).astype(float)
        self.feature_maxs = np.array(params["max"]).astype(float)

        # Define grid bin edges
        self.bin_edges = [np.linspace(self.feature_mins[i], self.feature_maxs[i], n_bins + 1) for i in self.behavior_indices]

        # Find Binary Indices
        self.binary_indices = [
            self.X_train_df.columns.get_loc(col) for col in self.X_train_df.columns if set(self.X_train_df[col].dropna().unique()) == {0, 1}
        ]

    def evaluate_feature_vector(self, feature_vector):
        """
        Evaluates a feature vector using the trained PyTorch regression model.
        Returns:
        - Predicted lung function score (fitness)
        - A lower-dimensional descriptor (first 2 features)
        """
        # Convert to PyTorch tensor
        feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(device)

        # Ensure model is in evaluation mode
        self.wrapper.eval()

        with torch.no_grad():
            predicted_value = self.wrapper(feature_tensor).cpu().numpy().flatten()[0]  # Convert to NumPy

        # Use the first `dim_map` features as the descriptor
        descriptor = feature_vector[self.behavior_indices]  
        return predicted_value, descriptor  # Ensure correct shape

    def initialize_archive(self):
        """Initializes the archive with random candidates and stores them correctly."""
        for _ in range(int(self.params["random_init_batch"])):
            feature_vector = np.random.uniform(self.feature_mins, self.feature_maxs, self.dim_x)
            fitness, descriptor = self.evaluate_feature_vector(feature_vector)
            descriptor = np.array(descriptor).flatten()
            cell_index = tuple(np.clip(np.digitize(descriptor[i], self.bin_edges[i]) - 1, 0, self.n_bins - 1) for i in range(self.dim_map))

            if cell_index not in self.archive:
                self.archive[cell_index] = {"best": None, "population": []}

            self.archive[cell_index]["population"].append(feature_vector)

            if self.archive[cell_index]["best"] is None or fitness > self.archive[cell_index]["best"][1]:
                self.archive[cell_index]["best"] = (feature_vector, fitness)

    def select_parent1(self):
        """Selects the best individual from a randomly chosen cell in the archive."""
        if len(self.archive) == 0:
            raise ValueError("Archive is empty! Cannot select a parent.")

        cell_index = random.choice(list(self.archive.keys()))
        best_solution = self.archive[cell_index]["best"]
        return best_solution[0].copy()

    def select_parent2(self):
        """Selects a random individual from the archive's population."""
        if len(self.archive) == 0:
            raise ValueError("Archive is empty! Cannot select a parent.")

        cell_index = random.choice(list(self.archive.keys()))
        parent = random.choice(self.archive[cell_index]["population"])
        return parent.copy()

    def sbx_crossover(self, x, y):
        """Simulated Binary Crossover (SBX)."""
        eta = 10.0
        xl = np.array(self.params['min'])
        xu = np.array(self.params['max'])
        z = x.copy()
        r1 = np.random.random(size=len(x))
        r2 = np.random.random(size=len(x))

        for i in range(len(x)):
            if abs(x[i] - y[i]) > 1e-15:
                x1 = min(x[i], y[i])
                x2 = max(x[i], y[i])

                beta = 1.0 + (2.0 * (x1 - xl[i]) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)
                rand = r1[i]

                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

                c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

                beta = 1.0 + (2.0 * (xu[i] - x2) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)

                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

                c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

                c1 = min(max(c1, xl[i]), xu[i])
                c2 = min(max(c2, xl[i]), xu[i])

                z[i] = c2 if r2[i] <= 0.5 else c1

        z[self.binary_indices] = np.round(z[self.binary_indices])
        z[self.binary_indices] = np.clip(z[self.binary_indices], 0, 1)
        return z

    def mutate(self, parent, mutation_rate=0.6):
        """Applies mutation to a given parent solution."""
        mutation_mask = np.random.rand(*parent.shape) < mutation_rate
        mutation_step = np.random.uniform(-0.5, 0.5, size=parent.shape) * mutation_mask
        child = parent + mutation_step
        child = np.clip(child, self.feature_mins, self.feature_maxs)
        child[self.binary_indices] = np.round(child[self.binary_indices])
        child[self.binary_indices] = np.clip(child[self.binary_indices], 0, 1)
        return child

    def evaluate_and_store(self, child):
        """Evaluates and stores the child in the archive."""
        fitness, descriptor = self.evaluate_feature_vector(child, self.wrapper)
        cell_index = tuple(np.clip(np.digitize(descriptor[i], self.bin_edges[i]) - 1, 0, self.n_bins - 1) for i in range(self.dim_map))

        if cell_index not in self.archive:
            self.archive[cell_index] = {
                "best": (child, fitness),
                "population": [child]
            }
        else:
            self.archive[cell_index]["population"].append(child)
            if fitness > self.archive[cell_index]["best"][1]:
                self.archive[cell_index]["best"] = (child, fitness)

    def run(self):
        """Runs the MAP-Elites algorithm."""
        self.initialize_archive()

        for _ in range(int(self.max_evals - self.params["random_init_batch"])):
            try:
                parent_1 = self.select_parent1()
            except ValueError:
                print("ERROR: Archive is empty! Cannot select a parent.")
                return self.archive

            if self.use_crossover:
                try:
                    parent_2 = self.select_parent2()
                except ValueError:
                    print("ERROR: No second parent available for crossover.")
                    return self.archive

                child = self.sbx_crossover(parent_1, parent_2)
                child = self.mutate(child)
            else:
                child = self.mutate(parent_1)

            self.evaluate_and_store(child)

        return self.archive



class mapcf_trend_intercell:
    """ MAP-Elites implementation with mutation and crossover between parents of different cells"""

    def __init__(self, dim_map, dim_x, n_bins, max_evals, params, X_train_df, behavior_indices, wrapper, use_crossover=False):
        self.dim_map = dim_map
        self.dim_x = dim_x
        self.n_bins = n_bins
        self.max_evals = max_evals
        self.params = params
        self.use_crossover = use_crossover
        self.archive = {}
        self.X_train_df = X_train_df
        self.wrapper = wrapper
        self.behavior_indices = behavior_indices

        self.feature_mins = np.array(params["min"]).astype(float)
        self.feature_maxs = np.array(params["max"]).astype(float)
        
        self.bin_edges = [np.linspace(self.feature_mins[i], self.feature_maxs[i], n_bins + 1) for i in self.behavior_indices]

    def evaluate_feature_vector(self, feature_vector):
        """
        Evaluates a feature vector using the trained PyTorch regression model.
        Returns:
        - Predicted lung function score (fitness)
        - A lower-dimensional descriptor (first 2 features)
        """
        # Convert to PyTorch tensor
        feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(device)

        # Ensure model is in evaluation mode
        self.wrapper.eval()

        with torch.no_grad():
            predicted_value = self.wrapper(feature_tensor).cpu().numpy().flatten()[0]  # Convert to NumPy

        # Use the first `dim_map` features as the descriptor
        descriptor = feature_vector[self.behavior_indices]  
        return predicted_value, descriptor  # Ensure correct shape

    def sbx_crossover(self, x, y):
        """ Simulated Binary Crossover (SBX) """
        binary_indices = [self.X_train_df.columns.get_loc(col) for col in self.X_train_df.columns
                          if set(self.X_train_df[col].dropna().unique()) == {0, 1}]
        eta = 10.0  # Distribution parameter
        xl = self.feature_mins
        xu = self.feature_maxs
        z = x.copy()
        r1 = np.random.random(size=len(x))
        r2 = np.random.random(size=len(x))

        for i in range(len(x)):
            if abs(x[i] - y[i]) > 1e-15:
                x1, x2 = min(x[i], y[i]), max(x[i], y[i])
                beta = 1.0 + (2.0 * (x1 - xl[i]) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)
                rand = r1[i]

                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

                c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))
                beta = 1.0 + (2.0 * (xu[i] - x2) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)

                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

                c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))
                z[i] = c2 if r2[i] <= 0.5 else c1

        z[binary_indices] = np.round(z[binary_indices])  # Ensure binary features stay 0 or 1
        z[binary_indices] = np.clip(z[binary_indices], 0, 1)

        return z

    def mutate(self, parent, mutation_rate=0.6):
        """ Applies mutation to a given parent solution """
        binary_indices = [self.X_train_df.columns.get_loc(col) for col in self.X_train_df.columns
                          if set(self.X_train_df[col].dropna().unique()) == {0, 1}]

        mutation_mask = np.random.rand(*parent.shape) < mutation_rate
        mutation_step = np.random.uniform(-0.5, 0.5, size=parent.shape) * mutation_mask
        child = parent + mutation_step
        child = np.clip(child, self.feature_mins, self.feature_maxs)

        child[binary_indices] = np.round(child[binary_indices])
        child[binary_indices] = np.clip(child[binary_indices], 0, 1)
        child[binary_indices] = child[binary_indices].astype(int)

        return child

    def initialize_archive(self):
        """ Initializes the archive with random candidates """
        for _ in range(int(self.params["random_init_batch"])):
            feature_vector = np.random.uniform(self.feature_mins, self.feature_maxs, self.dim_x)
            fitness, descriptor = self.evaluate_feature_vector(feature_vector)
            descriptor = np.array(descriptor).flatten()
            cell_index = tuple(np.clip(np.digitize(descriptor[i], self.bin_edges[i]) - 1, 0, self.n_bins - 1)
                               for i in range(self.dim_map))

            if cell_index not in self.archive:
                self.archive[cell_index] = {"best": None, "population": []}

            self.archive[cell_index]["population"].append(feature_vector)

            if self.archive[cell_index]["best"] is None or fitness > self.archive[cell_index]["best"][1]:
                self.archive[cell_index]["best"] = (feature_vector, fitness)

    def select_two_parents(self):
        """ Selects two parents from different cells """
        if len(self.archive) < 2:
            raise ValueError("Not enough cells in archive to perform crossover.")

        cells = list(self.archive.keys())
        cell_1, cell_2 = random.sample(cells, 2)

        parent_1 = self.archive[cell_1]["best"][0]
        parent_2 = self.archive[cell_2]["best"][0]

        return parent_1.copy(), parent_2.copy()

    def evaluate_and_store(self, child):
        """ Evaluates a child and stores it in the correct cell """
        fitness, descriptor = self.evaluate_feature_vector(child)
        descriptor = np.array(descriptor).flatten()
        new_cell_index = tuple(np.clip(np.digitize(descriptor[i], self.bin_edges[i]) - 1, 0, self.n_bins - 1)
                               for i in range(self.dim_map))

        if new_cell_index not in self.archive:
            self.archive[new_cell_index] = {"best": (child, fitness), "population": [child]}
        else:
            self.archive[new_cell_index]["population"].append(child)
            if fitness > self.archive[new_cell_index]["best"][1]:
                self.archive[new_cell_index]["best"] = (child, fitness)

    def run(self):
        """ Runs the MAP-Elites algorithm """
        self.initialize_archive()

        for i in range(int(self.max_evals - self.params["random_init_batch"])):
            
            try:
                if self.use_crossover:
                    parent_1, parent_2 = self.select_two_parents()
                    child = self.sbx_crossover(parent_1, parent_2)
                    child = self.mutate(parent_1)  # Mutation
                else:
                    parent_1 = self.select_parent1()
                    child = self.mutate(parent_1)  # Mutation
            except ValueError:
                print("ERROR: Archive is empty! Cannot select parents.")
                return self.archive

            self.evaluate_and_store(child)


        return self.archive

import numpy as np
import random
class mapcf_instance:
    """MAP-Elites where crossover and mutation happen within the same cell."""

    def __init__(self, dim_map, dim_x, max_evals, params, cell_feature_sets, X_train_df, feature_categories, X_reference, wrapper):
        self.dim_map = dim_map
        self.dim_x = dim_x
        self.max_evals = max_evals
        self.params = params
        self.cell_feature_sets = cell_feature_sets
        self.archive = {}
        self.X_train_df = X_train_df
        self.feature_categories = feature_categories
        self.X_reference = X_reference
        self.wrapper = wrapper

        self.feature_mins = np.array(params["min"]).astype(float)
        self.feature_maxs = np.array(params["max"]).astype(float)

        self.binary_features = [
            col for col in self.X_train_df.columns 
            if set(self.X_train_df[col].dropna().unique()) == {0, 1}
        ]
        self.feature_indices = {feature: list(self.X_train_df.columns).index(feature) for feature in self.X_train_df.columns}

        self.is_sklearn_model = isinstance(wrapper, BaseEstimator)

    def evaluate_batch(self, feature_matrix):
        feature_matrix = np.array(feature_matrix, dtype=np.float32)
        
        if self.is_sklearn_model:
            return self.wrapper.predict(feature_matrix)
        else:
            feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32).to(device)
            self.wrapper.eval()
            with torch.no_grad():
                predictions = self.wrapper(feature_tensor).cpu().numpy().flatten()
            return predictions


    def evaluate_instance(self, feature_vector):
        if self.is_sklearn_model:
            feature_vector = np.array(feature_vector).reshape(1, -1)
            predicted_value = self.wrapper.predict(feature_vector)[0]
        else:
            feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(device)
            self.wrapper.eval()
            with torch.no_grad():
                predicted_value = self.wrapper(feature_tensor).cpu().numpy().flatten()[0]
        return predicted_value

    def ensure_binary_validity(self, child):
        for feature in self.binary_features:
            index = self.feature_indices[feature]
            child[index] = np.clip(round(child[index]), 0, 1)
        return child

    def sbx_crossover(self, x, y, mutable_features):
        eta = 1.0
        xl = np.array(self.params['min'])
        xu = np.array(self.params['max'])
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        child = x.copy()
        r1 = np.random.random(size=len(mutable_features))
        r2 = np.random.random(size=len(mutable_features))

        for idx, feature in enumerate(mutable_features):
            feature_index = self.feature_indices[feature]

            if abs(x[feature_index] - y[feature_index]) > 1e-15:
                x1, x2 = min(x[feature_index], y[feature_index]), max(x[feature_index], y[feature_index])
                beta = 1.0 + (2.0 * max(0, x1 - xl[feature_index])) / max(1e-15, (x2 - x1))
                alpha = max(1.0, 2.0 - beta ** -(eta + 1))
                rand = r1[idx]
                if (2.0 - rand * alpha) > 0:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                else:
                    beta_q = 1.0
                c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))
                c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))
                c1 = min(max(c1, xl[feature_index]), xu[feature_index])
                c2 = min(max(c2, xl[feature_index]), xu[feature_index])
                child[feature_index] = c2 if r2[idx] <= 0.5 else c1

        child = np.clip(child, xl, xu)
        return child

    def mutate(self, child, mutable_features, mutation_strength=0.2):
        mutation_step = np.zeros_like(child)
        for feature in mutable_features:
            if feature not in self.binary_features:
                feature_index = self.feature_indices[feature]
                mutation_step[feature_index] = np.random.uniform(-mutation_strength, mutation_strength)
        child += mutation_step
        return child

    def initialize_archive(self):
        batch_vectors = []
        batch_cells = []
        batch_size = int(self.params["random_init_batch"])

        for _ in range(batch_size):
            i, j = sorted(np.random.choice(range(len(self.feature_categories)), size=2, replace=True))
            cell_index = (i, j)

            if cell_index not in self.cell_feature_sets:
                continue

            mutable_features = self.cell_feature_sets[cell_index]
            feature_vector = self.X_reference.copy()

            for feature in mutable_features:
                feature_index = self.feature_indices[feature]
                feature_vector[feature_index] = np.random.uniform(self.feature_mins[feature_index], self.feature_maxs[feature_index])

            batch_vectors.append(feature_vector)
            batch_cells.append(cell_index)

        batch_fitnesses = self.evaluate_batch(batch_vectors)

        for vec, fit, cell_index in zip(batch_vectors, batch_fitnesses, batch_cells):
            if cell_index not in self.archive:
                self.archive[cell_index] = {"best": None, "population": []}
            self.archive[cell_index]["population"].append((vec, fit))
            if self.archive[cell_index]["best"] is None or fit > self.archive[cell_index]["best"][1]:
                self.archive[cell_index]["best"] = (vec, fit)

    def select_parents(self, cell_index):
        population = self.archive[cell_index]["population"]
        parent_1 = max(population, key=lambda vec_fitness: vec_fitness[1])[0]
        parent_2 = random.choice(population)[0]
        return parent_1.copy(), parent_2.copy()

    def evaluate_and_store(self, child, cell_index):
        fitness = self.evaluate_instance(child)
        if cell_index not in self.archive:
            self.archive[cell_index] = {"best": (child, fitness), "population": [(child, fitness)]}
        else:
            self.archive[cell_index]["population"].append((child, fitness))
            if fitness > self.archive[cell_index]["best"][1]:
                self.archive[cell_index]["best"] = (child, fitness)

    def run(self):
        self.initialize_archive()

        batch_size = 512  # ✅ you can set 32 or 64
        child_batch = []
        cell_batch = []

        total_steps = int(self.max_evals - self.params["random_init_batch"])

        for a in range(total_steps):
            #print(a)

            i, j = sorted(np.random.choice(range(len(self.feature_categories)), size=2, replace=True))
            cell_index = (i, j)

            if cell_index not in self.archive:
                continue

            mutable_features = self.cell_feature_sets[cell_index]
            parent_1, parent_2 = self.select_parents(cell_index)

            child = self.sbx_crossover(parent_1, parent_2, mutable_features)
            # child = self.mutate(child, mutable_features)
            child = self.ensure_binary_validity(child)

            child_batch.append(child)
            cell_batch.append(cell_index)

            # 🧠 When batch is full or last step
            if len(child_batch) == batch_size or a == total_steps - 1:
                fitness_batch = self.evaluate_batch(child_batch)

                for child_vec, fitness, cell in zip(child_batch, fitness_batch, cell_batch):
                    if cell not in self.archive:
                        self.archive[cell] = {"best": (child_vec, fitness), "population": [(child_vec, fitness)]}
                    else:
                        self.archive[cell]["population"].append((child_vec, fitness))
                        if fitness > self.archive[cell]["best"][1]:
                            self.archive[cell]["best"] = (child_vec, fitness)

                # Clear batches
                child_batch = []
                cell_batch = []

        return self.archive


class mapcf_instance_single:
    """MAP-Elites where crossover and mutation happen within the same cell."""

    def __init__(self, dim_map, dim_x, max_evals, params, cell_feature_sets, X_train_df, feature_categories, X_reference, wrapper):
        self.dim_map = dim_map
        self.dim_x = dim_x
        self.max_evals = max_evals
        self.params = params
        self.cell_feature_sets = cell_feature_sets
        self.archive = {}
        self.X_train_df = X_train_df
        self.feature_categories = feature_categories
        self.X_reference = X_reference
        self.wrapper = wrapper

        self.feature_mins = np.array(params["min"]).astype(float)
        self.feature_maxs = np.array(params["max"]).astype(float)

        self.binary_features = [
            col for col in self.X_train_df.columns 
            if set(self.X_train_df[col].dropna().unique()) == {0, 1}
        ]
        self.feature_indices = {feature: list(self.X_train_df.columns).index(feature) for feature in self.X_train_df.columns}

        self.is_sklearn_model = isinstance(wrapper, BaseEstimator) #checks if the model is sklearn based or pytorch

    def evaluate_instance(self, feature_vector):
        """
        Evaluates a feature vector using the trained PyTorch regression model.
        Returns:
        - Predicted lung function score (fitness)
        """
        if self.is_sklearn_model:
            feature_vector = np.array(feature_vector).reshape(1, -1)
            predicted_value = self.wrapper.predict(feature_vector)[0]
        else:
            feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(device)
            self.wrapper.eval()
            with torch.no_grad():
                predicted_value = self.wrapper(feature_tensor).cpu().numpy().flatten()[0]
        return predicted_value  # Ensure correct shape
    
    def ensure_binary_validity(self, child):
        """Ensures binary features remain valid (0 or 1)."""
        for feature in self.binary_features:
            index = self.feature_indices[feature]
            child[index] = np.clip(round(child[index]), 0, 1)
        return child

    
    def sbx_crossover(self, x, y, mutable_features):
        """
        Simulated Binary Crossover (SBX), applied only to mutable features.

        - Uses `eta` parameter to control offspring distribution.
        - Ensures crossover affects **only** mutable features.
        - Keeps binary features (0 or 1) intact.
        """
        eta = 1.0  # Distribution parameter
        xl = np.array(self.params['min'])
        xu = np.array(self.params['max'])
        
        # Ensure x and y are NumPy arrays
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        
        child = x.copy()  # Start with a copy of the first parent
        r1 = np.random.random(size=len(mutable_features))
        r2 = np.random.random(size=len(mutable_features))

        for idx, feature in enumerate(mutable_features):
            feature_index = self.feature_indices[feature] # Convert feature name to column index

            if abs(x[feature_index] - y[feature_index]) > 1e-15:
                x1, x2 = min(x[feature_index], y[feature_index]), max(x[feature_index], y[feature_index])

                beta = 1.0 + (2.0 * max(0, x1 - xl[feature_index])) / max(1e-15, (x2 - x1))
                alpha = max(1.0, 2.0 - beta ** -(eta + 1))
                rand = r1[idx]

                if (2.0 - rand * alpha) > 0:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                else:
                    beta_q = 1.0  # Avoid NaN issues

                c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

                ''' below is redundant
                beta = 1.0 + (2.0 * max(0, xu[feature_index] - x2)) / max(1e-15, (x2 - x1))
                alpha = max(1.0, 2.0 - beta ** -(eta + 1))

                if (2.0 - rand * alpha) > 0:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                else:
                    beta_q = 1.0  
                '''
                c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

                c1 = min(max(c1, xl[feature_index]), xu[feature_index])
                c2 = min(max(c2, xl[feature_index]), xu[feature_index])

                child[feature_index] = c2 if r2[idx] <= 0.5 else c1
        child = np.clip(child, xl, xu)

        return child


    def mutate(self, child, mutable_features, mutation_strength=0.2):
        """Mutation applied to the child after crossover (skip binary features)."""
        mutation_step = np.zeros_like(child)  
        '''
        mutation_values = np.random.uniform(-mutation_strength, mutation_strength, size=len(mutable_features))  

        for i, feature in enumerate(mutable_features):
            feature_index = list(self.X_train_df.columns).index(feature)
            mutation_step[feature_index] = mutation_values[i]
        '''
        for feature in mutable_features:
            if feature not in self.binary_features:  # Only mutate continuous
                feature_index = self.feature_indices[feature]
                mutation_step[feature_index] = np.random.uniform(-mutation_strength, mutation_strength)

        child += mutation_step
        return child

    def initialize_archive(self):
        """Initialize the archive with random feature vectors."""
        for _ in range(int(self.params["random_init_batch"])):
            i, j = sorted(np.random.choice(range(len(self.feature_categories)), size=2, replace=True))  
            cell_index = (i, j)

            if cell_index not in self.cell_feature_sets:
                print(f"ERROR: Cell {cell_index} not found in cell_feature_sets!")
                continue  # Skip invalid cells

            mutable_features = self.cell_feature_sets[cell_index]
            feature_vector = self.X_reference.copy()  # Base instance

            for feature in mutable_features:
                feature_index = list(self.X_train_df.columns).index(feature)
                feature_vector[feature_index] = np.random.uniform(self.feature_mins[feature_index], self.feature_maxs[feature_index])

            fitness = self.evaluate_instance(feature_vector)

            if cell_index not in self.archive:
                self.archive[cell_index] = {"best": None, "population": []}

            self.archive[cell_index]["population"].append((feature_vector, fitness))

            if self.archive[cell_index]["best"] is None or fitness > self.archive[cell_index]["best"][1]:
                self.archive[cell_index]["best"] = (feature_vector, fitness)

    def select_parents(self, cell_index):
        """Select best parent and random good parent from top 10% (fast)."""
        '''
        sorted_population = sorted(
            self.archive[cell_index]["population"],
            key=lambda vec_fitness: vec_fitness[1],  # Sort by precomputed fitness
            reverse=True
        )

        parent_1 = sorted_population[0][0]  # Take feature vector only
        #k = max(1, int(0.5 * len(sorted_population)))  # Top 10%
        #candidates = sorted_population[:k]
        
        parent_2 = random.choice(sorted_population)[0]  # Take feature vector only
        #parent_2 = sorted_population[1][0]
        
        parent_1 = self.archive[cell_index]["best"][0]
        parent_2 = random.choice(self.archive[cell_index]["population"])[0] 
        '''
        population = self.archive[cell_index]["population"]
        parent_1 = max(population, key=lambda vec_fitness: vec_fitness[1])[0]  # Best individual
        parent_2 = random.choice(population)[0]  # Random second parent

        return parent_1.copy(), parent_2.copy()


    def evaluate_and_store(self, child, cell_index):
        """Evaluates and stores the child in the archive."""
        fitness = self.evaluate_instance(child)

        if cell_index not in self.archive:
            self.archive[cell_index] = {"best": (child, fitness), "population": [(child, fitness)]}
        else:
            self.archive[cell_index]["population"].append((child, fitness))
            if fitness > self.archive[cell_index]["best"][1]:
                self.archive[cell_index]["best"] = (child, fitness)

    def run(self):
        """Runs MAP-Elites where crossover and mutation happen within the same cell."""
        self.initialize_archive()

        # Evolutionary Search Loop
        for a in range(int(self.max_evals - self.params["random_init_batch"])):
            print(a)
            i, j = sorted(np.random.choice(range(len(self.feature_categories)), size=2, replace=True))  
            cell_index = (i, j)

            if cell_index not in self.archive:
                continue  # Skip missing cells

            mutable_features = self.cell_feature_sets[cell_index]

            # Select two parents from the same cell
            parent_1, parent_2 = self.select_parents(cell_index)

            # Apply SBX crossover
            child = self.sbx_crossover(parent_1, parent_2, mutable_features)

            # Apply mutation
            #child = self.mutate(child, mutable_features)

            # Ensure binary feature validity
            child = self.ensure_binary_validity(child)

            # Store the new solution
            self.evaluate_and_store(child, cell_index)

        return self.archive




if __name__ == "__main__":
    import pandas as pd
    from pandas import Series, DataFrame
    #import matplotlib.pyplot as plt
    import sklearn
    import math
    import pickle


    from IPython.display import display
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import StandardScaler


    import torch
    import torch.nn as nn
    import torch.optim as optim

    ### defines a function that displays the counterfactuals
    def display_df(df, show_only_changes, save_to=None):
        from IPython.display import display
        if show_only_changes is False:
            display(df)  # works only in Jupyter notebook
        else:
            newdf = df.values.tolist()
            org = query_instance.values.tolist()[0]
            for ix in range(df.shape[0]):
                for jx in range(len(org)):
                    if not isinstance(newdf[ix][jx], str):
                        if math.isclose(newdf[ix][jx], org[jx], rel_tol=abs(org[jx]/10000)):
                            newdf[ix][jx] = '-'
                        else:
                            newdf[ix][jx] = str(newdf[ix][jx])
                    else:
                        if newdf[ix][jx] == org[jx]:
                            newdf[ix][jx] = '-'
                        else:
                            newdf[ix][jx] = str(newdf[ix][jx])
            df_changes = pd.DataFrame(newdf, columns=df.columns, index=df.index)
            if save_to:
                file_extension = save_to.split('.')[-1].lower()
                if file_extension == 'csv':
                    df_changes.to_csv(save_to, index=True)
                elif file_extension == 'xlsx':
                    df_changes.to_excel(save_to, index=True)
                elif file_extension == 'json':
                    df_changes.to_json(save_to)
            return df_changes

    ### defines function for finding continuous and categorical variables
    def find_binary_columns(df):
        binary_columns = []
        for column in df.columns:
            unique_values = df[column].dropna().unique()
            if len(unique_values) == 2:
                binary_columns.append(column)
        return binary_columns

    def find_continuous_columns(df):
        continuous_columns = []
        for column in df.columns:
            unique_values = df[column].dropna().unique()
            if len(unique_values) > 2:
                continuous_columns.append(column)
        return continuous_columns

    #Loading packages
    # Loading packages
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    from sklearn.base import BaseEstimator, TransformerMixin
    import os

    # Custom scaler that ignores NaN values
    class NaNIgnoringScaler(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.mean_ = None
            self.scale_ = None

        def fit(self, X, y=None):
            self.mean_ = np.nanmean(X, axis=0)
            self.scale_ = np.nanstd(X, axis=0)
            return self

        def transform(self, X):
            return (X - self.mean_) / self.scale_

        def fit_transform(self, X, y=None):
            return self.fit(X, y).transform(X)

    # Define the main multi-view model
    class MultiViewModel(nn.Module):
        def __init__(self, input_dims, hidden_dims, latent_dim, output_dim):
            super(MultiViewModel, self).__init__()
            self.encoders = nn.ModuleList([nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim)
            ) for input_dim, hidden_dim in zip(input_dims, hidden_dims)])
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim * len(input_dims), 160),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(160, output_dim)
            )

        def forward(self, inputs, masks):
            latent_representations = []
            for i, encoder in enumerate(self.encoders):
                latent_representation = [] 
                for row, mask_row in enumerate(masks[i]):
                    if mask_row.sum() > 0:
                        masked_input = inputs[i][row] * mask_row.unsqueeze(0)
                        latent_representation.append(encoder(masked_input.unsqueeze(0)).squeeze(0))
                    else:
                        latent_representation.append(torch.zeros(encoder[-1].out_features, device=inputs[i].device))
                latent_representations.append(torch.stack(latent_representation))
            latent_concat = torch.cat(latent_representations, dim=1)
            return self.decoder(latent_concat)

    # Wrapper for handling splitting and preprocessing
    class MultiViewWrapper(nn.Module):
        def __init__(self, model, input_dims, scaler, binary_columns):
            super(MultiViewWrapper, self).__init__()
            self.model = model
            self.input_dims = input_dims
            self.scaler = scaler
            self.binary_columns = binary_columns

        def split_input(self, combined_input):
            splits = []
            start = 0
            for dim in self.input_dims:
                splits.append(combined_input[:, start:start + dim])
                start += dim
            return splits

        def create_masks(self, views):
            masks = []
            for view in views:
                mask = (~torch.isnan(view).any(dim=1)).float()
                masks.append(mask)
            return masks

        def forward(self, combined_input, target=None):
            mask = [i for i in range(combined_input.size(1)) if i not in self.binary_columns]
            views = self.split_input(combined_input)
            masks = self.create_masks(views)
            return self.model(views, masks)

    # Load datasets
    view1 = pd.read_csv('~/multimodal/data/view1.csv').drop(columns=["Unnamed: 0", "fev075fvc_z_score_preBD1"])
    view3 = pd.read_csv('~/multimodal/data/view3.csv').drop(columns=["Unnamed: 0", "fev075fvc_z_score_preBD1"])
    lung_func_df = pd.read_csv("/global/project/hpcg1553/DUAN_LAB_DATABASE/CHILD_STUDY_DATA/CHILD_HEALTH/lung_function_updated.csv")

    # Merge datasets
    two_view = view1.merge(view3, on="FID", how="outer")
    dataset = lung_func_df[["FID", "fev075fvc_z_score_preBD1"]].merge(two_view, on="FID")

    # Split into train and test
    test_FID = pd.read_csv("~/multimodal/data/2_view_test.csv")["FID"]
    test_dataset = dataset[dataset["FID"].isin(test_FID)].drop(columns="FID")
    train_dataset = dataset[~dataset["FID"].isin(test_FID)].drop(columns="FID")

    # Prepare data
    X_train = train_dataset.drop(columns=["fev075fvc_z_score_preBD1"]).values
    y_train = train_dataset["fev075fvc_z_score_preBD1"].values
    X_val = test_dataset.drop(columns=["fev075fvc_z_score_preBD1"]).values
    y_val = test_dataset["fev075fvc_z_score_preBD1"].values

    #scaling
    scaler = NaNIgnoringScaler()
    indices_to_exclude = find_binary_columns(pd.DataFrame(X_train))
    mask = np.array([i for i in range(X_train.shape[1]) if i not in indices_to_exclude])
    X_train_transformed = scaler.fit_transform(X_train[:, mask], y_train)
    X_train[:, mask] = X_train_transformed
    #On X and y test
    X_val_transformed= scaler.transform(X_val[:, mask])
    X_val[:, mask] = X_val_transformed

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_train_combined = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_val_combined = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    # Model parameters
    input_dims = [21, 1]
    hidden_dims = [160, 160]
    latent_dim = 4
    output_dim = 1
    binary_columns = find_binary_columns(pd.DataFrame(X_train))

    # Initialize the model and wrapper
    #model = MultiViewModel(input_dims, hidden_dims, latent_dim, output_dim).to(device)
    #wrapper = MultiViewWrapper(model, input_dims, NaNIgnoringScaler(), binary_columns).to(device)

    # Training setup
    criterion = nn.MSELoss()
    #optimizer = torch.optim.SGD(wrapper.parameters(), lr=0.05, weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)


    ###### load the model
    import os
    import pickle

    # Load the model
    # # Instantiate the model architecture
    model_path = os.path.expanduser("~/models/spec_1.pth")
    criterion = nn.MSELoss()

    # Load the state dictionary
    #model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully")



    wrapper = torch.load(model_path, map_location=torch.device('cpu'))  # unpickles the entire wrapper object
    wrapper.eval()
    with torch.no_grad():
        val_predictions = wrapper(X_val_combined)
        val_loss = criterion(val_predictions, y_val).item()
        val_r2 = r2_score(y_val.cpu().numpy(), val_predictions.cpu().numpy())
        print(f" Validation Loss: {val_loss:.4f}, R²: {val_r2:.4f}")
    #load dataset and process it to make sure the model works/gets the same results
    #print(f"val_predictions: {val_predictions}")

    ########### setting the model up for dice_ml to use
    # Remove 'FID' so that DiCE sees only features + outcome
    dataset_for_dice = dataset.drop(columns=["FID"]).astype(np.float32)

    # Identify categorical vs. continuous features by dropping the outcome column
    temp_feats = dataset_for_dice.drop(columns=["fev075fvc_z_score_preBD1"])
    cat_feat = find_binary_columns(temp_feats)
    cont_feat = find_continuous_columns(temp_feats)

    # Convert X_train tensor back to a DataFrame with the correct column names
    X_train_df = pd.DataFrame(X_train, columns=temp_feats.columns)

    X_val_df = pd.DataFrame(X_val, columns=temp_feats.columns)

    # Convert y_train tensor back to a Series
    y_train_series = pd.Series(y_train.flatten(), name="fev075fvc_z_score_preBD1")

    # Combine X_train and y_train into a DataFrame
    dataset_for_dice = pd.concat([X_train_df, y_train_series], axis=1)

    #replace values outside -3, 3
    dataset_for_dice = dataset_for_dice.applymap(lambda x: x if -3 <= x <= 3 else np.nan)

    # Handle missing values using KNNImputer
    imputer = KNNImputer(n_neighbors=5)
    dataset_for_dice = pd.DataFrame(imputer.fit_transform(dataset_for_dice), columns=dataset_for_dice.columns)

    # Ensure categorical features are rounded (since imputer may introduce decimals)
    dataset_for_dice[cat_feat] = dataset_for_dice[cat_feat].round()

    ###### set random seed
    seed = 9
    random.seed(seed)
    np.random.seed(seed)

    dim_map = 1  # Dimensionality of the behavior space
    dim_x = X_train.shape[1]  # Dimensionality of the feature space
    #n_niches = 100  # Number of niches
    max_evals = 1e4 # Maximum number of evaluations

    behavior_features = ["PRS_CS.zscr.Shrine.FEV1FVC"]  # Choose specific features "PNDIET_retinol", 

    # Extract indices of selected features
    behavior_indices = [list(dataset_for_dice.columns).index(f) for f in behavior_features]
    '''
    params = {
        "min": np.array(dataset_for_dice.drop(columns='fev075fvc_z_score_preBD1').min()),
        "max": np.array(dataset_for_dice.drop(columns='fev075fvc_z_score_preBD1').max()),
        "random_init_batch": 100,
    }


    # Run Fixed-Grid MAP-Elites
    elites = mapcf_trend_intercell(
        dim_map, dim_x, evaluate_feature_vector,
        n_bins=100, max_evals=max_evals, params=params, use_crossover=True
    )

    archive = elites.run()

    # Extract elite feature vectors and their corresponding predicted values
    elite_features = []
    elite_predictions = []

    for cell_index, data in archive.items():
        if "best" in data and data["best"] is not None:
            best_vector, best_fitness = data["best"]
            elite_features.append(best_vector)
            elite_predictions.append(best_fitness)

    # Convert to DataFrame
    df_elites = pd.DataFrame(elite_features, columns=X_train_df.columns)  # Fix feature names
    df_elites["fev075fvc_z_score_preBD1"] = elite_predictions  # Add predicted target values

    # Display the top 5 elite feature vectors
    print(df_elites.head())

    # generates a plot for each feature

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Define the descriptor (x-axis)
    descriptor_x = "PRS_CS.zscr.Shrine.FEV1FVC"

    # List of fitness metrics (excluding the descriptor itself)
    fitness_metrics = [
        'mom_allergy_drug', 'mom_arthritis', 'mom_diabetes', 'mom_reflux',
        'PNDIET_calcium', 'PNDIET_retinol', 'PNDIET_vitb12', 'PNDIET_water',
        'PNDIET_hei5', 'PNDIET_hei8', 'season_birth_1', 'season_birth_3',
        'season_milk_2', 'season_milk_4', 'matvit_yn', 'AnyPets3M',
        'PregnantAnyPets', 'PregnantCatDog', 'infant_sex', 'PNDIET_protein',
        'bronch_3m'
    ]

    # Define subplot grid size
    num_cols = 4  # Number of columns in the grid
    num_rows = -(-len(fitness_metrics) // num_cols)  # Round up division to get rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 4 * num_rows))
    axes = axes.flatten()  # Flatten for easier indexing

    #df_elites=elite_df_named

    # Generate scatter plots
    for i, metric in enumerate(fitness_metrics):
        ax = axes[i]
        ax.scatter(df_elites[descriptor_x], df_elites[metric], alpha=0.5, color="blue")
        ax.set_xlabel(descriptor_x)
        ax.set_ylabel(metric)
        ax.set_title(f"{descriptor_x} vs {metric}")

    # Remove unused subplots if there are extra spaces
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    
    plt.show()
    plt.savefig("plot.pdf")
    plt.close()
    '''
    import itertools

    # Define Feature Categories
    feature_categories = {
        "A": ["mom_allergy_drug", "mom_arthritis", "mom_diabetes", "mom_reflux"],  # Maternal health
        "B": ["PRS_CS.zscr.Shrine.FEV1FVC"],  # Genetic score
        "C": ["season_birth_1", "season_birth_3", "season_milk_2", "season_milk_4", "matvit_yn", "AnyPets3M", "PregnantAnyPets", "PregnantCatDog", "infant_sex"],  # Environmental exposure
        "D": ["PNDIET_calcium", "PNDIET_retinol", "PNDIET_vitb12", "PNDIET_water", "PNDIET_hei5", "PNDIET_hei8", "PNDIET_protein"],  # Diet-related features
        "E": ["bronch_3m", "infant_sex"]  # Infant information
    }

    # Get category labels
    category_labels = list(feature_categories.keys())
    num_categories = len(category_labels)

    # Create a dictionary for unique feature sets (upper triangular matrix only)
    cell_feature_sets = {}

    # Store only the upper triangle of the matrix (including diagonal)
    for i in range(num_categories):
        for j in range(i, num_categories):  # Ensures j >= i (upper triangular)
            if (i, j) not in cell_feature_sets:
                cell_feature_sets[(i, j)] = feature_categories[category_labels[i]]

            if i != j:  # If it's off-diagonal, add the second category
                cell_feature_sets[(i, j)] += feature_categories[category_labels[j]]

    # Print example cells
    for key, value in cell_feature_sets.items():
        print(f"Cell {key} ({category_labels[key[0]]} x {category_labels[key[1]]}): {value}")

    # Define min/max bounds for each feature
    params = {
        "min": np.nanmin(X_train, axis=0),
        "max": np.nanmax(X_train, axis=0),
        "random_init_batch": 1000
    }

    X_reference = X_train_df.iloc[1]


    elite = mapcf_instance(
        dim_map=1,  # Single-dimensional feature space binning
        dim_x=X_train_df.shape[1],  # Number of total features
        eval_func=evaluate_feature_vector,
        max_evals=10000,  # Total evaluations
        params=params,
        cell_feature_sets=cell_feature_sets
    )

    archive = elite.run()

    import numpy as np
    import matplotlib.pyplot as plt

    # Create an empty matrix with NaN values
    fitness_matrix = np.full((num_categories, num_categories), np.nan)

    # Fill matrix with best fitness values from archive (both upper and lower triangles)
    for (i, j), data in archive.items():
        if data["best"] is not None:
            _, best_fitness = data["best"]
            fitness_matrix[i, j] = best_fitness  # Assign value (upper)
            fitness_matrix[j, i] = best_fitness  # Assign value (lower)

    # Plot the full symmetric matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(fitness_matrix, cmap="viridis", interpolation="nearest")

    # Flip y-axis so (0,0) is bottom-left
    plt.gca().invert_yaxis()

    # Add colorbar
    plt.colorbar(label="Best Fitness Value")

    # Set axis labels
    plt.xticks(range(num_categories), category_labels)
    plt.yticks(range(num_categories), category_labels)
    plt.xlabel("Feature Category 2")
    plt.ylabel("Feature Category 1")
    plt.title("Best Elites Fitness Matrix (Full Symmetric)")

    # Add values inside cells
    for i in range(num_categories):
        for j in range(num_categories):  # Show entire matrix
            if not np.isnan(fitness_matrix[i, j]):
                plt.text(j, i, f"{fitness_matrix[i, j]:.2f}", ha="center", va="center", color="white")

    plt.show()
    plt.savefig("plotinst.pdf")
    plt.close()



