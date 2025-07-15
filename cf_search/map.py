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


class mapcf_instance:
    """MAP-Elites where crossover and mutation happen within the same cell."""

    def __init__(self, dim_map, dim_x, max_evals, params, cell_feature_sets, X_train_df, feature_categories, X_reference, wrapper, method, mutation_rate=None, seed=6):
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
        self.method = method
        self.mutation_rate = mutation_rate

        self.feature_mins = np.array(params["min"]).astype(float)
        self.feature_maxs = np.array(params["max"]).astype(float)
        self.X_reference = np.clip(self.X_reference, self.feature_mins, self.feature_maxs)

        def is_binary_like(series, tol=1e-4):
            unique_vals = np.unique(np.round(series.dropna(), decimals=4))
            return len(unique_vals) == 2

        self.binary_features = [
            col for col in self.X_train_df.columns
            if is_binary_like(self.X_train_df[col])
        ]

        self.feature_indices = {feature: list(self.X_train_df.columns).index(feature) for feature in self.X_train_df.columns}

        self.is_sklearn_model = isinstance(wrapper, BaseEstimator)
        self.is_torch_model = hasattr(wrapper, "eval") and callable(wrapper.eval)
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
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
                if feature in self.binary_features:
                    # Choose from actual binary-like values found in training data
                    values = np.unique(np.round(self.X_train_df[feature].dropna(), decimals=4))
                    feature_vector[feature_index] = np.random.choice(values)
                else:
                    # 20% chance of min, 20% chance of max, 60% uniform
                    rand = np.random.rand()
                    
                    if rand < 0.05:
                        feature_vector[feature_index] = self.feature_mins[feature_index]
                    elif rand < 0.1:
                        feature_vector[feature_index] = self.feature_maxs[feature_index]
                    else:
                        feature_vector[feature_index] = np.random.uniform(
                            self.feature_mins[feature_index], self.feature_maxs[feature_index]
                        )

            '''
            for feature in mutable_features:
                feature_index = self.feature_indices[feature]
                values = self.X_train_df[feature].dropna().to_numpy()
                feature_vector[feature_index] = np.random.choice(values)
            '''
            batch_vectors.append(feature_vector)
            batch_cells.append(cell_index)

        batch_fitnesses = self.evaluate_batch(batch_vectors)

        for vec, fit, cell_index in zip(batch_vectors, batch_fitnesses, batch_cells):
            if cell_index not in self.archive:
                self.archive[cell_index] = {"best": None, "population": []}
            self.archive[cell_index]["population"].append((vec, fit))
            if self.archive[cell_index]["best"] is None or fit > self.archive[cell_index]["best"][1]:
                self.archive[cell_index]["best"] = (vec, fit)
    
    def evaluate_instance(self, feature_vector):
        if self.is_torch_model:
            self.wrapper.eval()
            feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                predicted_value = self.wrapper(feature_tensor).cpu().numpy().flatten()[0]
            return predicted_value
        else:
            feature_vector = np.array(feature_vector).reshape(1, -1)
            return self.wrapper.predict(feature_vector)[0]

    def evaluate_batch(self, feature_matrix):
        feature_matrix = np.array(feature_matrix, dtype=np.float32)

        if self.is_torch_model:
            self.wrapper.eval()
            feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32).to(device)
            with torch.no_grad():
                predictions = self.wrapper(feature_tensor).cpu().numpy().flatten()
            return predictions
        else:
            return self.wrapper.predict(feature_matrix)

    def ensure_binary_validity(self, child):
        for feature in self.binary_features:
            index = self.feature_indices[feature]
            original_val = child[index]
            # Find original two valid values
            valid_values = np.unique(np.round(self.X_train_df[feature].dropna(), decimals=4))
            if len(valid_values) == 2:
                # Snap to closest of the two
                child[index] = valid_values[np.argmin(np.abs(valid_values - original_val))]
        return child


    def single_point_crossover(self, p1, p2, mutable_features):
        """
        Single-point crossover applied only to mutable features.
        Non-mutable features are taken from the reference vector.
        """
        child = np.array(self.X_reference).copy()
        if len(mutable_features) < 2:
            # fallback: copy one parent
            selected = p1 if np.random.rand() < 0.5 else p2
            for feature in mutable_features:
                idx = self.feature_indices[feature]
                child[idx] = selected[idx]
            return child

        point = np.random.randint(1, len(mutable_features))
        for i, feature in enumerate(mutable_features):
            idx = self.feature_indices[feature]
            source = p1 if i < point else p2
            child[idx] = source[idx]
        return child


    def uniform_crossover(self, p1, p2, mutable_features):
        """
        Uniform crossover applied only to mutable features.
        Non-mutable features are taken from the reference vector.
        """
        child = np.array(self.X_reference).copy()
        for feature in mutable_features:
            idx = self.feature_indices[feature]
            if np.random.rand() < 0.5:
                child[idx] = p1[idx]
            else:
                child[idx] = p2[idx]
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
        
        r1 = np.random.random(size=len(mutable_features))
        r2 = np.random.random(size=len(mutable_features))

        child = np.array(self.X_reference).copy()  # Start from reference, then mutate only mutable features

        for idx, feature in enumerate(mutable_features):
            feature_index = self.feature_indices[feature]

            # If binary, skip SBX (you might handle binary features separately)
            if feature in self.binary_features:
                continue

            # Extract bounds
            x1 = min(x[feature_index], y[feature_index])
            x2 = max(x[feature_index], y[feature_index])
            if abs(x2 - x1) < 1e-15:
                continue  # Parents are too similar

            xl_i = xl[feature_index]
            xu_i = xu[feature_index]

            # SBX calculations
            beta = 1.0 + (2.0 * (x1 - xl_i)) / (x2 - x1)
            alpha = 2.0 - beta ** -(eta + 1)
            rand = r1[idx]
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha / 2.0) ** (1.0 / (eta + 1))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

                # Generate child candidates
                c1 = 0.5 * ((x1 + x2) - beta_q * (x2 - x1))
                c2 = 0.5 * ((x1 + x2) + beta_q * (x2 - x1))

                # Clip to bounds
                c1 = np.clip(c1, xl_i, xu_i)
                c2 = np.clip(c2, xl_i, xu_i)

                # Randomly choose one child
                child[feature_index] = c2 if r2[idx] <= 0.5 else c1

        # Done — `child` has mutable features modified, non-mutable ones set to reference
        return child

                
    def mutate(self, parent, mutable_features):
        mutation_rate=self.mutation_rate
        child = parent.copy()
        for feature in mutable_features:
            if np.random.rand() < mutation_rate:
                index = self.feature_indices[feature]
                if feature in self.binary_features:
                    child[index] = 1 - int(round(child[index]))
                else:
                    child[index] = np.random.uniform(self.feature_mins[index], self.feature_maxs[index])
        return child
    


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
        batch_size = 1024
        child_batch = []
        cell_batch = []

        total_steps = int(self.max_evals - self.params["random_init_batch"])

        for a in range(total_steps):
            i, j = sorted(np.random.choice(range(len(self.feature_categories)), size=2, replace=True))
            cell_index = (i, j)

            if cell_index not in self.archive:
                continue

            mutable_features = self.cell_feature_sets[cell_index]
            parent_1, parent_2 = self.select_parents(cell_index)

            # --- crossover method ---
            if self.method == "sbx":
                child = self.sbx_crossover(parent_1, parent_2, mutable_features)
            elif self.method == "uniform":
                child = self.uniform_crossover(parent_1, parent_2, mutable_features)
            elif self.method == "single_point":
                child = self.single_point_crossover(parent_1, parent_2, mutable_features)
            else:
                raise ValueError(f"Unknown crossover method: {self.method}")


            # --- Apply mutation ---
            if self.mutation_rate is not None and self.mutation_rate > 0:
                child = self.mutate(child, mutable_features)


            # --- Enforce binary validity ---
            child = self.ensure_binary_validity(child)

            # --- Store in batch ---
            child_batch.append(child)
            cell_batch.append(cell_index)


            # When batch is full or last step
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



from joblib import Parallel, delayed
from tqdm import trange 
class mapcf_instance_parallelized:
    """MAP-Elites where crossover and mutation happen within the same cell."""

    def __init__(self, dim_map, dim_x, max_evals, params, cell_feature_sets, X_train_df, feature_categories, X_reference, wrapper, method, mutation_rate=None, seed=6):
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
        self.method = method
        self.mutation_rate = mutation_rate

        self.feature_mins = np.array(params["min"]).astype(float)
        self.feature_maxs = np.array(params["max"]).astype(float)
        self.X_reference = np.clip(self.X_reference, self.feature_mins, self.feature_maxs)

        def is_binary_like(series, tol=1e-4):
            unique_vals = np.unique(np.round(series.dropna(), decimals=4))
            return len(unique_vals) == 2

        self.binary_features = [
            col for col in self.X_train_df.columns
            if is_binary_like(self.X_train_df[col])
        ]

        self.feature_indices = {feature: list(self.X_train_df.columns).index(feature) for feature in self.X_train_df.columns}

        self.is_sklearn_model = isinstance(wrapper, BaseEstimator)
        self.is_torch_model = hasattr(wrapper, "eval") and callable(wrapper.eval)
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

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
                if feature in self.binary_features:
                    # Choose from actual binary-like values found in training data
                    values = np.unique(np.round(self.X_train_df[feature].dropna(), decimals=4))
                    feature_vector[feature_index] = np.random.choice(values)
                else:
                    # 20% chance of min, 20% chance of max, 60% uniform
                    rand = np.random.rand()
                    
                    if rand < 0.05:
                        feature_vector[feature_index] = self.feature_mins[feature_index]
                    elif rand < 0.1:
                        feature_vector[feature_index] = self.feature_maxs[feature_index]
                    else:
                        feature_vector[feature_index] = np.random.uniform(
                            self.feature_mins[feature_index], self.feature_maxs[feature_index]
                        )

            '''
            for feature in mutable_features:
                feature_index = self.feature_indices[feature]
                values = self.X_train_df[feature].dropna().to_numpy()
                feature_vector[feature_index] = np.random.choice(values)
            '''
            batch_vectors.append(feature_vector)
            batch_cells.append(cell_index)

        batch_fitnesses = self.evaluate_batch(batch_vectors)

        for vec, fit, cell_index in zip(batch_vectors, batch_fitnesses, batch_cells):
            if cell_index not in self.archive:
                self.archive[cell_index] = {"best": None, "population": []}
            self.archive[cell_index]["population"].append((vec, fit))
            if self.archive[cell_index]["best"] is None or fit > self.archive[cell_index]["best"][1]:
                self.archive[cell_index]["best"] = (vec, fit)
    
    def evaluate_instance(self, feature_vector):
        if self.is_torch_model:
            self.wrapper.eval()
            feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                predicted_value = self.wrapper(feature_tensor).cpu().numpy().flatten()[0]
            return predicted_value
        else:
            feature_vector = np.array(feature_vector).reshape(1, -1)
            return self.wrapper.predict(feature_vector)[0]

    def evaluate_batch(self, feature_matrix):
        feature_matrix = np.array(feature_matrix, dtype=np.float32)

        if self.is_torch_model:
            self.wrapper.eval()
            feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32).to(device)
            with torch.no_grad():
                predictions = self.wrapper(feature_tensor).cpu().numpy().flatten()
            return predictions
        else:
            return self.wrapper.predict(feature_matrix)

    def ensure_binary_validity(self, child):
        for feature in self.binary_features:
            index = self.feature_indices[feature]
            original_val = child[index]
            # Find original two valid values
            valid_values = np.unique(np.round(self.X_train_df[feature].dropna(), decimals=4))
            if len(valid_values) == 2:
                # Snap to closest of the two
                child[index] = valid_values[np.argmin(np.abs(valid_values - original_val))]
        return child


    def single_point_crossover(self, p1, p2, mutable_features):
        """
        Single-point crossover applied only to mutable features.
        Non-mutable features are taken from the reference vector.
        """
        child = np.array(self.X_reference).copy()
        if len(mutable_features) < 2:
            # fallback: copy one parent
            selected = p1 if np.random.rand() < 0.5 else p2
            for feature in mutable_features:
                idx = self.feature_indices[feature]
                child[idx] = selected[idx]
            return child

        point = np.random.randint(1, len(mutable_features))
        for i, feature in enumerate(mutable_features):
            idx = self.feature_indices[feature]
            source = p1 if i < point else p2
            child[idx] = source[idx]
        return child


    def uniform_crossover(self, p1, p2, mutable_features):
        """
        Uniform crossover applied only to mutable features.
        Non-mutable features are taken from the reference vector.
        """
        child = np.array(self.X_reference).copy()
        for feature in mutable_features:
            idx = self.feature_indices[feature]
            if np.random.rand() < 0.5:
                child[idx] = p1[idx]
            else:
                child[idx] = p2[idx]
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
        
        r1 = np.random.random(size=len(mutable_features))
        r2 = np.random.random(size=len(mutable_features))

        child = np.array(self.X_reference).copy()  # Start from reference, then mutate only mutable features

        for idx, feature in enumerate(mutable_features):
            feature_index = self.feature_indices[feature]

            # If binary, skip SBX (you might handle binary features separately)
            if feature in self.binary_features:
                continue

            # Extract bounds
            x1 = min(x[feature_index], y[feature_index])
            x2 = max(x[feature_index], y[feature_index])
            if abs(x2 - x1) < 1e-15:
                continue  # Parents are too similar

            xl_i = xl[feature_index]
            xu_i = xu[feature_index]

            # SBX calculations
            beta = 1.0 + (2.0 * (x1 - xl_i)) / (x2 - x1)
            alpha = 2.0 - beta ** -(eta + 1)
            rand = r1[idx]
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha / 2.0) ** (1.0 / (eta + 1))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

                # Generate child candidates
                c1 = 0.5 * ((x1 + x2) - beta_q * (x2 - x1))
                c2 = 0.5 * ((x1 + x2) + beta_q * (x2 - x1))

                # Clip to bounds
                c1 = np.clip(c1, xl_i, xu_i)
                c2 = np.clip(c2, xl_i, xu_i)

                # Randomly choose one child
                child[feature_index] = c2 if r2[idx] <= 0.5 else c1

        # Done — `child` has mutable features modified, non-mutable ones set to reference
        return child

                
    def mutate(self, parent, mutable_features):
        mutation_rate=self.mutation_rate
        child = parent.copy()
        for feature in mutable_features:
            if np.random.rand() < mutation_rate:
                index = self.feature_indices[feature]
                if feature in self.binary_features:
                    child[index] = 1 - int(round(child[index]))
                else:
                    child[index] = np.random.uniform(self.feature_mins[index], self.feature_maxs[index])
        return child
    


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

    def generate_child(self):
        while True:
            i, j = sorted(np.random.choice(range(len(self.feature_categories)), size=2, replace=True))
            cell_index = (i, j)
            if cell_index not in self.archive:
                continue

            mutable_features = self.cell_feature_sets[cell_index]
            parent_1, parent_2 = self.select_parents(cell_index)

            # Crossover
            if self.method == "sbx":
                child = self.sbx_crossover(parent_1, parent_2, mutable_features)
            elif self.method == "uniform":
                child = self.uniform_crossover(parent_1, parent_2, mutable_features)
            elif self.method == "single_point":
                child = self.single_point_crossover(parent_1, parent_2, mutable_features)
            else:
                raise ValueError(f"Unknown crossover method: {self.method}")

            # Mutation
            if self.mutation_rate is not None and self.mutation_rate > 0:
                child = self.mutate(child, mutable_features)

            # Binary fix
            child = self.ensure_binary_validity(child)

            return child, cell_index

    def run(self):
        self.initialize_archive()
        batch_size = 1024
        child_batch = []
        cell_batch = []

        total_steps = int(self.max_evals - self.params["random_init_batch"])

        # --- Store in batch ---
        NUM_CORES = 1  # adjust as needed
        for _ in trange(0, total_steps, batch_size):
            results = Parallel(n_jobs=NUM_CORES)(
                delayed(self.generate_child)() for _ in range(batch_size)
            )

            child_batch, cell_batch = zip(*results)
            fitness_batch = self.evaluate_batch(child_batch)

            for child_vec, fitness, cell in zip(child_batch, fitness_batch, cell_batch):
                if cell not in self.archive:
                    self.archive[cell] = {"best": (child_vec, fitness), "population": [(child_vec, fitness)]}
                else:
                    self.archive[cell]["population"].append((child_vec, fitness))
                    if fitness > self.archive[cell]["best"][1]:
                        self.archive[cell]["best"] = (child_vec, fitness)

        '''
        # When batch is full or last step
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
        '''
        return self.archive



if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import pandas as pd
    import numpy as np
    import pickle
    import json
    import os
    import torch
    from cf_search.map import mapcf_instance
    import warnings
    import torch.nn as nn
    import argparse
    warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)


    # ─────────────────────────────────────────────────────────────
    # 1. Setup: file paths and metadata
    # ─────────────────────────────────────────────────────────────

    # Define SimpleMLP (must match training definition)
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.Tanh(),
                nn.Linear(32, 16),
                #nn.Dropout(0.1),
                nn.Tanh(),
                nn.Linear(16, 16),
                nn.Tanh(),
                nn.Linear(16, 1)
            )
        def forward(self, x):
            return self.model(x)

    class PerfectWrapper:
        def __init__(self):
            # Define feature ordering if needed
            self.feature_names = ['G1', 'G2', 'G3', 'E1', 'E2', 'N1', 'N2', 'M1', 'M2']

        def predict(self, X):
            # If input is a DataFrame, convert to NumPy in correct order
            if hasattr(X, 'loc') or hasattr(X, 'iloc'):
                X = X[self.feature_names].values

            G1 = X[:, 0]
            G2 = X[:, 1]
            G3 = X[:, 2]
            E1 = X[:, 3]
            E2 = X[:, 4]
            N1 = X[:, 5]
            N2 = X[:, 6]
            M1 = X[:, 7]
            M2 = X[:, 8]

            y = (
                0.5 * G3 +
                0.7 * E2 +
                0.2 * N1 +
                0.3 * M1 +
                + 1.8 * G1 * E1 #interaction between 2 features
                + 1.8 * G2 * N2 * M2 # higher order interaction
            )
            return y

    '''
    model_info = {
        "perfect_model": f"~/MAP-CF/synthetic/perfect_model.pkl",
        "mlp_model": f"~/MAP-CF/synthetic/mlp_model.pth",
        "linear_model": f"~/MAP-CF/synthetic/linear_model.pkl",

        
        #"hgb_model": f"~/MAP-CF/synthetic/hgboost_model.pkl"
        
        
    }
    '''
    model_name = "perfect_model"
    model_path = f"~/MAP-CF/synthetic/perfect_model.pkl"

    train_df_file = f"~/MAP-CF/synthetic/synthetic_train.csv"
    reference_file = f"~/MAP-CF/synthetic/synthetic_test.csv"
    feature_json_file = f"~/MAP-CF/synthetic/synthetic_feature_categories_action.json"

    # Expand paths
    train_df_file = os.path.expanduser(train_df_file)
    reference_file = os.path.expanduser(reference_file)
    feature_json_file = os.path.expanduser(feature_json_file)


    # ─────────────────────────────────────────────────────────────
    # 2. Load training data, features, and reference data
    # ─────────────────────────────────────────────────────────────

    X_train_df = pd.read_csv(train_df_file)
    if "Unnamed: 0" in X_train_df.columns:
        X_train_df = X_train_df.drop(columns="Unnamed: 0")

    with open(feature_json_file, "r") as f:
        feature_categories = json.load(f)

    ref_df = pd.read_csv(reference_file)
    if "Unnamed: 0" in ref_df.columns:
        ref_df = ref_df.drop(columns="Unnamed: 0")

    if not all(col in ref_df.columns for col in X_train_df.columns):
        missing = [col for col in X_train_df.columns if col not in ref_df.columns]
        raise ValueError(f"Missing columns in reference data: {missing}")

    # ─────────────────────────────────────────────────────────────
    # 3. Shared MAP-Elites configuration
    # ─────────────────────────────────────────────────────────────
    '''
    params = {
        "min": np.array(X_train_df.min()),
        "max": np.array(X_train_df.max()),
        "random_init_batch": args.init_pop
    }
    '''
    params = {
        "min": np.full(X_train_df.shape[1], -3.0),
        "max": np.full(X_train_df.shape[1], 3.0),
        "random_init_batch": 500
    }


    #print(params)
    #print("min:", np.array(ref_df.min()), "max:", np.array(ref_df.max()))

    category_labels = list(feature_categories.keys())
    num_categories = len(category_labels)

    cell_feature_sets = {}
    for i in range(num_categories):
        for j in range(num_categories):
            features = feature_categories[category_labels[i]].copy()
            if i != j:
                features += feature_categories[category_labels[j]]
            cell_feature_sets[(i, j)] = features

    # ─────────────────────────────────────────────────────────────
    # 4. Loop through each model
    # ─────────────────────────────────────────────────────────────


    model_path = os.path.expanduser(model_path)
    print(f"\n=== Running MAP-Elites for: {model_name} ===")

    # Load model
    if model_path.endswith(".pth"):
        if model_name == "mlp_model":
            input_dim = X_train_df.shape[1]
            wrapper = SimpleMLP(input_dim)
            wrapper.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
            wrapper.eval()
    else:
        with open(model_path, "rb") as f:
            wrapper = pickle.load(f)  # sklearn model

    # Run MAP-Elites for each individual
    archive_dict = {}
    total = len(ref_df)

    for idx, row in ref_df[X_train_df.columns].iterrows():
        print(f"Processing individual {idx} for '{model_name}'...")
        X_reference = row.to_numpy().astype(float)

        elite = mapcf_instance(
            dim_map=num_categories,
            dim_x=X_train_df.shape[1],
            max_evals=1000,
            params=params,
            cell_feature_sets=cell_feature_sets,
            X_train_df=X_train_df,
            feature_categories=feature_categories,
            X_reference=X_reference,
            wrapper=wrapper,
            method="uniform",
            mutation_rate=0
        )
        archive = elite.run()

        data_records = []
        for cell_index, data in archive.items():
            if data["best"]:
                best_vector, best_fitness = data["best"]
                record = dict(zip(X_train_df.columns, best_vector))
                record["fitness"] = best_fitness
                record["cell"] = str(cell_index)
                data_records.append(record)

        archive_dict[idx] = pd.DataFrame(data_records)

    # Save archive
    archive_filename = f"~/MAP-CF/cf_search/counterfactuals.pkl"
    os.makedirs(os.path.dirname(archive_filename), exist_ok=True)
    with open(archive_filename, "wb") as f:
        pickle.dump(archive_dict, f)
    print(f"Saved: {archive_filename}")
