#Loading packages
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import sys
sys.path.append('/global/home/hpc5434/multimodal/Spec/TabSS')

from TabSS import TabShaSpec, TabShaSpecAttention, TabShaSpecAttentionGlobal
from TabSS import TabShaSpecWrapper  # <--- wrapper

print("packages A loaded")
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

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class NaNIgnoringScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        """
        Compute mean and std while ignoring NaN values.
        """
        self.mean_ = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0)
        return self

    def transform(self, X):
        """
        Scale data using the computed mean and std, ignoring NaNs.
        """
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X, y=None):
        """
        Compute mean and std, and scale the data.
        """
        return self.fit(X, y).transform(X)
    

### SYNTHETIC DATSET ###
np.random.seed(6)

n_samples = 500

# View 1: Polygenic Risk Score (PRS)
PRS = np.random.normal(0, 1, n_samples)

# View 2: Environmental exposures (binary and continuous)
E1 = np.random.binomial(n=1, p=0.2, size=n_samples)  # e.g., mom smoking
E2 = np.random.binomial(n=1, p=0.5, size=n_samples)  # e.g., any pets
E3 = np.random.normal(0, 1, n_samples)               # e.g., air pollution index

# View 3: Prenatal diet (correlated nutrients)
D1 = np.random.normal(0, 1, n_samples)               # e.g., dietary iron
D2 = 0.6 * D1 + np.random.normal(0, 0.5, n_samples)  # e.g., calcium
D3 = 0.4 * D1 + np.random.normal(0, 0.5, n_samples)  # e.g., retinol

# View 4: Milk metabolomics (with 50% missing)
M1 = np.random.normal(0, 1, n_samples)
M2 = 0.5 * M1 + np.random.normal(0, 0.5, n_samples)

mask = np.random.rand(n_samples) < 0.5
#M1[mask] = np.nan
#M2[mask] = np.nan

# Target: FEV1/FVC z-score
y = (
    0.1 * PRS +
    0.05 * E1 -              # penalty for smoking exposure
    0.02 * E2 + 
    0.03 * E3 -
    0.03 * D1 +
    0.025 * D2 +
    0.04 * D3 * (1 - E1) +
    0.05 * PRS * E3 +        # interaction term
    0.04 * PRS * E1 +
    0.08 * M1 +                     # stronger direct effect
    0.06 * M2 +                     
    0.05 * M1 * PRS +              # interaction with genetics
    0.04 * M2 * E3 +               # interaction with exposure
    -0.03 * (M1 ** 2) +            # nonlinear: excess is harmful
    -0.02 * (M2 ** 2) + 
    np.random.normal(0, 1, n_samples)  # noise
)


# Clip target to simulate z-score range
y = np.clip(y, -3, 3)

# Assemble into a DataFrame
synthetic_df = pd.DataFrame({
    'PRS': PRS,
    'E1': E1,
    'E2': E2,
    'E3': E3,
    'D1': D1,
    'D2': D2,
    'D3': D3,
    'M1': M1,
    'M2': M2,
    'y': y
})

# Preview
print(synthetic_df.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error

# Separate features and target
X = synthetic_df.dropna().drop(columns='y')
y = synthetic_df.dropna()['y']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression pipeline
linreg_model = LinearRegression()
linreg_model.fit(X_train, y_train)
y_pred_linreg = linreg_model.predict(X_test)

# Random Forest pipeline
#rf_model = make_pipeline(SimpleImputer(strategy='mean'), RandomForestRegressor(n_estimators=100, random_state=42))
rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluation
print("Linear Regression:")
print("R² Score:", r2_score(y_test, y_pred_linreg))
print("MSE:", mean_squared_error(y_test, y_pred_linreg))

print("\nRandom Forest:")
print("R² Score:", r2_score(y_test, y_pred_rf))
print("MSE:", mean_squared_error(y_test, y_pred_rf))

X_train, X_val, y_train, y_val = train_test_split(X.values, y.values, test_size=0.2, random_state=42)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Multimodal model (adapted for single-modal use)


# Model parameters
input_dims = [1, 3, 3, 2]

hidden_dim_pre = [4, 8, 8, 4]
shared_dim = 4
hidden_dim_shared = 4
r_dim = 4 



hidden_dims = [4, 8, 8, 4]
latent_dim = [4, 4, 4, 4]
       
output_dim = 1    

# Define input dimensions for each view
#input_dims = [22]  # Example: 3 views with 50, 30, and 20 features each



scaler = NaNIgnoringScaler()
binary_columns = find_binary_columns(pd.DataFrame(X_train))


# Initialize the model
print("Using TabShaSpec1")
model = TabShaSpecAttentionGlobal(input_dims, shared_dim, hidden_dim_pre, hidden_dim_shared, hidden_dims, r_dim, latent_dim, output_dim)
wrapper = TabShaSpecWrapper(model, input_dims, binary_columns)
wrapper = wrapper.to(device)  # Use 'cuda' if a GPU is available
criterion = nn.MSELoss()
num_epochs = 500

#2) loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-3)
#optimizer = torch.optim.Adam(wrapper.parameters(), lr=learning_rate, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


#On X and y tensor
# List of indices to exclude
# Pre-scale continuous columns only
scaler = NaNIgnoringScaler()
mask = np.array([i for i in range(X_train.shape[1]) if i not in binary_columns])
X_train[:, mask] = scaler.fit_transform(X_train[:, mask], y_train)
X_val[:, mask] = scaler.transform(X_val[:, mask])


# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Reshape y if necessary

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)  # Reshape y if necessary


X_train_combined = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_val_combined = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)


l1_lambda = 1e-3  # L1 regularization coefficient


# Training loop
for epoch in range(num_epochs):
    wrapper.train()
    optimizer.zero_grad()
    
    # Forward pass
    predictions = wrapper(X_train_combined)


    # Compute loss
    loss = criterion(predictions, y_train)
    
    # L1 regularization (manual)
    l1_penalty = sum(torch.sum(torch.abs(p)) for p in wrapper.parameters())
    total_loss = loss + l1_lambda * l1_penalty

    # Backward pass and optimizer step
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Print progress
    if (epoch + 1) % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Learning Rate: {current_lr:.6f}')


        # Validation phase
        model.eval()
        with torch.no_grad():
            val_predictions = wrapper(X_val_combined)
            val_loss = criterion(val_predictions, y_val).item()
            val_r2 = r2_score(y_val.cpu().numpy(), val_predictions.cpu().numpy())
            print(f'Validation Loss: {val_loss:.4f}, R²: {val_r2:.4f}')


import os
import pickle
with open("wrapper.pkl", "wb") as f:
    pickle.dump(wrapper, f)

model_dir = os.path.expanduser("~/models")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "synthetic.pth")
torch.save(wrapper, model_path)
print(f"Model saved successfully at {model_path}")


###### load the model
import pickle

# Load the model
# # Instantiate the model architecture
model_path = os.path.expanduser("~/models/synthetic.pth")
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

