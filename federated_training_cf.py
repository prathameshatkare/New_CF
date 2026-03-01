# File: federated_training_cf.py
# Purpose: Federated Learning with clients on a COPD dataset using Flower 2.x API

import numpy as np
import pandas as pd
import flwr as fl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -------------------------
# Step 1: Load dataset
# -------------------------
df = pd.read_csv("copd.csv")  # Replace with your CSV path
print("Dataset columns:", df.columns)

# Features and target
X = df.drop(columns=["target"]).values.astype(np.float32)
y = df["target"].values.astype(np.float32)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -------------------------
# Step 2: Split dataset for clients
# -------------------------
num_clients = 5
client_data = np.array_split(np.c_[X, y], num_clients)  # Split X+y together

# Save each client data if needed
for i, data in enumerate(client_data, 1):
    np.save(f"client_{i}_data.npy", data)
    print(f"Saved client {i} dataset: {data.shape}")

print("\nAll client datasets are ready for Federated Learning!")

# -------------------------
# Step 3: Define Keras model
# -------------------------
def create_model():
    model = Sequential([
        Dense(32, activation="relu", input_shape=(X.shape[1],)),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")  # Assuming binary classification
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# -------------------------
# Step 4: Define Flower client
# -------------------------
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, data):
        self.model = model
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data[:, :-1], data[:, -1], test_size=0.2, random_state=42
        )

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.X_train, self.y_train, epochs=1, batch_size=16, verbose=0)
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        return float(loss), len(self.X_test), {"accuracy": float(acc)}

# -------------------------
# Step 5: Start server
# -------------------------
strategy = fl.server.strategy.FedAvg()

num_rounds = 5
server_config = fl.server.ServerConfig(num_rounds=num_rounds)

# Start server (blocking)
if __name__ == "__main__":
    # Launch Flower server
    print("\nStarting Flower server...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=server_config,
        strategy=strategy
    )

# -------------------------
# Step 6: Start clients
# -------------------------
# Run each client in a separate script/terminal:
# python federated_training_cf_client.py --client_id 1

