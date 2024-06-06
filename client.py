import argparse
from datetime import datetime
import os
from pathlib import Path

import tensorflow as tf
from datasets import Dataset

import flwr as fl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

import numpy as np
from keras.utils import pad_sequences
from tensorflow.keras.initializers import GlorotUniform

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, log_dir):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.log_dir = log_dir

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self, config):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Create a TensoBoard callback with a unique log directory for each client
        log_dir_client = f"{self.log_dir}/client_{os.getpid()}"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir_client, histogram_freq=1)

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
            callbacks=[tensorboard_callback],  # Add the TensorBoard callback
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition",
        type=int,
        default=0,
        choices=range(0, 10),
        required=True,
        help="Specifies the artificial data partition of dataset to be used. "
        "Picks partition 0 by default",
    )
    parser.add_argument(
        "--toy",
        type=bool,
        default=False,
        required=False,
        help="Set to true to quicky run the client using only 10 datasamples. "
        "Useful for testing purposes. Default: False",
    )
    args = parser.parse_args()

    # Load and compile Keras model
    #model = tf.keras.applications.EfficientNetB0(
    #    input_shape=(32, 32, 3), weights=None, classes=10
    #)
    '''
    # Model with only Bidirectional GRU layer
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=512, return_sequences=False), input_shape=(1, 43)),
        tf.keras.layers.Dense(units=2, activation='sigmoid')
    ])

    # Model with only Bidirectional LSTM layer
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=512, return_sequences=False), input_shape=(1, 43)),
        tf.keras.layers.Dense(units=2, activation='sigmoid')  
    ])

    # Model with only LSTM layer
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=512, activation='relu', input_shape=(1, 43)),
        tf.keras.layers.Dense(units=2, activation='sigmoid')  # Output layer with sigmoid activation for regression
    ])

    # Model without attention mechanism 
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=512, return_sequences=True), input_shape=(1, 43)),
        tf.keras.layers.GRU(units=128, activation='tanh'),
        tf.keras.layers.Dense(units=2, activation='sigmoid')  
    ]) 

    # Model with attention mechanism
    # Custom Attention Layer
    class AttentionLayer(tf.keras.layers.Layer):
        def __init__(self):
            super(AttentionLayer, self).__init__()

        def build(self, input_shape):
            self.W_a = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer=GlorotUniform(), trainable=True)
            self.U_a = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer=GlorotUniform(), trainable=True)
            self.v_a = self.add_weight(shape=(input_shape[-1], 1), initializer=GlorotUniform(), trainable=True)

        def call(self, hidden_states):
            # Score computation
            score_first_part = tf.tensordot(hidden_states, self.W_a, axes=1)
            h_t = tf.tensordot(hidden_states, self.U_a, axes=1)
            score = tf.nn.tanh(score_first_part + h_t)
            
            # Attention weights
            attention_weights = tf.nn.softmax(tf.tensordot(score, self.v_a, axes=1), axis=1)
            
            # Context vector computation
            context_vector = tf.reduce_sum(attention_weights * hidden_states, axis=1)
            return context_vector

    # Define the model using TensorFlow layers
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=512, return_sequences=True), input_shape=(1, 43)),
        AttentionLayer(),  # Custom attention layer
        tf.keras.layers.Reshape((1, 1024)),  # Reshape to add the timestep dimension
        tf.keras.layers.GRU(units=128, activation='tanh', return_sequences=False),
        tf.keras.layers.Dropout(0.2),  # Adding dropout layer
        tf.keras.layers.Dense(units=2, activation='sigmoid')  
    ])
    '''

    # Model with only GRU layer
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(units=512, activation='tanh', input_shape=(1, 43)),
        tf.keras.layers.Dense(units=2, activation='sigmoid')  
    ])


    model.compile("adam", "mean_squared_error", metrics=["accuracy"])

    # Load a subset of dataset to simulate the local data partition
    (x_train, y_train), (x_test, y_test) = load_partition(args.partition)

    if args.toy:
        x_train, y_train = x_train[:10], y_train[:10]
        x_test, y_test = x_test[:10], y_test[:10]

    # Start Flower client
    log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    client = CifarClient(model, x_train, y_train, x_test, y_test, log_dir)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
        root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )


def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    assert idx in range(10)
    # Read the entire dataset
    df = pd.read_csv("preprocessed_data.csv")

    df['cpu_usage_distribution'] = df['cpu_usage_distribution'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
    df['tail_cpu_usage_distribution'] = df['tail_cpu_usage_distribution'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))

    # Padding sequences
    max_seq_length = df['cpu_usage_distribution'].apply(len).max()  # Find the maximum length of sequences
    df['cpu_usage_distribution_padded'] = df['cpu_usage_distribution'].apply(lambda x: pad_sequences([x], maxlen=max_seq_length, padding='post', dtype='float32')[0])
    tail_max_seq_length = df['tail_cpu_usage_distribution'].apply(len).max()  # Find the maximum length of sequences
    df['tail_cpu_usage_distribution_padded'] = df['tail_cpu_usage_distribution'].apply(lambda x: pad_sequences([x], maxlen=tail_max_seq_length, padding='post', dtype='float32')[0])

    # Feature Scaling
    scaler = StandardScaler()
    df['cpu_usage_distribution_scaled'] = df['cpu_usage_distribution_padded'].apply(lambda x: scaler.fit_transform(x.reshape(-1, 1)))
    df['tail_cpu_usage_distribution_scaled'] = df['tail_cpu_usage_distribution_padded'].apply(lambda x: scaler.fit_transform(x.reshape(-1, 1)))

    scaled_features = scaler.fit_transform(df[[ 'resource_request_cpus', 'resource_request_memory',  'poly_maximum_usage_cpus random_sample_usage_cpus', 
                                                'maximum_usage_cpus',  'poly_random_sample_usage_cpus', 'poly_random_sample_usage_cpus^2', 'memory_demand_rolling_mean',
                                                'maximum_usage_memory',  'interaction_feature', 'poly_maximum_usage_cpus^2', 'memory_demand_lag_1',
                                                'random_sample_usage_cpus', 'assigned_memory',  'poly_maximum_usage_cpus', 'memory_demand_rolling_std', 
                                                'start_hour', 'start_dayofweek', 'duration_seconds', 'sample_rate', 'cycles_per_instruction', 
                                                'memory_accesses_per_instruction', 'page_cache_memory', 'priority',
                                            ]])

    # Labels
    labels = df[['average_usage_cpus', 'average_usage_memory']]

    # Convert numpy arrays to pandas DataFrames
    scaled_features_df = pd.DataFrame(scaled_features, columns=[
        'resource_request_cpus', 'resource_request_memory', 'poly_maximum_usage_cpus random_sample_usage_cpus',
        'maximum_usage_cpus', 'poly_random_sample_usage_cpus', 'poly_random_sample_usage_cpus^2', 'memory_demand_rolling_mean',
        'maximum_usage_memory', 'interaction_feature', 'poly_maximum_usage_cpus^2', 'memory_demand_lag_1',
        'random_sample_usage_cpus', 'assigned_memory', 'poly_maximum_usage_cpus', 'memory_demand_rolling_std',
        'start_hour', 'start_dayofweek', 'duration_seconds', 'sample_rate', 'cycles_per_instruction',
        'memory_accesses_per_instruction', 'page_cache_memory', 'priority'])

    # Reshape the data to 2D array
    cpu_usage_reshaped = np.vstack(df['cpu_usage_distribution_scaled']).reshape(-1, max_seq_length)
    tail_cpu_usage_reshaped = np.vstack(df['tail_cpu_usage_distribution_scaled']).reshape(-1, tail_max_seq_length)

    # Create DataFrame
    cpu_usage_df = pd.DataFrame(cpu_usage_reshaped, columns=[f'cpu_usage_{i}' for i in range(max_seq_length)])
    tail_cpu_usage_df = pd.DataFrame(tail_cpu_usage_reshaped, columns=[f'tail_cpu_usage_{i}' for i in range(tail_max_seq_length)])

    # Concatenate all DataFrames
    scaled_features_concatenated = pd.concat([cpu_usage_df, tail_cpu_usage_df, scaled_features_df], axis=1)

    # Split the dataset into training, validation, and testing sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_features_concatenated, labels, test_size=0.2, random_state=42)

    # Convert NumPy arrays back to TensorFlow tensors
    X_train = tf.constant(X_train, dtype=tf.float32)
    X_test = tf.constant(X_test, dtype=tf.float32)
    y_train = tf.constant(y_train, dtype=tf.float32)
    y_test = tf.constant(y_test, dtype=tf.float32)

    # Reshape the input features to add a third dimension for time steps
    x_train_reshaped = tf.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    x_test_reshaped = tf.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


    return (
        x_train_reshaped[idx * 30000 : (idx + 1) * 30000],
        y_train[idx * 30000 : (idx + 1) * 30000],
    ), (
        x_test_reshaped[idx * 1000 : (idx + 1) * 1000],
        y_test[idx * 1000 : (idx + 1) * 1000],
    )


if __name__ == "__main__":
    main()
