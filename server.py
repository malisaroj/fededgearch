from typing import Dict, Optional, Tuple
from pathlib import Path
import flwr as fl
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
from keras.utils import pad_sequences
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score, explained_variance_score
from tensorflow.keras.initializers import GlorotUniform

def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    #model = tf.keras.applications.EfficientNetB0(
    #    input_shape=(32, 32, 3), weights=None, classes=10
    #)
    #model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
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

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=3,
        min_evaluate_clients=2,
        min_available_clients=10,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )

    # Start Flower server (SSL-enabled) for 100 rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=100),
        strategy=strategy,
        certificates=(
            Path(".cache/certificates/ca.crt").read_bytes(),
            Path(".cache/certificates/server.pem").read_bytes(),
            Path(".cache/certificates/server.key").read_bytes(),
        ),
    )

    # Save the trained model after the training is completed
    model_save_path  = Path(".cache")
    model_file = model_save_path / "trained_model.h5"

    # Check if the model file already exists, and replace it if necessary
    if model_file.exists():
        print("A trained model already exists. Replacing it.")
        try:
            model_file.unlink()  # This removes the file
        except PermissionError as e:
            print(f"Error removing existing model file: {e}")
            # Handle the error as needed, e.g., by renaming the existing file
            backup_file = model_save_path / "backup_trained_model.h5"
            model_file.rename(backup_file)
            print(f"Existing model file has been renamed to {backup_file}")
    else:
        print("No existing model file found.")

    # Save the new model
    model.save(model_file)
    print(f"Model saved to {model_file}")

    # Save the new model
    model.save(os.path.join(model_save_path, "trained_model.h5"))

    # Plot the metrics
    # Call this function passing all evaluation metrics lists as arguments after federated learning completes.
    plot_metrics(eval_loss, eval_accuracy, eval_mae, eval_rmse, eval_r2, eval_msle, eval_variance)

def get_evaluate_fn(model):

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

    # Calculate the dynamic end index based on the length of the training data
    # Use the last 5k training examples as a validation set

    dynamic_end_index = len(x_train_reshaped) - 5000

    x_val, y_val = x_train_reshaped[dynamic_end_index:], y_train[dynamic_end_index:]  # make the last part of the code dynamic


    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        predictions = model.predict(x_val)
        
        # Calculate additional metrics
        mae = mean_absolute_error(y_val, predictions)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        r2 = r2_score(y_val, predictions)
        msle = mean_squared_log_error(y_val, predictions)
        variance = explained_variance_score(y_val, predictions)

        # Append the additional metrics to their respective lists
        eval_mae.append(mae)
        eval_rmse.append(rmse)
        eval_r2.append(r2)
        eval_msle.append(msle)
        eval_variance.append(variance)

        # Return MSE and accuracy as before
        loss, accuracy = model.evaluate(x_val, y_val)
        eval_loss.append(loss)
        eval_accuracy.append(accuracy)

        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one local epoch,
    increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if server_round < 2 else 20,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five batches) during
    rounds one to three, then increase to ten local evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}

def plot_metrics(eval_loss, eval_accuracy, eval_mae, eval_rmse, eval_r2, eval_msle, eval_variance):
    rounds = np.arange(1, len(eval_loss) + 1)

    plt.figure(figsize=(18, 12))

    plt.subplot(3, 3, 1)
    plt.plot(rounds, eval_loss)
    plt.xlabel("Round")
    plt.ylabel("Evaluation Loss (MSE)")
    plt.title("Evaluation Loss (MSE) over Rounds")

    plt.subplot(3, 3, 2)
    plt.plot(rounds, eval_accuracy)
    plt.xlabel("Round")
    plt.ylabel("Evaluation Accuracy")
    plt.title("Evaluation Accuracy over Rounds")

    plt.subplot(3, 3, 3)
    plt.plot(rounds, eval_mae)
    plt.xlabel("Round")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.title("Mean Absolute Error (MAE) over Rounds")

    plt.subplot(3, 3, 4)
    plt.plot(rounds, eval_rmse)
    plt.xlabel("Round")
    plt.ylabel("Root Mean Squared Error (RMSE)")
    plt.title("Root Mean Squared Error (RMSE) over Rounds")

    plt.subplot(3, 3, 5)
    plt.plot(rounds, eval_msle)
    plt.xlabel("Round")
    plt.ylabel("Mean Squared Logarithmic Error (MSLE)")
    plt.title("Mean Squared Logarithmic Error (MSLE) over Rounds")

    plt.subplot(3, 3, 6)
    plt.plot(rounds, eval_variance)
    plt.xlabel("Round")
    plt.ylabel("Explained Variance Score")
    plt.title("Explained Variance Score over Rounds")

    plt.subplot(3, 3, 7)
    plt.plot(rounds, eval_r2)
    plt.xlabel("Round")
    plt.ylabel("R2")
    plt.title("R2 over Rounds")


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Initialize lists to store loss and accuracy
    eval_loss = []
    eval_accuracy = []
    # Initialize lists to store additional evaluation metrics
    eval_mae = []
    eval_rmse = []
    eval_r2 = []
    eval_msle = []
    eval_variance = []
    main()
