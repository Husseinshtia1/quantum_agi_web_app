import yfinance as yf
import numpy as np
import tensorflow as tf
import pennylane as qml
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import load_model  

# Define the quantum device
dev = qml.device("default.qubit", wires=2)

# Define the quantum circuit
@qml.qnode(dev, interface='tf')
def quantum_circuit(inputs, weights):
    qml.RX(inputs[0], wires=0)
    qml.RY(inputs[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RX(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    # Ensure the output is real
    return qml.expval(qml.PauliZ(0)).real

# Define the Cross Attention layer
class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(CrossAttention, self).__init__()
        self.units = units
        self.query_dense = tf.keras.layers.Dense(units)
        self.key_dense = tf.keras.layers.Dense(units)
        self.value_dense = tf.keras.layers.Dense(units)

    def call(self, query, key, value):
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)
        attention_output = tf.matmul(attention_scores, value)
        
        return attention_output

# Define the Quantum Stock Predictor class
class QuantumStockPredictor:
    def __init__(self, ticker, start_date, end_date, test_size=0.2, random_state=42):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def fetch_data(self):
        # Fetch historical stock data
        stock_data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        return stock_data

    def preprocess_data(self, stock_data):
        # Preprocess the data (e.g., normalize, handle missing values)
        stock_data = stock_data.dropna()
        stock_data['Normalized Close'] = stock_data['Close'] / stock_data['Close'].max()
        return stock_data

    def get_features_and_labels(self, stock_data, window_size=5):
        # Create features and labels for training
        features = []
        labels = []
        for i in range(len(stock_data) - window_size):
            features.append(stock_data['Normalized Close'].iloc[i:i+window_size].values)
            labels.append(stock_data['Normalized Close'].iloc[i+window_size])
        return np.array(features), np.array(labels)
    
    def split_data(self, features, labels):
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, labels, test_size=self.test_size, random_state=self.random_state
        )

    def build_model(self):
        # Build the hybrid quantum model with Cross Attention
        class HybridQuantumModel(tf.keras.Model):
            def __init__(self):
                super(HybridQuantumModel, self).__init__()
                self.dense1 = tf.keras.layers.Dense(4, activation='relu')
                self.dense2 = tf.keras.layers.Dense(2, activation='relu')
                self.cross_attention = CrossAttention(units=2)
                self.dense3 = tf.keras.layers.Dense(1, activation='sigmoid')

            def call(self, inputs):
                x = self.dense1(inputs)
                x = self.dense2(x)

                x_flat = tf.reshape(x, [-1])
                weights = np.random.randn(2)

                quantum_output = quantum_circuit(x_flat, weights)
                quantum_output = tf.reshape(quantum_output, (1, -1))

                # Apply Cross Attention
                attention_output = self.cross_attention(quantum_output, x, x)

                output = self.dense3(attention_output)
                return output
        
        self.model = HybridQuantumModel()

    def compile_model(self, learning_rate=0.001):
        # Compile the model with given learning rate
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                           loss='mse', metrics=['mae'])
    
    def train_model(self, epochs=10, batch_size=32):
        # Train the model
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size)

    def evaluate_model(self):
        # Evaluate the model on the test set
        loss, mae = self.model.evaluate(self.X_test, self.y_test)
        print(f"Test MAE: {mae:.4f}")
        return loss, mae

    def cross_validation(self, k=5):
        # Perform k-fold cross-validation
        kfold = KFold(n_splits=k, shuffle=True, random_state=self.random_state)
        cv_scores = []

        for train_idx, val_idx in kfold.split(self.X_train):
            X_train_fold, X_val_fold = self.X_train[train_idx], self.X_train[val_idx]
            y_train_fold, y_val_fold = self.y_train[train_idx], self.y_train[val_idx]

            self.build_model()
            self.compile_model()
            self.model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, verbose=0)
            val_loss, val_mae = self.model.evaluate(X_val_fold, y_val_fold, verbose=0)
            cv_scores.append(val_mae)

        print(f"Cross-validation MAE scores: {cv_scores}")
        print(f"Mean CV MAE: {np.mean(cv_scores):.4f}")

    def expand_model(self):
        # Experiment with adding more layers, adjusting the quantum circuit, or other hyperparameters
        # Example: Add an additional dense layer
        self.model.add(tf.keras.layers.Dense(4, activation='relu'))

    def deploy_model(self):
        # Integrate into Flask app or deploy as a service
        print("Model deployment is not yet implemented.")
        # Placeholder for deployment logic

    def save_model(self, model_filename='quantum_stock_predictor.h5'):
        # Save the trained model
        self.model.save(model_filename)
        print(f"Model saved as {model_filename}")

    def load_model(self, model_filename='quantum_stock_predictor.h5'):
        # Load a saved model
        self.model = load_model(model_filename)
        print(f"Model loaded from {model_filename}")

    def run(self):
        # Full workflow
        stock_data = self.fetch_data()
        processed_data = self.preprocess_data(stock_data)
        features, labels = self.get_features_and_labels(processed_data)
        self.split_data(features, labels)
        self.build_model()
        self.compile_model()
        self.train_model()
        self.evaluate_model()
        self.cross_validation()
        self.save_model()

# Example usage
if __name__ == "__main__":
    predictor = QuantumStockPredictor(ticker='AAPL', start_date='2020-01-01', end_date='2021-01-01')
    predictor.run()
