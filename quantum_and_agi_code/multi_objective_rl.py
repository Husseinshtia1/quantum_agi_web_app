
import tensorflow as tf
import numpy as np

class MultiObjectiveRLAgent:
    def __init__(self, state_dim, action_dim, objectives):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.objectives = objectives
        self.models = [self.build_model() for _ in objectives]
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_shape=(self.state_dim,), activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def select_action(self, state):
        q_values = [model.predict(state[np.newaxis]) for model in self.models]
        avg_q_values = np.mean(q_values, axis=0)
        return np.argmax(avg_q_values)

    def update(self, state, action, rewards, next_state, done):
        for i, model in enumerate(self.models):
            target = rewards[i] + (1 - done) * 0.99 * np.amax(model.predict(next_state[np.newaxis]))
            target_q = model.predict(state[np.newaxis])
            target_q[0][action] = target
            model.fit(state[np.newaxis], target_q, epochs=1, verbose=0)
