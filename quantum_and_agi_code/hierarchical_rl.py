
import tensorflow as tf
import numpy as np

class HierarchicalRLAgent:
    def __init__(self, state_dim, action_dim, sub_task_dims):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sub_task_models = [self.build_sub_task_model(sub_dim) for sub_dim in sub_task_dims]
        self.meta_model = self.build_meta_model()

    def build_sub_task_model(self, input_dim):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_shape=(input_dim,), activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def build_meta_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_shape=(self.state_dim,), activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(self.sub_task_models), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model

    def select_sub_task(self, state):
        probabilities = self.meta_model.predict(state[np.newaxis])
        return np.argmax(probabilities)

    def update(self, state, sub_task_index, action, reward, next_state, done):
        sub_task_model = self.sub_task_models[sub_task_index]
        target = reward + (1 - done) * 0.99 * np.amax(sub_task_model.predict(next_state[np.newaxis]))
        target_q = sub_task_model.predict(state[np.newaxis])
        target_q[0][action] = target
        sub_task_model.fit(state[np.newaxis], target_q, epochs=1, verbose=0)
        
        meta_target = reward + (1 - done) * 0.99 * np.amax(self.meta_model.predict(next_state[np.newaxis]))
        meta_q = self.meta_model.predict(state[np.newaxis])
        meta_q[0][sub_task_index] = meta_target
        self.meta_model.fit(state[np.newaxis], meta_q, epochs=1, verbose=0)
