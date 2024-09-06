
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from quantum_and_agi_code.qml_integration import HybridQuantumModel
from quantum_and_agi_code.multi_objective_rl import MultiObjectiveRLAgent
from quantum_and_agi_code.hierarchical_rl import HierarchicalRLAgent


app = Flask(__name__)

# Load models
quantum_model = HybridQuantumModel()
multi_objective_agent = MultiObjectiveRLAgent(state_dim=10, action_dim=3, objectives=['obj1', 'obj2'])
hierarchical_agent = HierarchicalRLAgent(state_dim=10, action_dim=3, sub_task_dims=[5, 7])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/quantum', methods=['GET', 'POST'])
def quantum():
    if request.method == 'POST':
        input_data = request.form['input_data']
        input_array = np.array([float(i) for i in input_data.split(',')])
        
        # Reshape the input array to be 2D (1, features)
        input_array = np.expand_dims(input_array, axis=0)
        
        prediction = quantum_model(input_array)
        return render_template('quantum.html', prediction=prediction.numpy())
    return render_template('quantum.html')


@app.route('/multi-objective-rl', methods=['GET', 'POST'])
def multi_objective_rl():
    if request.method == 'POST':
        state = request.form['state']
        state_array = np.array([float(i) for i in state.split(',')])
        action = multi_objective_agent.select_action(state_array)
        return render_template('multi_objective_rl.html', action=action)
    return render_template('multi_objective_rl.html')

@app.route('/hierarchical-rl', methods=['GET', 'POST'])
def hierarchical_rl():
    if request.method == 'POST':
        state = request.form['state']
        state_array = np.array([float(i) for i in state.split(',')])
        sub_task = hierarchical_agent.select_sub_task(state_array)
        return render_template('hierarchical_rl.html', sub_task=sub_task)
    return render_template('hierarchical_rl.html')

if __name__ == '__main__':
    app.run(debug=True)
