import random
import pandas as pd
import tensorflow as tf
import numpy as np

def get_paras(net):
    """Get all trainable parameters from the network"""
    paras = []
    for layer in net.layers:
        if hasattr(layer, 'weights'):
            for weight in layer.weights:
                paras.append(weight.numpy())
    return paras

def get_active_neurons4(net, sample):
    """Get activations from all 4 layers using GradientTape for intermediate outputs"""
    # Handle different input formats
    if hasattr(sample, 'numpy'):
        sample = sample.numpy()
    
    # Ensure sample is numpy array and reshape to match expected input
    sample = np.array(sample, dtype=np.float32)
    
    # If sample is scalar or wrong shape, pad/reshape to (18,)
    if sample.size == 1:
        sample = np.pad(sample.flatten(), (0, 17), 'constant')
    elif sample.size < 18:
        sample = np.pad(sample.flatten(), (0, 18 - sample.size), 'constant')
    elif sample.size > 18:
        sample = sample.flatten()[:18]
    else:
        sample = sample.flatten()
    
    # Add batch dimension: (18,) -> (1, 18)
    sample_tensor = tf.expand_dims(tf.convert_to_tensor(sample, dtype=tf.float32), 0)
    
    neurons = []
    layer_outputs = []
    x = sample_tensor
    
    for layer in net.layers:
        x = layer(x)
        if hasattr(layer, 'activation') or 'dense' in layer.name.lower():
            layer_outputs.append(x)
    
    # Get the first 4 layer outputs
    for i in range(min(4, len(layer_outputs))):
        neurons.append(layer_outputs[i].numpy())
    
    return neurons

def get_contrib4(paras, neurons):
    """Calculate contributions for the first 3 layers"""
    contrib_list = []
    
    # Assuming paras are [weight1, bias1, weight2, bias2, weight3, bias3, weight4, bias4]
    for i in range(3):
        if i*2+2 < len(paras):
            # Get weights for next layer
            weights = paras[i*2+2]  # weights for layer i+1
            # Calculate contribution: neuron_output * weights
            if len(neurons[i].shape) > 1:
                neuron_vals = neurons[i][0]  # Remove batch dimension
            else:
                neuron_vals = neurons[i]
            
            # Matrix multiplication: neuron_vals @ weights
            contrib = np.outer(neuron_vals, weights.T) if len(weights.shape) > 1 else neuron_vals * weights
            contrib_list.append(contrib)
    
    return contrib_list

def get_path_set4(net, sample, GAMMA=0.9):
    """Get path set for a sample with TensorFlow model"""
    active_neuron_indice = [[], [], [], []]
    path_set = set()
    
    neurons = get_active_neurons4(net, sample)
    paras = get_paras(net)
    contrib_list = get_contrib4(paras, neurons)
    
    # Get output neuron with highest activation
    if len(neurons) >= 4:
        output_neurons = neurons[3][0] if len(neurons[3].shape) > 1 else neurons[3]
        active_neuron_indice[3].append(np.argmax(output_neurons))
    
    # Backtrack through layers
    for i in range(3):
        L = 3 - i
        for j in active_neuron_indice[L]:
            if L-1 < len(contrib_list) and j < len(contrib_list[L-1]):
                # Get contributions for this neuron
                if len(contrib_list[L-1].shape) > 1:
                    contribs = contrib_list[L-1][j] if contrib_list[L-1].shape[0] > j else contrib_list[L-1][:, j]
                else:
                    contribs = contrib_list[L-1]
                
                # Sort contributions in descending order
                sorted_indices = np.argsort(contribs)[::-1]
                sorted_values = contribs[sorted_indices]
                
                # Get neuron activation for threshold
                neuron_val = neurons[L][0][j] if len(neurons[L].shape) > 1 else neurons[L][j]
                threshold = GAMMA * neuron_val
                
                cum_sum = 0
                for k in range(len(sorted_values)):
                    cum_sum += sorted_values[k]
                    active_neuron_indice[L-1].append(sorted_indices[k])
                    path_set.add((L, sorted_indices[k], j))
                    if cum_sum >= threshold:
                        break
    
    return path_set

def sample_sort(net, train_dataset, THETA=1e-3, GAMMA=0.9):
    """TensorFlow version of sample_sort"""
    # No need for .cpu() in TensorFlow
    path_set_list = []
    
    for i in range(len(train_dataset)):
        # Extract sample (assuming train_dataset[i][0] is the input)
        sample = train_dataset[i][0]
        if hasattr(sample, 'numpy'):
            sample = sample.numpy()
        
        path_set = get_path_set4(net, sample, GAMMA=GAMMA)
        path_set_list.append(path_set)
    
    # Convert to pandas for counting
    path_set_tuples = [tuple(sorted(ps)) for ps in path_set_list]
    v = pd.Series(path_set_tuples).value_counts().reset_index()
    v.columns = ['pathset', 'counts']
    
    # Find adversarial samples
    threshold = max(v.iloc[0]['counts'] * THETA, 1)
    rare_pathsets = set(v[v['counts'] <= threshold]['pathset'])
    
    adv_data_idx = []
    for i in range(len(path_set_list)):
        if tuple(sorted(path_set_list[i])) in rare_pathsets:
            adv_data_idx.append(i)
    
    print("frac:{}".format(len(adv_data_idx)/len(train_dataset)))
    return adv_data_idx

v_list = []
def sample_sort_test(net, train_dataset, THETA=1e-3, GAMMA=0.9):
    """TensorFlow version of sample_sort_test"""
    global v_list
    
    path_set_list = []
    for i in range(len(train_dataset)):
        sample = train_dataset[i][0]
        if hasattr(sample, 'numpy'):
            sample = sample.numpy()
        
        path_set = get_path_set4(net, sample, GAMMA=GAMMA)
        path_set_list.append(path_set)
    
    path_set_tuples = [tuple(sorted(ps)) for ps in path_set_list]
    v = pd.Series(path_set_tuples).value_counts().reset_index()
    v.columns = ['pathset', 'counts']
    v_list.append(v)
    
    threshold = max(v.iloc[0]['counts'] * THETA, 1)
    rare_pathsets = set(v[v['counts'] <= threshold]['pathset'])
    
    adv_data_idx = []
    for i in range(len(path_set_list)):
        if tuple(sorted(path_set_list[i])) in rare_pathsets:
            adv_data_idx.append(i)
    
    print("frac:{}".format(len(adv_data_idx)/len(train_dataset)))
    return adv_data_idx

def get_adv(train_dataset, adv_data_idx, BATCH_SIZE=128):
    """Convert to TensorFlow dataset format"""
    # Separate adversarial and benign samples
    adv_samples = []
    benign_samples = []
    
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        
        if i in adv_data_idx:
            adv_samples.append(sample)
        else:
            benign_samples.append(sample)
    
    # Create simple TensorFlow datasets
    if adv_samples:
        adv_dataset = tf.data.Dataset.from_tensor_slices(adv_samples).batch(BATCH_SIZE)
    else:
        adv_dataset = None
    
    if benign_samples:
        benign_dataset = tf.data.Dataset.from_tensor_slices(benign_samples).batch(BATCH_SIZE)
    else:
        benign_dataset = None
    
    return adv_dataset, benign_dataset

def get_adv_rand(train_dataset, adv_data_idx, BATCH_SIZE=128):
    """Random version with TensorFlow"""
    # Generate random indices with same length as adv_data_idx
    random_adv_idx = random.choices(range(len(train_dataset)), k=len(adv_data_idx))
    return get_adv(train_dataset, random_adv_idx, BATCH_SIZE)