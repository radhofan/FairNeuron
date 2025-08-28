import time
import random
import math
from ray import tune

import tensorflow as tf
import numpy as np
from utils.path_analysis import sample_sort, get_adv
from Evaluate import train_and_evaluate_drop, get_metrics


def Fixate_with_val(net,data_class,epoch=10,dataset='compas',BATCH_SIZE=128):
    def training_function(config):
        THETA, GAMMA = config['THETA'], config['GAMMA']
        train_dataset_s=config['train_dataset_s']
        val_dataset_s=config['val']
        test_dataset_s=config['test']
        x_train_tensor_s=config['x_tensor']
        
        # Convert back to TensorFlow datasets inside the function
        val_loader_s = tf.data.Dataset.from_tensor_slices(val_dataset_s).batch(BATCH_SIZE)
        test_loader_s = tf.data.Dataset.from_tensor_slices(test_dataset_s).batch(BATCH_SIZE)

        adv_data_idx = sample_sort(net,train_dataset_s,THETA,GAMMA)
        if len(adv_data_idx) == len(train_dataset_s[0] if isinstance(train_dataset_s, tuple) else train_dataset_s):
            adv_loader = tf.data.Dataset.from_tensor_slices(train_dataset_s).batch(BATCH_SIZE).shuffle(buffer_size=1000)
            benign_loader = None
        elif len(adv_data_idx) == 0:
            adv_loader = None
            benign_loader = tf.data.Dataset.from_tensor_slices(train_dataset_s).batch(BATCH_SIZE).shuffle(buffer_size=1000)
        else:
            adv_loader, benign_loader = get_adv(train_dataset_s,adv_data_idx,BATCH_SIZE=BATCH_SIZE)
        net_drop, results = train_and_evaluate_drop(adv_loader, benign_loader, val_loader_s, test_loader_s, device='cuda', input_shape=x_train_tensor_s.shape[1],
                                                grl_lambda=0,dataset=config['dataset'])
        result = get_metrics(results, data_class.threshold, 0,dataset=config['dataset'])
        complex_score = result['DP']+result['EO']+(1-result['DP ratio'])-0.01*result['acc']
        if math.isnan(complex_score):
            complex_score = result['DP']+result['EO']-0.01*result['acc']
        tune.report(mean_loss=complex_score)

    our_start=time.time()
    
    # Just use the datasets as they are - no splitting for small datasets
    train_dataset_s = data_class.val_dataset
    val_dataset_s = data_class.val_dataset  
    test_dataset_s = data_class.test_dataset
    
    val_loader_s = tf.data.Dataset.from_tensor_slices(val_dataset_s).batch(BATCH_SIZE)
    test_loader_s = tf.data.Dataset.from_tensor_slices(test_dataset_s).batch(BATCH_SIZE)
    
    x_train_tensor_s = data_class.val_dataset[0] if isinstance(data_class.val_dataset, tuple) else data_class.val_dataset

    val_loader = tf.data.Dataset.from_tensor_slices(data_class.val_dataset).batch(BATCH_SIZE)
    test_loader = tf.data.Dataset.from_tensor_slices(data_class.test_dataset).batch(BATCH_SIZE)
    
    # Convert TensorFlow objects to numpy for Ray Tune compatibility
    if isinstance(train_dataset_s, tuple):
        train_dataset_s_numpy = (train_dataset_s[0].numpy(), train_dataset_s[1].numpy())
    else:
        train_dataset_s_numpy = train_dataset_s.numpy()
    
    if isinstance(val_dataset_s, tuple):
        val_dataset_s_numpy = (val_dataset_s[0].numpy(), val_dataset_s[1].numpy())
    else:
        val_dataset_s_numpy = val_dataset_s.numpy()
        
    if isinstance(test_dataset_s, tuple):
        test_dataset_s_numpy = (test_dataset_s[0].numpy(), test_dataset_s[1].numpy())
    else:
        test_dataset_s_numpy = test_dataset_s.numpy()
        
    x_train_tensor_s_numpy = x_train_tensor_s.numpy()

    analysis = tune.run(
        training_function,
        config={
            'THETA': tune.grid_search([0.1, 0.01, 3e-3, 1e-3, 3e-4, 1e-4]),
            'GAMMA': tune.grid_search([0.95, 0.9, 0.85, 0.8, 0.7, 0.6]),
            'dataset':dataset,
            'train_dataset_s':train_dataset_s_numpy,
            'val':val_dataset_s_numpy,
            'test':test_dataset_s_numpy,
            'x_tensor':x_train_tensor_s_numpy
        },
        resources_per_trial={
            "cpu": 8,
            "gpu": 0,
        }
    )
    best_config=analysis.get_best_config(metric="mean_loss", mode="min")
    print("Best config: ",best_config)
    THETA = best_config['THETA']
    GAMMA = best_config['GAMMA']
    val_end=time.time()
    for i in range(epoch):
        adv_data_idx = sample_sort(net,data_class.train_dataset,THETA,GAMMA)
        PA_end=time.time()
        adv_loader, benign_loader = get_adv(data_class.train_dataset,adv_data_idx,BATCH_SIZE=BATCH_SIZE)
        SS_end=time.time()
        net_drop, results = train_and_evaluate_drop(adv_loader, benign_loader, val_loader, test_loader, device='cuda', input_shape=data_class.x_tensor.shape[1],
                                                grl_lambda=0,dataset=dataset)
        Dropout_end=time.time()
        result = get_metrics(results, data_class.threshold, 0, dataset=dataset)
        data_class.global_results.append(result)

    our_end=time.time()
    cost_time=our_end-our_start
    val_time=val_end-our_start
    PA_time=PA_end-val_end
    SS_time=SS_end-PA_end
    Dropout_time=Dropout_end-SS_end
    print('param selection costs:{} s'.format(val_time))
    print('path analysis costs:{} s'.format(PA_time))
    print('sample separation costs:{} s'.format(SS_time))
    print('partial dropout training costs:{} s'.format(Dropout_time))
    print('total time costs:{} s'.format(cost_time))


def Fixate_with_val_rand(net,data_class,epoch=10,dataset='compas',BATCH_SIZE=128):
    def training_function(config):
        THETA, GAMMA = config['THETA'], config['GAMMA']
        train_dataset_s=config['train_dataset_s']
        val_dataset_s=config['val']
        test_dataset_s=config['test']
        x_train_tensor_s=config['x_tensor']
        
        # Convert back to TensorFlow datasets inside the function
        val_loader_s = tf.data.Dataset.from_tensor_slices(val_dataset_s).batch(BATCH_SIZE)
        test_loader_s = tf.data.Dataset.from_tensor_slices(test_dataset_s).batch(BATCH_SIZE)

        adv_data_idx = sample_sort(net,train_dataset_s,THETA,GAMMA)
        if len(adv_data_idx) == len(train_dataset_s[0] if isinstance(train_dataset_s, tuple) else train_dataset_s):
            benign_loader = None
        else:
            adv_data_idx = random.sample(range(len(train_dataset_s[0] if isinstance(train_dataset_s, tuple) else train_dataset_s)),len(adv_data_idx))
            adv_loader, benign_loader = get_adv(train_dataset_s,adv_data_idx,BATCH_SIZE=BATCH_SIZE)
        net_drop, results = train_and_evaluate_drop(adv_loader, benign_loader, val_loader_s, test_loader_s, device='cuda', input_shape=x_train_tensor_s.shape[1],
                                                grl_lambda=0,dataset=config['dataset'])
        result = get_metrics(results, data_class.threshold, 0,dataset=config['dataset'])
        complex_score = result['DP']+result['EO']+(1-result['DP ratio'])-0.01*result['acc']
        tune.report(mean_loss=complex_score)

    our_start=time.time()
    
    # Just use the datasets as they are - no splitting for small datasets
    train_dataset_s = data_class.val_dataset
    val_dataset_s = data_class.val_dataset  
    test_dataset_s = data_class.test_dataset
    
    val_loader_s = tf.data.Dataset.from_tensor_slices(val_dataset_s).batch(BATCH_SIZE)
    test_loader_s = tf.data.Dataset.from_tensor_slices(test_dataset_s).batch(BATCH_SIZE)
    
    x_train_tensor_s = data_class.val_dataset[0] if isinstance(data_class.val_dataset, tuple) else data_class.val_dataset

    val_loader = tf.data.Dataset.from_tensor_slices(data_class.val_dataset).batch(BATCH_SIZE)
    test_loader = tf.data.Dataset.from_tensor_slices(data_class.test_dataset).batch(BATCH_SIZE)
    
    # Convert TensorFlow objects to numpy for Ray Tune compatibility
    if isinstance(train_dataset_s, tuple):
        train_dataset_s_numpy = (train_dataset_s[0].numpy(), train_dataset_s[1].numpy())
    else:
        train_dataset_s_numpy = train_dataset_s.numpy()
    
    if isinstance(val_dataset_s, tuple):
        val_dataset_s_numpy = (val_dataset_s[0].numpy(), val_dataset_s[1].numpy())
    else:
        val_dataset_s_numpy = val_dataset_s.numpy()
        
    if isinstance(test_dataset_s, tuple):
        test_dataset_s_numpy = (test_dataset_s[0].numpy(), test_dataset_s[1].numpy())
    else:
        test_dataset_s_numpy = test_dataset_s.numpy()
        
    x_train_tensor_s_numpy = x_train_tensor_s.numpy()

    analysis = tune.run(
        training_function,
        config={
            'THETA': tune.grid_search([0.1, 0.01, 3e-3, 1e-3, 3e-4, 1e-4]),
            'GAMMA': tune.grid_search([0.95, 0.9, 0.85, 0.8, 0.7, 0.6]),
            'dataset':dataset,
            'train_dataset_s':train_dataset_s_numpy,
            'val':val_dataset_s_numpy,
            'test':test_dataset_s_numpy,
            'x_tensor':x_train_tensor_s_numpy
        },
        resources_per_trial={
            "cpu": 8,
            "gpu": 0,
        }
    )
    best_config=analysis.get_best_config(metric="mean_loss", mode="min")
    print("Best config: ",best_config)
    THETA = best_config['THETA']
    GAMMA = best_config['GAMMA']
    val_end=time.time()
    for i in range(epoch):
        adv_data_idx = sample_sort(net,data_class.train_dataset,THETA,GAMMA)
        PA_end=time.time()
        adv_loader, benign_loader = get_adv(data_class.train_dataset,adv_data_idx,BATCH_SIZE=BATCH_SIZE)
        SS_end=time.time()
        net_drop, results = train_and_evaluate_drop(adv_loader, benign_loader, val_loader, test_loader, device='cuda', input_shape=data_class.x_tensor.shape[1],
                                                grl_lambda=0,dataset=dataset)
        Dropout_end=time.time()
        result = get_metrics(results, data_class.threshold, 0, dataset=dataset)
        data_class.global_results.append(result)

    our_end=time.time()
    cost_time=our_end-our_start
    val_time=val_end-our_start
    PA_time=PA_end-val_end
    SS_time=SS_end-PA_end
    Dropout_time=Dropout_end-SS_end
    print('param selection costs:{} s'.format(val_time))
    print('path analysis costs:{} s'.format(PA_time))
    print('sample separation costs:{} s'.format(SS_time))
    print('partial dropout training costs:{} s'.format(Dropout_time))
    print('total time costs:{} s'.format(cost_time))