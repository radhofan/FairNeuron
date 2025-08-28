import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, losses, metrics
from pycm import ConfusionMatrix
from tqdm import trange

from models import Net, Net_CENSUS, Net_nodrop


class bm:
    def __init__(self, df):
        self._df = df

    def P(self, **kwargs):
        """
        Declares the random variables from the set `kwargs`.
        """
        self._variables = kwargs
        return self

    def given(self, **kwargs):
        """
        Calculates the probability on a finite set of samples with `kwargs` in the
        conditioning set. 
        """
        self._given = kwargs
        
        # Here's where the magic happens
        prior = True
        posterior = True
        
        for k in self._variables:
            if type(self._variables[k]) == type(lambda x:x):
                posterior = posterior & (self._df[k].apply(self._variables[k]))
            else:
                posterior = posterior & (self._df[k] == self._variables[k])

        
        for k in self._given:
            if type(self._given[k]) == type(lambda x:x):
                prior = prior & (self._df[k].apply(self._given[k]))
                posterior = posterior & (self._df[k].apply(self._given[k]))
            else:
                prior = prior & (self._df[k] == self._given[k])
                posterior = posterior & (self._df[k] == self._given[k])
        return posterior.sum()/prior.sum()


def get_metrics(results, threshold, fraction, dataset='compas'):
    "Create the metrics from an output df."

    # Calculate biases after training
    dem_parity = abs(
        bm(results).P(pred=lambda x: x > threshold).given(race=0)
        - bm(results).P(pred=lambda x: x > threshold).given(
            race=1))

    eq_op = abs(
        bm(results).P(pred=lambda x: x > threshold).given(race=0, compas=True)
        - bm(results).P(pred=lambda x: x > threshold).given(race=1, compas=True))

    dem_parity_ratio = abs(
        bm(results).P(pred=lambda x: x > threshold).given(race=0)
        / bm(results).P(pred=lambda x: x > threshold).given(
            race=1))

    cm = ConfusionMatrix(actual_vector=(results['true'] == True).values,
                         predict_vector=(results['pred'] > threshold).values)
    if dataset=='compas':
        cm_high_risk = ConfusionMatrix(actual_vector=(results['compas'] > 8).values,
                             predict_vector=(results['pred'] > 8).values)

        result = {"DP": dem_parity,
                  "EO": eq_op,
                  "DP ratio": dem_parity_ratio,
                  "acc": cm.Overall_ACC,
                  "acc_ci_min": cm.CI95[0],
                  "acc_ci_max": cm.CI95[1],
                  "f1": cm.F1_Macro,
                  "acc_high_risk": cm_high_risk.Overall_ACC,
                  "acc_ci_min_high_risk": cm_high_risk.CI95[0],
                  "acc_ci_max_high_risk": cm_high_risk.CI95[1],
                  "f1_high_risk": cm_high_risk.F1_Macro,
                  "adversarial_fraction": fraction
                  }
    else:
        result = {"DP": dem_parity,
                  "EO": eq_op,
                  "DP ratio": dem_parity_ratio,
                  "acc": cm.Overall_ACC,
                  "acc_ci_min": cm.CI95[0],
                  "acc_ci_max": cm.CI95[1],
                  "f1": cm.F1_Macro,
                  "adversarial_fraction": fraction
                  }

    return result

def train_and_evaluate(train_loader,
                       val_loader,
                       test_loader,
                       device,
                       input_shape,
                       grl_lambda=None,
                       model=None,
                       dataset='compas'):
    """

    :param train_loader: TensorFlow-like DataLoader with training data.
    :param val_loader: TensorFlow-like DataLoader with validation data.
    :param test_loader: TensorFlow-like DataLoader with testing data.
    :param device: The target device for the training.
    :return: A tuple: (trained TensorFlow model, dataframe with results on test set)
    """

    tf.random.set_seed(0)

    # grl_lambda = 50
    epochs = 50

    if model is None:
        # Redefine the model
        if dataset=='census':
            model = Net_CENSUS(input_shape=input_shape, grl_lambda=grl_lambda)
        elif dataset=='compas_nodrop':
            model = Net_nodrop(input_shape=input_shape, grl_lambda=grl_lambda)
        else:
            model = Net(input_shape=input_shape, grl_lambda=grl_lambda)

    criterion = losses.MeanSquaredError()
    criterion_bias = losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = optimizers.Adam(learning_rate=1e-2)
    
    # TensorFlow doesn't have exact equivalent to ReduceLROnPlateau, so we'll implement a simple version
    best_val_loss = float('inf')
    patience_counter = 0
    reduce_lr_patience = 5
    reduce_lr_threshold = 0.3
    lr_reduce_factor = 0.5

    training_losses = []
    validation_losses = []

    t_prog = trange(epochs, desc='Training neural network', leave=False, position=1, mininterval=5)

    for epoch in t_prog:
        
        batch_losses = []
        for x_batch, y_batch, _, s_batch in train_loader:
            x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
            y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)
            s_batch = tf.convert_to_tensor(s_batch, dtype=tf.float32)

            with tf.GradientTape() as tape:
                if grl_lambda is not None and grl_lambda != 0:
                    outputs, outputs_protected = model(x_batch, training=True)
                    loss = criterion(y_batch, outputs) + criterion_bias(tf.argmax(s_batch, axis=1), outputs_protected)
                else:
                    outputs = model(x_batch, training=True)
                    loss = criterion(y_batch, outputs)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            batch_losses.append(loss.numpy())

        training_loss = np.mean(batch_losses)
        training_losses.append(training_loss)

        val_losses = []
        for x_val, y_val, _, s_val in val_loader:
            x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
            y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
            s_val = tf.convert_to_tensor(s_val, dtype=tf.float32)
            
            if grl_lambda is not None and grl_lambda != 0:
                yhat, s_hat = model(x_val, training=False)
                val_loss = (criterion(y_val, yhat) + criterion_bias(tf.argmax(s_val, axis=1), s_hat)).numpy()
            else:
                yhat = model(x_val, training=False)
                val_loss = criterion(y_val, yhat).numpy()
            val_losses.append(val_loss)
        
        validation_loss = np.mean(val_losses)
        validation_losses.append(validation_loss)

        # Simple learning rate scheduler
        if validation_loss < best_val_loss - reduce_lr_threshold:
            best_val_loss = validation_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= reduce_lr_patience:
                new_lr = optimizer.learning_rate * lr_reduce_factor
                optimizer.learning_rate.assign(new_lr)
                patience_counter = 0

        t_prog.set_postfix({"epoch": epoch, "training_loss": training_loss,
                            "validation_loss": validation_loss}, refresh=False)

    test_losses = []
    test_results = []
    for x_test, y_test, ytrue, s_true in test_loader:
        x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
        y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
        s_true = tf.convert_to_tensor(s_true, dtype=tf.float32)
        
        if grl_lambda is not None and grl_lambda != 0:
            yhat, s_hat = model(x_test, training=False)
            test_loss = (criterion(y_test, yhat) + criterion_bias(tf.argmax(s_true, axis=1), s_hat)).numpy()
            test_losses.append(test_loss)
            test_results.append({"y_hat": yhat, "y_true": ytrue, "y_compas": y_test, "s": s_true, "s_hat": s_hat})
        else:
            yhat = model(x_test, training=False)
            test_loss = criterion(y_test, yhat).numpy()
            test_losses.append(test_loss)
            test_results.append({"y_hat": yhat, "y_true": ytrue, "y_compas": y_test, "s": s_true})

    results = test_results[0]['y_hat']
    outcome = test_results[0]['y_true']
    compas = test_results[0]['y_compas']
    protected_results = test_results[0]['s']
    if grl_lambda is not None and grl_lambda != 0:
        protected = test_results[0]['s_hat']
    
    for r in test_results[1:]:
        results = tf.concat([results, r['y_hat']], axis=0)
        outcome = tf.concat([outcome, r['y_true']], axis=0)
        compas = tf.concat([compas, r['y_compas']], axis=0)
        protected_results = tf.concat([protected_results, r['s']], axis=0)
        if grl_lambda is not None and grl_lambda != 0:
            protected = tf.concat([protected, r['s_hat']], axis=0)

    df = pd.DataFrame(data=results.numpy(), columns=['pred'])

    df['true'] = outcome.numpy()
    df['compas'] = compas.numpy()
    df['race'] = protected_results.numpy()[:, 0]
    if grl_lambda is not None and grl_lambda != 0:
        df['race_hat'] = protected.numpy()[:, 0]

    return model, df


def train_and_evaluate_drop(adv_loader,
                            benign_loader,
                            val_loader,
                            test_loader,
                            device,
                            input_shape,
                            grl_lambda=None,
                            model=None,
                            dataset='compas'):
    """

    :param adv_loader: TensorFlow-like DataLoader with adversarial training data.
    :param benign_loader: TensorFlow-like DataLoader with benign training data.
    :param val_loader: TensorFlow-like DataLoader with validation data.
    :param test_loader: TensorFlow-like DataLoader with testing data.
    :param device: The target device for the training.
    :return: A tuple: (trained TensorFlow model, dataframe with results on test set)
    """

    epochs = 50

    if model is None:
        if dataset=='CENSUS':
            model = Net_CENSUS(input_shape=input_shape, grl_lambda=grl_lambda)
        else:
            model = Net(input_shape=input_shape, grl_lambda=grl_lambda)

    criterion = losses.MeanSquaredError()
    criterion_bias = losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = optimizers.Adam(learning_rate=1e-2)
    
    # Simple learning rate scheduler
    best_val_loss = float('inf')
    patience_counter = 0
    reduce_lr_patience = 5
    reduce_lr_threshold = 0.3
    lr_reduce_factor = 0.5

    training_losses = []
    validation_losses = []

    t_prog = trange(epochs, desc='Training neural network', leave=False, position=1, mininterval=5)

    for epoch in t_prog:
        batch_losses = []
        
        # Train on adversarial data
        for x_batch, y_batch, _, s_batch in adv_loader:
            x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
            y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)
            s_batch = tf.convert_to_tensor(s_batch, dtype=tf.float32)

            with tf.GradientTape() as tape:
                if grl_lambda is not None and grl_lambda != 0:
                    outputs, outputs_protected = model(x_batch, training=True)
                    loss = criterion(y_batch, outputs) + criterion_bias(tf.argmax(s_batch, axis=1), outputs_protected)
                else:
                    outputs = model(x_batch, training=True)
                    loss = criterion(y_batch, outputs)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            batch_losses.append(loss.numpy())
            
        # Train on benign data
        if benign_loader is not None:
            for x_batch, y_batch, _, s_batch in benign_loader:
                x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
                y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)
                s_batch = tf.convert_to_tensor(s_batch, dtype=tf.float32)

                with tf.GradientTape() as tape:
                    if grl_lambda is not None and grl_lambda != 0:
                        outputs, outputs_protected = model(x_batch, training=False)
                        loss = criterion(y_batch, outputs) + criterion_bias(tf.argmax(s_batch, axis=1), outputs_protected)
                    else:
                        outputs = model(x_batch, training=False)
                        loss = criterion(y_batch, outputs)

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                batch_losses.append(loss.numpy())

        training_loss = np.mean(batch_losses)
        training_losses.append(training_loss)

        val_losses = []
        for x_val, y_val, _, s_val in val_loader:
            x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
            y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
            s_val = tf.convert_to_tensor(s_val, dtype=tf.float32)
            
            if grl_lambda is not None and grl_lambda != 0:
                yhat, s_hat = model(x_val, training=False)
                val_loss = (criterion(y_val, yhat) + criterion_bias(tf.argmax(s_val, axis=1), s_hat)).numpy()
            else:
                yhat = model(x_val, training=False)
                val_loss = criterion(y_val, yhat).numpy()
            val_losses.append(val_loss)
        
        validation_loss = np.mean(val_losses)
        validation_losses.append(validation_loss)

        # Simple learning rate scheduler
        if validation_loss < best_val_loss - reduce_lr_threshold:
            best_val_loss = validation_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= reduce_lr_patience:
                new_lr = optimizer.learning_rate * lr_reduce_factor
                optimizer.learning_rate.assign(new_lr)
                patience_counter = 0

        t_prog.set_postfix({"epoch": epoch, "training_loss": training_loss,
                            "validation_loss": validation_loss}, refresh=False)

    test_losses = []
    test_results = []
    for x_test, y_test, ytrue, s_true in test_loader:
        x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
        y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
        s_true = tf.convert_to_tensor(s_true, dtype=tf.float32)
        
        if grl_lambda is not None and grl_lambda != 0:
            yhat, s_hat = model(x_test, training=False)
            test_loss = (criterion(y_test, yhat) + criterion_bias(tf.argmax(s_true, axis=1), s_hat)).numpy()
            test_losses.append(test_loss)
            test_results.append({"y_hat": yhat, "y_true": ytrue, "y_compas": y_test, "s": s_true, "s_hat": s_hat})
        else:
            yhat = model(x_test, training=False)
            test_loss = criterion(y_test, yhat).numpy()
            test_losses.append(test_loss)
            test_results.append({"y_hat": yhat, "y_true": ytrue, "y_compas": y_test, "s": s_true})

    results = test_results[0]['y_hat']
    outcome = test_results[0]['y_true']
    compas = test_results[0]['y_compas']
    protected_results = test_results[0]['s']
    if grl_lambda is not None and grl_lambda != 0:
        protected = test_results[0]['s_hat']
    
    for r in test_results[1:]:
        results = tf.concat([results, r['y_hat']], axis=0)
        outcome = tf.concat([outcome, r['y_true']], axis=0)
        compas = tf.concat([compas, r['y_compas']], axis=0)
        protected_results = tf.concat([protected_results, r['s']], axis=0)
        if grl_lambda is not None and grl_lambda != 0:
            protected = tf.concat([protected, r['s_hat']], axis=0)

    df = pd.DataFrame(data=results.numpy(), columns=['pred'])

    df['true'] = outcome.numpy()
    df['compas'] = compas.numpy()
    df['race'] = protected_results.numpy()[:, 0]
    if grl_lambda is not None and grl_lambda != 0:
        df['race_hat'] = protected.numpy()[:, 0]

    return model, df