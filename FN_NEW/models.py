import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

@tf.custom_gradient
def gradient_reversal_function(x, lambda_):
    def grad(dy):
        return -lambda_ * dy, None
    return tf.identity(x), grad

class GradientReversal(layers.Layer):
    def __init__(self, lambda_=1, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.lambda_ = lambda_
    
    def call(self, x):
        return gradient_reversal_function(x, self.lambda_)
    
    def get_config(self):
        config = super().get_config()
        config.update({'lambda_': self.lambda_})
        return config

class Net(keras.Model):
    def __init__(self, input_shape, grl_lambda=100, **kwargs):
        super(Net, self).__init__(**kwargs)
        self.input_shape_param = input_shape
        self._grl_lambda = grl_lambda
        self.fc1 = layers.Dense(32, input_shape=(input_shape,))
        self.fc2 = layers.Dense(32)
        self.fc3 = layers.Dense(32)
        self.fc4 = layers.Dense(1)
        if self._grl_lambda != 0:
            self.grl = GradientReversal(grl_lambda)
            self.fc5 = layers.Dense(2)
    
    def call(self, x, training=None):
        hidden = self.fc1(x)
        hidden = tf.nn.relu(hidden)
        hidden = tf.keras.layers.Dropout(0.1)(hidden, training=training)
        hidden = self.fc2(hidden)
        hidden = tf.nn.relu(hidden)
        hidden = self.fc3(hidden)
        hidden = tf.nn.relu(hidden)
        y = self.fc4(hidden)
       
        if self._grl_lambda != 0:
            s = self.grl(hidden)
            s = self.fc5(s)
            return y, s
        else:
            return y
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape_param,
            'grl_lambda': self._grl_lambda
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def __reduce__(self):
        return (self.__class__, (self.input_shape_param, self._grl_lambda))

class Net_nodrop(keras.Model):
    def __init__(self, input_shape, grl_lambda=100, **kwargs):
        super(Net_nodrop, self).__init__(**kwargs)
        self.input_shape_param = input_shape
        self._grl_lambda = grl_lambda
        self.fc1 = layers.Dense(32, input_shape=(input_shape,))
        self.fc2 = layers.Dense(32)
        self.fc3 = layers.Dense(32)
        self.fc4 = layers.Dense(1)
        if self._grl_lambda != 0:
            self.grl = GradientReversal(grl_lambda)
            self.fc5 = layers.Dense(2)
    
    def call(self, x, training=None):
        hidden = self.fc1(x)
        hidden = tf.nn.relu(hidden)
        hidden = self.fc2(hidden)
        hidden = tf.nn.relu(hidden)
        hidden = self.fc3(hidden)
        hidden = tf.nn.relu(hidden)
        y = self.fc4(hidden)
       
        if self._grl_lambda != 0:
            s = self.grl(hidden)
            s = self.fc5(s)
            return y, s
        else:
            return y
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape_param,
            'grl_lambda': self._grl_lambda
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def __reduce__(self):
        return (self.__class__, (self.input_shape_param, self._grl_lambda))

class Net_CENSUS(keras.Model):
    def __init__(self, input_shape, grl_lambda=100, **kwargs):
        super(Net_CENSUS, self).__init__(**kwargs)
        self.input_shape_param = input_shape
        self._grl_lambda = grl_lambda
        self.fc1 = layers.Dense(128, input_shape=(input_shape,))
        self.fc2 = layers.Dense(128)
        self.fc3 = layers.Dense(128)
        self.fc4 = layers.Dense(1)
        if self._grl_lambda != 0:
            self.grl = GradientReversal(grl_lambda)
            self.fc5 = layers.Dense(2)
    
    def call(self, x, training=None):
        hidden = self.fc1(x)
        hidden = tf.nn.relu(hidden)
        hidden = tf.keras.layers.Dropout(0.1)(hidden, training=training)
        hidden = self.fc2(hidden)
        hidden = tf.nn.relu(hidden)
        hidden = self.fc3(hidden)
        hidden = tf.nn.relu(hidden)
        y = self.fc4(hidden)
       
        if self._grl_lambda != 0:
            s = self.grl(hidden)
            s = self.fc5(s)
            return y, s
        else:
            return y
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape_param,
            'grl_lambda': self._grl_lambda
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def __reduce__(self):
        return (self.__class__, (self.input_shape_param, self._grl_lambda))