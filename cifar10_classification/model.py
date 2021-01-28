from tensorflow.keras.layers import Layer
import tensorflow as tf

class Model(tf.keras.Model):

    def __init__(self):
    
        # 3 conv/pool blocks with 2 conv layers each. 2 dense hidden layers.

        # init super class
        super(Model, self).__init__()

        # define layers
        self.conv_layer_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='tanh', input_shape=(32,32,3))
        self.conv_layer_1_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='tanh')
        self.max_pool_layer_1 = tf.keras.layers.MaxPool2D()

        self.conv_layer_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=None)
        self.conv_layer_2_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=None)
        self.norm_layer_2 = tf.keras.layers.BatchNormalization()
        self.tanh_layer_2 = tf.keras.activations.tanh
        self.relu_layer_2 = tf.keras.activations.relu
        self.max_pool_layer_2 = tf.keras.layers.MaxPool2D()

        self.conv_layer_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation=None)
        self.conv_layer_3_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation=None)
        self.norm_layer_3 = tf.keras.layers.BatchNormalization()
        self.relu_layer_3 = tf.keras.activations.relu
        self.tanh_layer_3 = tf.keras.activations.tanh
        self.max_pool_layer_3 = tf.keras.layers.MaxPool2D()
        
        self.mlp_input = tf.keras.layers.GlobalAveragePooling2D()
        self.hidden_layer_1 = tf.keras.layers.Dense(units=64, activation=tf.keras.activations.tanh)
        self.hidden_layer_2 = tf.keras.layers.Dense(units=32, activation=tf.keras.activations.tanh)
        self.output_layer = tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax)

    @tf.function
    def call(self, x, training=True):

        x = self.conv_layer_1(x)
        x = self.conv_layer_1_2(x)
        x = self.max_pool_layer_1(x)
        
        x = self.conv_layer_2(x)
        x = self.conv_layer_2_2(x)
        x = self.norm_layer_2(x, training)
        # x = self.relu_layer_2(x)
        x = self.tanh_layer_2(x) # better performance than relu in this case
        x = self.max_pool_layer_2(x)
        
        x = self.conv_layer_3(x)
        x = self.conv_layer_3_2(x)
        x = self.norm_layer_3(x, training)
        # x = self.relu_layer_3(x)
        x = self.tanh_layer_3(x)
        x = self.max_pool_layer_3(x)       
        
        x = self.mlp_input(x)
        x = self.hidden_layer_1(x)
        x = self.hidden_layer_2(x)
        x = self.output_layer(x)
        
        return x
