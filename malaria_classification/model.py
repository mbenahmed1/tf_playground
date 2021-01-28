from tensorflow.keras.layers import Layer
import tensorflow as tf

class Model(Layer):

    def __init__(self):
    
        # 3 conv/pool blocks with 2 conv layers each. 2 dense hidden layers.

        # init super class
        super(Model, self).__init__()

        # define layers
        self.conv_layer_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=(128,128,3))
        self.conv_layer_1_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')
        self.max_pool_layer_1 = tf.keras.layers.MaxPool2D()

        self.conv_layer_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')
        self.conv_layer_2_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')
        self.norm_layer_2 = tf.keras.layers.BatchNormalization()
        self.max_pool_layer_2 = tf.keras.layers.MaxPool2D()

        self.conv_layer_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')
        self.conv_layer_3_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')
        self.norm_layer_3 = tf.keras.layers.BatchNormalization()
        self.max_pool_layer_3 = tf.keras.layers.MaxPool2D()
        
        self.mlp_input = tf.keras.layers.GlobalAveragePooling2D()
        self.hidden_layer_1 = tf.keras.layers.Dense(units=128, activation=tf.keras.activations.sigmoid)
        self.hidden_layer_2 = tf.keras.layers.Dense(units=64, activation=tf.keras.activations.sigmoid)
        self.output_layer = tf.keras.layers.Dense(units=2, activation=tf.keras.activations.softmax)

    @tf.function
    def call(self, x):

        x = self.conv_layer_1(x)
        x = self.conv_layer_1_2(x)
        x = self.max_pool_layer_1(x)
        
        x = self.conv_layer_2(x)
        x = self.conv_layer_2_2(x)
        x = self.norm_layer_2(x)
        x = self.max_pool_layer_2(x)
        
        x = self.conv_layer_3(x)
        x = self.conv_layer_3_2(x)
        x = self.norm_layer_3(x)
        x = self.max_pool_layer_3(x)       
        
        x = self.mlp_input(x)
        x = self.hidden_layer_1(x)
        x = self.hidden_layer_2(x)
        x = self.output_layer(x)
        
        return x
