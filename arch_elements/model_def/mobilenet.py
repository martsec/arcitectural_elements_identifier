import tensorflow as tf
import tensorflow_hub as hub

class Mobilenet(tf.keras.Model):

    def __init__(self, pixels:int, num_classes:int, do_fine_tuning:bool=False):
        super(Mobilenet, self).__init__()
        model_base_name = "mobilenet_v2_100_"
        module_selection = model_base_name + str(pixels)
        module_url ="https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(module_selection)
        #image_size = (pixels, pixels)

        #self.input1 = tf.keras.layers.InputLayer(input_shape=image_size + (3,))
        self.pretrained = hub.KerasLayer(module_url, trainable=do_fine_tuning)
        self.dropout1 = tf.keras.layers.Dropout(rate=0.2)
        self.dense1 = tf.keras.layers.Dense(num_classes, kernel_regularizer=tf.keras.regularizers.l2(0.0001))

    def call(self, inputs):
        x1 = self.pretrained(inputs)
        x2 = self.dropout1(x1)
        return self.dense1(x2)
