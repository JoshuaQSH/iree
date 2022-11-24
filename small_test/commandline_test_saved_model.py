import tensorflow.compat.v2 as tf

loaded_model = tf.saved_model.load('/home/shenghao/iree/saved_model/mobilenet_v2/')
print(list(loaded_model.signatures.keys()))
