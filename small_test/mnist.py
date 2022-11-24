# Import IREE's TensorFlow Compiler and Runtime.
import iree.compiler.tf
import iree.runtime

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import time
import timeit
import functools

tf.random.set_seed(91)
np.random.seed(91)

# Define a function that calculate the running time
def time_count(func):
    # @functools.wraps(func)
    def runtime(*args, **kw):
        time_mean = []
        # Iteration time
        iter = 5
        for i in range(iter):
            start_time = time.time()
            func(*args, **kw)
            end_time = time.time()
        
        time_mean.append(end_time - start_time)

        print(func.__name__ + f" Runtime: {np.mean(time_mean):.10f}")
  
    return runtime

def plot_train_loss(losses, BATCH_SIZE):
    #@title Plot the training results
    import bottleneck as bn
    smoothed_losses = bn.move_mean(losses, BATCH_SIZE)
    x = np.arange(len(losses))

    plt.plot(x, smoothed_losses, linewidth=2, label='loss (moving average)')
    plt.scatter(x, losses, s=16, alpha=0.2, label='loss (per training step)')

    plt.ylim(0)
    plt.legend(frameon=True)
    plt.xlabel("training step")
    plt.ylabel("cross-entropy")
    plt.title("training loss")
    plt.show()

# Print version information for future notebook users to reference.
print("TensorFlow version: ", tf.__version__)
print("Numpy version: ", np.__version__)


# Keras datasets don't provide metadata.
NUM_CLASSES = 10
NUM_ROWS, NUM_COLS = 28, 28

path = '/home/shenghao/dataset/MNIST/mnist.npz'
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path)

# Reshape into grayscale images:
x_train = np.reshape(x_train, (-1, NUM_ROWS, NUM_COLS, 1))
x_test = np.reshape(x_test, (-1, NUM_ROWS, NUM_COLS, 1))

# Rescale uint8 pixel values into float32 values between 0 and 1:
x_train = x_train.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255

# IREE doesn't currently support int8 tensors, so we cast them to int32:
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

BATCH_SIZE = 32

class TrainableDNN(tf.Module):

  def __init__(self):
    super().__init__()

    # Create a Keras model to train.
    inputs = tf.keras.layers.Input((NUM_COLS, NUM_ROWS, 1))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dense(10)(x)
    outputs = tf.keras.layers.Softmax()(x)
    self.model = tf.keras.Model(inputs, outputs)

    # Create a loss function and optimizer to use during training.
    self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
    self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)
  
  @tf.function(input_signature=[
      tf.TensorSpec([BATCH_SIZE, NUM_ROWS, NUM_COLS, 1])  # inputs
  ])
  # @time_count
  def predict(self, inputs):
    return self.model(inputs, training=False)

  # We compile the entire training step by making it a method on the model.
  @tf.function(input_signature=[
      tf.TensorSpec([BATCH_SIZE, NUM_ROWS, NUM_COLS, 1]),  # inputs
      tf.TensorSpec([BATCH_SIZE], tf.int32)  # labels
  ])
  # @time_count
  def learn(self, inputs, labels):
    # Capture the gradients from forward prop...
    with tf.GradientTape() as tape:
      probs = self.model(inputs, training=True)
      loss = self.loss(labels, probs)

    # ...and use them to update the model's weights.
    variables = self.model.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))
    return loss

exported_names = ["predict", "learn"]
# @param [ "vmvx (CPU)", "llvm-cpu (CPU)", 
# "vulkan-spirv (GPU/SwiftShader – requires additional drivers) " ]
backend_choice = "llvm-cpu (CPU)"
backend_choice = backend_choice.split(' ')[0]
# Compile the TrainableDNN module
# Note: extra flags are needed to i64 demotion, see
# https://github.com/iree-org/iree/issues/8644
vm_flatbuffer = iree.compiler.tf.compile_module(TrainableDNN(),
            target_backends=[backend_choice],
            exported_names=exported_names,
            extra_args=["--iree-mhlo-demote-i64-to-i32=false",
                        "--iree-flow-demote-i64-to-i32"])
compiled_model = iree.runtime.load_vm_flatbuffer(
                vm_flatbuffer, backend=backend_choice)

# noop model
model_noop = TrainableDNN()
print("Original Training...")
# model_noop.predict(x_train[:BATCH_SIZE])

# Predicting
time_mean = []
for i in range(50):
    start_time = time.time()
    model_noop.predict(x_train[:BATCH_SIZE])
    end_time = time.time()    
    time_mean.append(end_time - start_time)
print("Predict Time (Noop): {}".format((np.mean(time_mean))))

# Training
time_mean = []
for i in range(50):
    start_time = time.time()
    model_noop.learn(x_train[:BATCH_SIZE], y_train[:BATCH_SIZE])
    end_time = time.time()    
    time_mean.append(end_time - start_time)
model_noop.learn(x_train[:BATCH_SIZE], y_train[:BATCH_SIZE])
print("Training Time (Noop): {}".format((np.mean(time_mean))))

# Compiled model
#@title Benchmark inference and training
print("Compiled model Training...")
time_mean = []
for i in range(50):
    start_time = time.time()
    compiled_model.predict(x_train[:BATCH_SIZE])
    end_time = time.time()    
    time_mean.append(end_time - start_time)
print("Predict Time (Compiled): {}".format((np.mean(time_mean))))

time_mean = []
for i in range(50):
    start_time = time.time()
    compiled_model.learn(x_train[:BATCH_SIZE], y_train[:BATCH_SIZE])
    end_time = time.time()    
    time_mean.append(end_time - start_time)
print("Training Time (Compiled): {}".format((np.mean(time_mean))))

# Run the core training loop.
losses = []

step = 0
max_steps = x_train.shape[0] // BATCH_SIZE

for batch_start in range(0, x_train.shape[0], BATCH_SIZE):
    if batch_start + BATCH_SIZE > x_train.shape[0]:
        continue

    inputs = x_train[batch_start:batch_start + BATCH_SIZE]
    labels = y_train[batch_start:batch_start + BATCH_SIZE]

    loss = compiled_model.learn(inputs, labels).to_host()
    losses.append(loss)

    step += 1
    print(f"\rStep {step:4d}/{max_steps}: loss = {loss:.4f}", end="")
print("Done")

# plot the training loss (Optional)
#  plot_train_loss(losses, BATCH_SIZE)
