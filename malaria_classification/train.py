import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import numpy as np
import time
from model import Model


def test(model, test_data, loss_function):
    # test over complete test data

    test_accuracy_aggregator = []
    test_loss_aggregator = []

    for (input, target) in test_data:
        prediction = model(input)
        sample_test_loss = loss_function(target, prediction)
        sample_test_accuracy = np.argmax(
            target, axis=1) == np.argmax(prediction, axis=1)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

    test_loss = np.mean(test_loss_aggregator)
    test_accuracy = np.mean(test_accuracy_aggregator)

    return test_loss, test_accuracy


def investigate(images, labels):

    # rescaling images to a uniform size
    # not keeping aspect ration
    # images contain cropped photos of cells with black paddings
    image = tf.image.resize(images, (128, 128))

    # normalization
    image /= 255  # normalize between 0 and 1

    # one_hot encoding labels
    label = tf.one_hot(labels, 2)
    print(label)
    return image, label


def train_step(model, input, target, loss_function, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(input)
        loss = loss_function(target, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# hyperparameters
num_epochs = 25
learning_rate = 0.00008
running_average_factor = 0.95
BATCH_SIZE = 12

tf.keras.backend.clear_session()

# since there is not test split, we take 20% of the train data as our test split
# distribution is arbitrary
train_dataset, train_info = tfds.load(
    'malaria', split='train[:80%]', shuffle_files=True, as_supervised=True, with_info=True)
test_dataset, test_info = tfds.load(
    'malaria', split='train[:20%]', shuffle_files=True, as_supervised=True, with_info=True)


train_dataset = train_dataset.shuffle(buffer_size=10)
train_dataset = train_dataset.map(investigate, num_parallel_calls=4)
# caching does not work in my case (to few memory)
#train_dataset = train_dataset.cache()
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(1)


test_dataset = test_dataset.shuffle(buffer_size=10)
test_dataset = test_dataset.map(investigate, num_parallel_calls=4)
#test_dataset = test_dataset.cache()
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.prefetch(1)

# init lists for visualization
train_losses = []
test_losses = []
test_accuracies = []


# init model
model = Model()

# init loss
cross_entropy_loss = tf.keras.losses.BinaryCrossentropy()

# init optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate)


start = time.time()

print("")
print("Number of Epochs: " + str(num_epochs))


# train for number of epochs
for epoch in range(num_epochs):
    print('Epoch:___' + str(epoch))

    train_dataset = train_dataset.shuffle(buffer_size=128)
    test_dataset = test_dataset.shuffle(buffer_size=128)

    running_average = 0
    for (data, target) in train_dataset:
        train_loss = train_step(model, data, target,
                                cross_entropy_loss, optimizer)
        running_average = running_average_factor * running_average + \
            (1 - running_average_factor) * train_loss
    train_losses.append(running_average)

    test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    if epoch == 0:
        end = time.time()
        end = end - start
        end = end * num_epochs - 1
        print("")
        print("Estimated training time: " + str(int(end / 60)) + " mins.")
        print("")


end = time.time()
print("")
print("Time elapsed: " + str(int((end - start)/60)) + " mins.")
print("")

# plot loss and accuracy over epochs
plt.figure()
line1, = plt.plot(train_losses)
line2, = plt.plot(test_losses)
plt.xlabel("Training steps")
plt.ylabel("Loss")
plt.legend((line1, line2), ("training", "test"))
plt.show()

plt.figure()
line1, = plt.plot(test_accuracies)
plt.xlabel("Training steps")
plt.ylabel("Accuracy")
plt.show()
