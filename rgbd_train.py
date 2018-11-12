#---------------------------------------------------
#---------------------------------------------------
# author: mark cass
# date: 11/08/2018
# source: https://github.com/mithi/semantic-segmentation/blob/master/main.py
# description: Jointly train RGB and Depth models using back propogation
#---------------------------------------------------
#---------------------------------------------------

import sys
import tensorflow as tf
ctblayers = tf.contrib.layers
import rgbd_predict


# train models
def train_model(tfsession, epochs, learning_rate, batch_op, dataset_init_op, model, model_input_vars):
    # add training loss and back prop to create training computation graph
    loss = calc_loss(model, model_input_vars[2], 16)
    train_op = backprop_loss(loss, learning_rate)
    # add predict op to create prediction computation graph
    predict_op = rgbd_predict.calc_prediction(model, model_input_vars[2], 16)
    # initalize session varaibles
    tfsession.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        print("------------------")
        print("Training Epoch: {:<3} of {:<3}".format(epoch + 1, epochs))
        print("------------------")
        # Initialize the data queue
        tfsession.run(dataset_init_op)

        losses, i = [], 0

        while True:
            i += 1
            try:
                batch_rgb_features, batch_depth_features, batch_labels = tfsession.run(batch_op)
            except tf.errors.OutOfRangeError:
                break
            feed = {model_input_vars[0]: batch_rgb_features,
                    model_input_vars[1]: batch_depth_features,
                    model_input_vars[2]: batch_labels}

            _, partial_loss = tfsession.run(train_op, feed_dict=feed)

            #print("---> iteration: ", i, " partial loss:", partial_loss)
            sys.stdout.write("---> Batch: {:<4}   Loss: {:0.4f}\r".format(i, partial_loss))
            sys.stdout.flush()
            losses.append(partial_loss)

        print(" ")
        training_loss = sum(losses) / float(len(losses))
        #all_training_losses.append(training_loss)
        print("   TRAINING LOSS : {:3.4}".format(training_loss))

        rgbd_predict.predict_model(tfsession, batch_op, dataset_init_op, predict_op, model_input_vars)


# compute the loss for each parameter and return the average
def calc_loss(model, labels, num_classes):
    # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
    logits = tf.reshape(model, (-1, num_classes), name="fcn_logits")
    labels_f = tf.to_float(labels)
    correct_label_reshaped = tf.reshape(labels_f, (-1, num_classes))

    # Calculate distance from actual labels using cross entropy
    cross_entropy = tf.losses.softmax_cross_entropy(correct_label_reshaped, logits)
    # Take mean for total loss
    loss = tf.reduce_mean(cross_entropy, name="fcn_loss")

    return loss


# back propogate the loss using gradient descent with adaptive gradient and
# momentum by way of ADAM optimizer
def backprop_loss(loss, learning_rate):
    # The model implements this operation to find the weights/parameters that
    # would yield correct pixel labels
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, name="fcn_train_op")

    return train_op, loss
