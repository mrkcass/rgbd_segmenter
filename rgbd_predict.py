#---------------------------------------------------
#---------------------------------------------------
# author: mark cass
# date: 11/08/2018
# description: computational graph predict nodes (operations)
#---------------------------------------------------
#---------------------------------------------------

import sys
import tensorflow as tf
ctblayers = tf.contrib.layers


# run the forward pass on the dataset and output the overall accuracy.
def predict_model(tfsession, batch_op, dataset_init_op, predict_op, model_input_vars):
    accuracies, i = [], 0

    tfsession.run(dataset_init_op)

    while True:
        i += 1
        try:
            batch_rgb_features, batch_depth_features, batch_labels = tfsession.run(batch_op)
        except tf.errors.OutOfRangeError:
            break
        feed = {model_input_vars[0]: batch_rgb_features,
                model_input_vars[1]: batch_depth_features,
                model_input_vars[2]: batch_labels}

        _, acc_all = tfsession.run(predict_op, feed_dict=feed)

        sys.stdout.write("---> batch: {:<4}   Accuracy: {:3.4f}\r".format(i, acc_all))
        sys.stdout.flush()
        accuracies.append(acc_all)

    print(" ")
    model_acc = sum(accuracies) / float(len(accuracies))
    print("   MODEL ACCURACY: {:3.4}%".format(model_acc * 100.0))


# flatten the labels and model ouptut features then count the number of correct
# pixels and return the average number right as a ratio of correct pixels/total pixels
def calc_prediction(model, labels, num_classes):
    # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
    logits = tf.reshape(model, (-1, num_classes), name="fcn_logits")
    labels_f = tf.to_float(labels)
    correct_label_reshaped = tf.reshape(labels_f, (-1, num_classes))

    predicted = tf.nn.softmax(logits)
    # compare each pixel with label and assign 1 to correct pixels
    correct_prediction = tf.equal(tf.round(predicted), tf.round(correct_label_reshaped))
    # count and average correct pixels for each class
    class_accuracy = tf.reduce_mean(tf.to_float(correct_prediction), axis=1)
    # count and average correct classes
    overall_accuracy = tf.reduce_mean(tf.to_float(class_accuracy), axis=0)

    return class_accuracy, overall_accuracy
