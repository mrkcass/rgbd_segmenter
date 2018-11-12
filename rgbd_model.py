#---------------------------------------------------
#---------------------------------------------------
# author: mark cass
# date: 11/08/2018
# description: Neural network models for semantically segmenting RGB images
#              with depth. Network is composed of two separate networks that
#              are jointly trained via multi-task learning.
#---------------------------------------------------
#---------------------------------------------------
import re
import tensorflow as tf
ctblayers = tf.contrib.layers


# build the two segmenter models. one for RGB data and one for depth data
def RGBD_SEGMENTER(batch_size, feature_width, feature_height, channels, label_class_count):
    rgb_features = tf.placeholder(tf.uint8, shape=(batch_size, feature_width, feature_height, channels[0]))
    if channels[1] < 3:
        depth_features = tf.placeholder(tf.uint16, shape=(batch_size, feature_width, feature_height, 1))
    else:
        depth_features = tf.placeholder(tf.uint8, shape=(batch_size, feature_width, feature_height, channels[1]))
    labels = tf.placeholder(tf.uint8, shape=(batch_size, feature_width, feature_height, label_class_count))

    model_rgb = CNN_RGB(rgb_features)
    model_d = CNN_D(depth_features)

    #create a combined image with same dimensions and depth 2
    #join_axis = 3
    #final_fc_width = 16
    # create a combined image with width, 2x height and same depth
    join_axis = 2
    final_fc_width = 8
    print("CNN_RGBD MUTLI-TASK ARCHITECTURE ---------------------")
    joined = join_nets(model_rgb, model_d, join_axis)
    fully_connected_1 = fully_connected(joined, 128)
    relu = relu_non_linerarity(fully_connected_1)
    model = fully_connected(relu, final_fc_width)
    print("------------------------------------------")

    return model, (rgb_features, depth_features, labels)


# convolutional neural network architecture for classifying rgb features
def CNN_RGB(features):
    print("CNN_RGB ARCHITECTURE ---------------------")
    features_f = tf.to_float(features)
    print("   RGB FEATURES.      input shape = {}".format(parse_shape(features)))
    cnn_1 = convolution(features_f, 5, 5, 1, 128)
    relu_1 = relu_non_linerarity(cnn_1)
    cnn_2 = convolution(relu_1, 3, 3, 1, 128)
    relu_2 = relu_non_linerarity(cnn_2)
    max_pool = max_pooling(relu_2, 3, 3, 1)
    deconv = deconvolution(max_pool, 9, 9, 1, 128)
    fully_connected_1 = fully_connected(deconv, 128)
    relu_3 = relu_non_linerarity(fully_connected_1)
    fully_connected_2 = fully_connected(relu_3, 16)
    print("------------------------------------------")
    return fully_connected_2


# convolutional neural network architecture for classifying depth features
def CNN_D(features):
    print("CNN_D ARCHITECTURE ---------------------")
    features_f = tf.to_float(features)
    print("   DEPTH FEATURES.    input shape = {}".format(parse_shape(features)))
    cnn_1 = convolution(features_f, 5, 5, 1, 128)
    relu_1 = relu_non_linerarity(cnn_1)
    cnn_2 = convolution(relu_1, 3, 3, 1, 128)
    relu_2 = relu_non_linerarity(cnn_2)
    max_pool = max_pooling(relu_2, 3, 3, 1)
    deconv = deconvolution(max_pool, 9, 9, 1, 128)
    fully_connected_1 = fully_connected(deconv, 128)
    relu_3 = relu_non_linerarity(fully_connected_1)
    fully_connected_2 = fully_connected(relu_3, 16)
    print("------------------------------------------")
    return fully_connected_2


#2d convolution
def convolution(features, shape_x, shape_y, stride, num_filters):
    conv_filter_shape = (shape_x, shape_y)
    conv_stride = (stride, stride)
    conv_2d = tf.layers.conv2d(features,
                               num_filters,
                               conv_filter_shape,
                               strides=conv_stride,
                               activation=None,
                               padding='VALID')
    print("   2D CONVOLUTION.    ouput shape = {}".format(parse_shape(conv_2d)))
    return conv_2d


# de-convolution to upsample
def deconvolution(features, shape_x, shape_y, stride, num_filters):
    conv_filter_shape = (shape_x, shape_y)
    conv_stride = (stride, stride)
    deconv_2d = tf.layers.conv2d_transpose(features,
                                           num_filters,
                                           conv_filter_shape,
                                           strides=conv_stride,
                                           activation=None,
                                           padding='VALID')
    print("   2D DE-CONVOLUTION. ouput shape = {}".format(parse_shape(deconv_2d)))
    return deconv_2d


# Element-wise perceptron with RELU activation function
def relu_non_linerarity(features):
    relu = tf.nn.relu(features)
    print("   NON-LINEAR RELU.   ouput shape = {}".format(parse_shape(relu)))
    return relu


# pool the maximum values within a window
def max_pooling(features, shape_x, shape_y, stride):
    pool_filter_shape = (shape_x, shape_y)
    pool_stride = [stride, stride]
    pool = ctblayers.max_pool2d(features,
                                pool_filter_shape,
                                stride=pool_stride,
                                padding='VALID'
                                )
    print("   2D MAX POOL.       ouput shape = {}".format(parse_shape(pool)))
    return pool


# multilayer perceptron with linear activation in the output layer
def fully_connected(features, output_width):
    fc = ctblayers.fully_connected(features,
                                   output_width,
                                   activation_fn=None)
    print("   FULLY CONNECTED.   ouput shape = {}".format(parse_shape(fc)))
    return fc


# multilayer perceptron with linear activation in the output layer
def join_nets(features_rgb, features_depth, axis):
    joined = tf.concat([features_rgb, features_depth], axis)
    print("   JOIN NETS.         ouput shape = {}".format(parse_shape(joined)))
    return joined


# parse the shape from the string printed to the display.
def parse_shape(features):
    raw = "%s" % features
    search_result = re.search(r'\([0-9 ,]+\)', raw, re.M | re.I)
    return search_result.group()
