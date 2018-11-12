#---------------------------------------------------
#---------------------------------------------------
# author: mark cass
# date: 11/08/2018
# description: model training and inference entry point
#---------------------------------------------------
#---------------------------------------------------

import tensorflow as tf
import numpy as np
import os

import _solarized as palette
from rgbd_data import DataLoader as data_loader
import rgbd_model
import rgbd_train


def main():
    # training hyper parameters
    batch_size = 10
    learning_rate = 1e-6
    feature_image_width = 300
    feature_image_height = 300
    # depth in order: rgb images, depth_images, label_images
    image_byte_depths = [3, 1, 3]
    label_classes = 16
    augmentation_seed = 42

    print("Tensorflow version = {}".format(tf.VERSION))

    # generate lists of features (rgb and depth images) and labels (indexed color image masks)
    rgb_image_path = "nyu_depth/images_rgb"
    depth_image_path = "nyu_depth/images_d"
    label_path = "nyu_depth/labels"
    rgb_image_list = [os.path.join(rgb_image_path, x) for x in os.listdir(rgb_image_path) if x.endswith('.png')]
    depth_image_list = [os.path.join(depth_image_path, x) for x in os.listdir(depth_image_path) if x.endswith('.png')]
    label_list = [os.path.join(label_path, x) for x in os.listdir(label_path) if x.endswith('.png')]
    # rgb colors associated with label index color
    np_palette = np.array(palette.colors)

    print("loading dataset from nyu_depth")

    #load the dataset images, scale to 300 x 300 pixels, add random cropping
    dataset = data_loader(rgb_image_paths=rgb_image_list,
                          depth_image_paths=depth_image_list,
                          mask_paths=label_list,
                          image_size=[feature_image_width, feature_image_height],
                          crop_percent=0.8,
                          channels=image_byte_depths,
                          palette=np_palette,
                          seed=augmentation_seed)
    # configure the dataset for batch, shuffle and data augmentation
    batch_op, dataset_init_op = dataset.data_batch(augment=True,
                                                   shuffle=True,
                                                   one_hot_encode=True,
                                                   batch_size=batch_size,
                                                   num_threads=4,
                                                   buffer=60)
    print("   dataset load complete")

    # build the computation graph (model) and get the variables where
    #features and labels should be fed
    model, model_input_vars = rgbd_model.RGBD_SEGMENTER(batch_size,
                                                        feature_image_width,
                                                        feature_image_height,
                                                        image_byte_depths,
                                                        label_classes)

    #create the session and traing the model
    with tf.Session() as sess:
        rgbd_train.train_model(sess,
                               batch_size,
                               learning_rate,
                               batch_op, dataset_init_op,
                               model,
                               model_input_vars)


#---------------------------------------------------
#---------------------------------------------------
#call the entry point
main()
