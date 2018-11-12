**RGB-D Semantic Segmentation**

This project implements a neural network solution for object detection and localization
of images augmented with depth of field information.

#
**Dataset**


NYU Depth Dataset V2 - https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

NYU Depth is a labelled dataset of indoor scene images. Each scene is comprised of
two images. The first is an RGB image of the scene and is visually similar to a
photograph. The second image is RGB, but unlike a photgraph, each pixels is colored
to represent the distance between the camera and the pixel. This image is referred
to as the depth image. The dataset also provides a ground truth image where each
pixel is specifically colored so as to correspond to an object class.

The dataset also includes unprocessed version of the above mention which were not
used in this project.

    * Images in dataset - 1449
    * Image resolution  - 640x480
    * Image format      - 24 bit RGB

For this project the images were preprocessed in the following way:

    * Each image was extracted from the Math Works source archive. Seperate PNG
      images were created for RGB image, Depth image, and ground truth label image.
    * The depth image was converted from 24 bit RGB to 16 bit gray scale.
    * The label image was indexed to a pallet of 4 colors. One color for each object
      class.
    * RGB, Depth, and label images were resized to 300x300.
    * RGB, Depth and Label images were randomly cropped.
    * RGB image brightness, constrast, and saturation were randomly perturbed.


#

**Neural Network Platform**

Tensorflow - https://www.tensorflow.org/

Tensorflow is a machine learning library supporting a wide array of algorithms including
neural networks. The API is published in multiple languages.

This project uses the Python API version 1.12.



#

**Neural Network Architecture**

The project employes multi-task learning to jointly train two convolutional
neural networks. The first network, CNN_RGB, is fed rgb images. The second network,
CNN_D is fed depth images. The output of both networks is then fed to a final network,
CNN_RGBD, where the classification occurs.

#### CNN_RGB ARCHITECTURE
| LAYER | SHAPE |
|--------------- | ------------------ |
| RGB FEATURES.      | input shape = (10, 300, 300, 3) |
| 2D CONVOLUTION.    | ouput shape = (10, 296, 296, 128) |
| NON-LINEAR RELU.   | ouput shape = (10, 296, 296, 128) |
| 2D CONVOLUTION.    | ouput shape = (10, 294, 294, 128) |
| NON-LINEAR RELU.   | ouput shape = (10, 294, 294, 128) |
| 2D MAX POOL.       | ouput shape = (10, 292, 292, 128) |
| 2D DE-CONVOLUTION. | ouput shape = (10, 300, 300, 128) |
| FULLY CONNECTED.   | ouput shape = (10, 300, 300, 128) |
| NON-LINEAR RELU.   | ouput shape = (10, 300, 300, 128) |
| FULLY CONNECTED.   | ouput shape = (10, 300, 300, 16) |

#### CNN_D ARCHITECTURE
| LAYER | SHAPE |
|--------------- | ------------------ |
| DEPTH FEATURES.    | input shape = (10, 300, 300, 1) |
| 2D CONVOLUTION.    | ouput shape = (10, 296, 296, 128) |
| NON-LINEAR RELU.   | ouput shape = (10, 296, 296, 128) |
| 2D CONVOLUTION.    | ouput shape = (10, 294, 294, 128) |
| NON-LINEAR RELU.   | ouput shape = (10, 294, 294, 128) |
| 2D MAX POOL.       | ouput shape = (10, 292, 292, 128) |
| 2D DE-CONVOLUTION. | ouput shape = (10, 300, 300, 128) |
| FULLY CONNECTED.   | ouput shape = (10, 300, 300, 128) |
| NON-LINEAR RELU.   | ouput shape = (10, 300, 300, 128) |
| FULLY CONNECTED.   | ouput shape = (10, 300, 300, 16) |


#### CNN_RGBD MULTICLASS ARCHITECTURE
| LAYER | SHAPE |
|--------------- | ------------------ |
| JOIN NETS.         | ouput shape = (10, 300, 600, 16) |
| FULLY CONNECTED.   | ouput shape = (10, 300, 600, 128) |
| NON-LINEAR RELU.   | ouput shape = (10, 300, 600, 128) |
| FULLY CONNECTED.   | ouput shape = (10, 300, 600, 8) |


#

**Training**

Training employees multi-task learning to jointly train the networks in a single
session. The following training parameters were used.

| PARAMETER | VALUE |
|---------------    | ------------------ |
| Epochs            | 10            |
| Batch size        | 10            |
| Learning rate     | 1e-5          |
| Optimizer         | Adam          |

The time to train the network, including accuracy predictions each epoch is on the order
of 20 minutes using an NVidia 1080 ti. The networks and data consumed 100 megabytes
of GPU memory. During training the GPU continuiously consumed and average of 230 watts
of power.

#

**Results**

Results were generated for training CNN_RGB and CNN_RGBD. Below are the results for
CNN_RGBD and CNN_RGB.

#### Observations
* One interesting finding is that the manner in which the CNN_RGB and CNN_D output are
combined is important. When combined to form a 4 component RGBD image, with the same
width and height as the source image, the resulting accuracy was similar to that of
CNN_RGB alone. When combined to form a image that was twice as wide as the source
and with as many channels, the accuracy improved. This makes sense some sense when
compared to the human vision system. I think having both images side-by-side, and
especially with one being depth of field, caused the network to see the scene in
a binary fashion.
* This network is a proof of concept and is most certainly over-fitting. The next task
would be to reduce the network using a random grid search of the.
* The selected architecture is based on research of current known solutions. The specific
choices of layers and sizes were chosen to prefer a simpler and easier to implement solution.
It would be a better idea to implement state-of-the-art algorithms.
* I'm quite happy with how little effort was needed to engineer the features. The
depth and label images were resized and cropped randomly, while the RGB images had
additional generic changes to augment the data. All of these modifications were able
to be performed using the Tensorflow API and thus required minimal effort to implement.


#### CNN_RGBD
| Epoch | LOSS | ACCURACY |
| ------------ | ---------- | ------------- |
|1   of 10 | 6.704 | 93.95% |
|2   of 10 | 2.638 | 94.11% |
|3   of 10 | 2.306 | 94.19% |
|4   of 10 | 2.135 | 94.21% |
|5   of 10 | 2.095 | 94.24% |
|6   of 10 | 2.096 | 94.23% |
|7   of 10 | 2.063 | 94.22% |
|8   of 10 | 2.052 | 94.19% |
|9   of 10 | 2.045 | 94.15% |
|10  of 10 | 2.067 | 94.06% |

#### CNN_RGB
| Epoch | LOSS | ACCURACY |
| ------------ | ---------- | ------------- |
|1   of 10 | 4.416 | 93.04% |
|2   of 10 | 2.010 | 93.42% |
|3   of 10 | 1.794 | 93.57% |
|4   of 10 | 1.670 | 93.70% |
|5   of 10 | 1.159 | 93.78% |
|6   of 10 | 1.529 | 93.83% |
|7   of 10 | 1.412 | 93.92% |
|8   of 10 | 1.465 | 93.96% |
|9   of 10 | 1.448 | 93.98% |
|10  of 10 | 1.441 | 94.00% |