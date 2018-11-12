#!/bin/bash
#---------------------------------------------------
#---------------------------------------------------
# author: mark cass
# date: 11/08/2018
# description: prepare nyu depth data for classification by rgbd models
#---------------------------------------------------
#---------------------------------------------------

#python script to extract images from NYU depth data mathlab files
python nyu_to_png.py nyu_depth/nyu_depth_v2_labeled.mat 80 nyu_depth 0 16


#create directory for each type: rgb image, depth image, label image then copy
#associated images to directories.

image_rgb_dir="images_rgb"
image_d_dir="images_d"
labels_dir="labels"

if [[ ! -e  "$image_rgb_dir" ]]; then
    mkdir $image_rgb_dir
fi

if [[ ! -e  "$image_d_dir" ]]; then
    mkdir $image_d_dir
fi

if [[ ! -e  "$labels_dir" ]]; then
    mkdir $labels_dir
fi

echo "copying input RGB images to $image_rgb_dir"
image_list="$(find testing -name "*_colors.png")"
while read -r fname; do
    cp -f $fname $image_rgb_dir
done <<< "$image_list"

image_list="$(find training -name "*_colors.png")"
while read -r fname; do
    cp -f $fname $image_rgb_dir
done <<< "$image_list"


echo "copying input depth images to $image_d_dir"
image_list="$(find testing -name "*_depth.png")"
while read -r fname; do
    cp -f $fname $image_d_dir
done <<< "$image_list"

image_list="$(find training -name "*_depth.png")"
while read -r fname; do
    cp -f $fname $image_d_dir
done <<< "$image_list"


echo "copying ground truth label images to $labels_dir"
image_list="$(find testing -name "*_ground_truth.png")"
while read -r fname; do
    cp -f $fname $labels_dir
done <<< "$image_list"

image_list="$(find training -name "*_ground_truth.png")"
while read -r fname; do
    cp -f $fname $labels_dir
done <<< "$image_list"

