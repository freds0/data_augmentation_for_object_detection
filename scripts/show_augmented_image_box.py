import os
import argparse
import pandas as pd
import random
import cv2
from matplotlib import pyplot as plt
import re
import sys
sys.path.append('./show_image_box')

from show_image_box import get_image_with_box


def create_comparison_plot(aug_df: pd.DataFrame, aug_img_folder: str, aug_img_name: str, df: pd.DataFrame, orig_img_folder: str, img_name: str):
    """
    This function will:
    1. load the bounding boxes of the augmented images and draw them on the aug_image
    2. do the same with original image
    3. make them side by side on comparison subpot
    :param aug_df: DataFrame that holds all bounding boxes for augmented image
    :param aug_img_folder: str object that defines the folders where to find the augmented images
    :param aug_img_name: str object to locate bounding boxes in DataFrame
    :param df: DataFrame that holds all bounding boxes for image
    :param orig_img_folder: str object that defines the folders where to find the original images
    :param img_name: str object to locate bounding boxes in DataFrame
    """
    # load augmented image
    aug_img = cv2.imread(os.path.join(aug_img_folder, aug_img_name))
    aug_img = get_image_with_box(aug_img, aug_img_name, aug_df)

    aug_img = aug_img[:, :, [2, 1, 0]]

    # load original image
    org_img = cv2.imread(os.path.join(orig_img_folder, img_name))
    org_img = get_image_with_box(org_img, img_name, df)

    org_img = org_img[:, :, [2, 1, 0]]

    fig, axs = plt.subplots(1, 2, figsize=[20, 10])

    ax = axs[0]
    ax.imshow(org_img)
    ax.set_title('original image')
    ax.axis('off')

    ax = axs[1]
    ax.imshow(aug_img)
    ax.set_title('augmented image')
    ax.axis('off')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./')
    parser.add_argument('--orig_folder', default='images/test')
    parser.add_argument('--aug_folder', default='images/test_aug')
    parser.add_argument('--orig_csv', default='annotations/test.csv')
    parser.add_argument('--aug_csv', default='annotations/test_aug.csv')
    args = parser.parse_args()

    # define input folder
    orig_folder = os.path.join(args.base_dir, args.orig_folder)
    # define and create output_folder
    aug_folder = os.path.join(args.base_dir, args.aug_folder)
    # load df_aug to pick a random image from it
    aug_df = pd.read_csv(os.path.join(args.base_dir, args.aug_csv))

    # pick a random image from the table
    rand_idx = random.sample(range(aug_df.shape[0]), 1)[0]
    img_aug_name = aug_df.loc[rand_idx, 'filename']

    img_orig_name = re.sub('_data_aug[0-9].jpg', '.jpg', img_aug_name)
    # get original image name
    img_name = [img for img in os.listdir(orig_folder) if img in img_orig_name]
    # check if image exist
    if img_name:
        img_name = img_name[0]
    else:
        raise FileNotFoundError()

    # load df with original image bounding box information
    df = pd.read_csv(os.path.join(args.base_dir, args.orig_csv))

    create_comparison_plot(aug_df, aug_folder, img_aug_name, df, orig_folder, img_name)

    plt.show()

