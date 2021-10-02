import os
import pandas as pd
import random
import cv2
from matplotlib import pyplot as plt

from scripts.show_image_box import get_image_with_box


def create_comparison_plot(aug_df: pd.DataFrame, aug_img_name: str, df: pd.DataFrame, img_name: str, folder: str):
    """
    This function will:
    1. load the bounding boxes of the augmented images and draw them on the aug_image
    2. do the same with original image
    3. make them side by side on comparison subpot
    :param aug_df: DataFrame that holds all bounding boxes for augmented image
    :param aug_img_name: str object to locate bounding boxes in DataFrame
    :param df: DataFrame that holds all bounding boxes for image
    :param img_name: str object to locate bounding boxes in DataFrame
    :param folder: str object that defines the folders where to find the images
    """
    aug_folder = os.path.join('..', 'images', f'{folder}_aug')
    # load augmented image
    aug_img = cv2.imread(os.path.join('..', 'images', aug_folder, aug_img_name))
    aug_img = get_image_with_box(aug_img, aug_img_name, aug_df)

    aug_img = aug_img[:, :, [2, 1, 0]]

    org_folder = os.path.join('..', 'images', folder)
    # load augmented image
    org_img = cv2.imread(os.path.join('..', 'images', org_folder, img_name))
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
    # define folder
    folder = 'test'
    # load df_aug to pick a random image from it
    aug_df = pd.read_csv(os.path.join('..', 'annotations', f'{folder}_aug.csv'))

    # pick a random image from the table
    rand_idx = random.sample(range(aug_df.shape[0]), 1)[0]
    img_aug_name = aug_df.loc[rand_idx, 'filename']

    # get original image name
    img_name = [img for img in os.listdir(os.path.join('..', 'images', folder)) if img in img_aug_name]
    # check if image exist
    if img_name:
        img_name = img_name[0]
    else:
        raise FileNotFoundError()

    # load df with original image bounding box information
    df = pd.read_csv(os.path.join('..', 'annotations', f'{folder}.csv'))

    create_comparison_plot(aug_df, img_aug_name, df, img_name, folder)

    plt.show()

