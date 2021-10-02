import os
import pandas as pd
import numpy as np
import random
import cv2
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


def get_image_with_box(img: np.ndarray, img_name: str, df: pd.DataFrame) -> np.ndarray:
    """
    This function will:
    1. get all bounding boxes that relate to the image for the DataFrame
    2. concatenate them with imgaug package to a BoundingBoxesOnImage object
    3. draw the Bounding boxes onto the image
    :param img: image as np array
    :param img_name: filename to locate the bounding boxes in the df
    :param df: DataFrame that holds the information about all bounding boxes
    :return: image with bounding boxes drawn onto it
    """
    # create bounding box with imgaug
    img_boxs = df[df.filename == img_name]

    bbs = list()
    for _, row in img_boxs.iterrows():
        x1 = row.xmin
        y1 = row.ymin
        x2 = row.xmax
        y2 = row.ymax
        bbs.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))

    # convert single bounding boxes to BOundingBoxOnImage instance to draw it on the picture
    bbs = BoundingBoxesOnImage(bbs, img.shape[:-1])

    # draw image
    img = bbs.draw_on_image(img)

    return img


if __name__ == '__main__':
    # define image folder
    folder = 'train'

    # load annotations of specified folder
    df = pd.read_csv(os.path.join('..', 'annotations', f'{folder}.csv'))

    # pick a random image from the table
    rand_idx = random.sample(range(df.shape[0]), 1)[0]
    img_name = df.loc[rand_idx, 'filename']

    # load random image from defined folder
    img = cv2.imread(os.path.join('..', 'images', folder, img_name))

    # draw boxes onto the image
    img = get_image_with_box(img, img_name, df)
    # show image
    cv2.imshow(img_name, img)

    cv2.waitKey(0)

