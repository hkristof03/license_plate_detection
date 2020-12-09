import os
from typing import List
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def list_images(path: str = None) -> List[str]:
    """Return a lists of image paths.

    :param path: Path to the image folders
    :return: List of image paths
    """
    path_image_dirs = os.path.join(os.getcwd(),
                                   'baza_slika') if not path else path
    image_dirs = list(filter(
        lambda x: os.path.isdir(os.path.join(path_image_dirs, x)),
        os.listdir(path_image_dirs)
    ))
    images = [glob.glob(os.path.join(path_image_dirs, img_dir, '*.jpg')) for
              img_dir in image_dirs]
    images = [img for image_dir in images for img in image_dir]

    return images


def read_df_labels(path: str = None) -> pd.DataFrame:
    """

    :param path:
    :return:
    """
    df = pd.read_csv(
        os.path.join(os.getcwd(), 'img_labels.csv') if not path else path
    )
    df['detected_labels'] = ''

    return df


def plot_images(
    img1: np.ndarray,
    img2: np.ndarray,
    title1: str = "",
    title2: str = ""
) -> None:
    """Plots two images with their corresponding titles.

    :param img1:
    :param img2:
    :param title1:
    :param title2:
    :return:
    """
    fig = plt.figure(figsize=[8, 8])
    ax1 = fig.add_subplot(121)
    ax1.imshow(img1, cmap='gray')
    ax1.set(xticks=[], yticks=[], title=title1)

    ax2 = fig.add_subplot(122)
    ax2.imshow(img2, cmap='gray')
    ax2.set(xticks=[], yticks=[], title=title2)
    plt.waitforbuttonpress(5.0)
    plt.close(fig)


def plot_images2(
    img1: np.ndarray,
    img2: np.ndarray,
    img3: np.ndarray,
    img4: np.ndarray,
    img5: np.ndarray,
    img6: np.ndarray,
    detected_text: str
) -> None:
    """
    """
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
    axs[0, 0].imshow(img1)
    axs[0, 0].set(xticks=[], yticks=[], title="Original")
    axs[0, 1].imshow(img2, cmap='gray')
    axs[0, 1].set(xticks=[], yticks=[], title="Gray")
    axs[1, 0].imshow(img3, cmap='gray')
    axs[1, 0].set(xticks=[], yticks=[], title="Blur")
    axs[1, 1].imshow(img4, cmap='gray')
    axs[1, 1].set(xticks=[], yticks=[], title="Canny Edge Detector")
    axs[2, 0].imshow(img5)
    axs[2, 0].set(xticks=[], yticks=[], title="Contours")
    axs[2, 1].imshow(img6, cmap='gray')
    axs[2, 1].set(xticks=[], yticks=[], title=detected_text)

    plt.waitforbuttonpress(30.0)
    plt.close(fig)