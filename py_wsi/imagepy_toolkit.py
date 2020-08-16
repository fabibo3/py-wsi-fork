
'''

An image-displaying function from imagepy.toolkit on GitHub:

https://github.com/ysbecca/imagepy-toolkit

Author: @ysbecca, Fabian Bongratz


'''

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def show_images(images, per_row, per_column):
    ''' Displays up to per_row*per_column images with per_row images per row, per_column images per column.
    '''
    fig = plt.figure(figsize=(25, 25))
    data = images[:(per_row*per_column)]

    for i, image in enumerate(data):
        plt.subplot(per_column, per_row, i+1)
        plt.imshow(image)
        plt.axis("off")
    
    plt.show()


def show_labeled_patches(images, clss):
    fig = plt.figure(figsize=(20, 10))
    data = images[:50]
    labels = clss[:50]

    for i, image in enumerate(data):
        plt.subplot(5, 10, i+1)
        plt.imshow(image)
        plt.title(str(labels[i]))
        plt.axis("off")

    plt.show()


def show_images_and_gt(images, coords, pixel_classes, seg_maps):
    """
    Show a whole WSI image together with its gt annotation, partioned into
    patches
    """
    max_coords = np.max(coords, axis=0)
    min_coords = np.min(coords, axis=0)
    per_row = max_coords[0] - min_coords[0] + 1
    per_column = max_coords[1] - min_coords[1] + 1
    fig = plt.figure(figsize=(per_row, per_column))
    gs1 = gridspec.GridSpec(per_column, per_row) 
    gs1.update(wspace=0.05, hspace=0.05) # set the spacing between axes. 
    for i, image in enumerate(images):
        ax = plt.subplot(gs1[i])
        ax.imshow(image)
        ax.imshow(seg_maps[i].squeeze(), alpha=0.25, cmap='Greens',
                  vmin=np.min(pixel_classes), vmax=np.max(pixel_classes))
        ax.axis("off")
        ax.set_aspect('equal')

    print(i)
    plt.show()

def show_patch_and_gt(images, seg_maps, pixel_classes, per_row, per_column):
    """
    Show WSI patch together with its gt annotation
    """
    fig = plt.figure(figsize=(25, 25))
    data = images[:(per_row*per_column)]

    for i, image in enumerate(data):
        plt.subplot(per_column, per_row, i+1)
        plt.imshow(image)
        plt.imshow(seg_maps[i].squeeze(), alpha=0.25, cmap='Greens',
                  vmin=np.min(pixel_classes), vmax=np.max(pixel_classes))
        plt.axis("off")
    
    plt.show()
