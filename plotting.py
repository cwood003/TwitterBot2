# plotting functions

import matplotlib.pyplot as plt


def plottingImages(imageArr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(imageArr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()