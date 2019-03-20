import numpy as np
from skimage.draw import circle
import matplotlib.pyplot as plt
from matplotlib import patches


def projected_cell(cell, roi, probestep=0.15):
    cell_size = cell.cell.diagonal()[:2]
    xyshape = np.round(cell_size/probestep).astype(int)
    img = np.zeros(xyshape)

    radius = 0.6  # atomic radius approx in Å
    radius = radius/probestep
    for pos, num in zip(cell.positions, cell.numbers):
        x, y, _ = pos / probestep
        rr, cc = circle(x, y, round(radius), shape=img.shape)
        img[rr, cc] += 1*num**2
    return img


def _get_roi_height_width(roi, img):
    xmin, xmax, ymin, ymax = roi
    height = (xmin - xmax) * img.shape[0]
    width = (ymax - ymin) * img.shape[1]
    return height, width

def get_pixel_number(cell, roi, probestep=0.15):
    img = projected_cell(cell, roi, probestep)
    height, width = _get_roi_height_width(roi, img)
    size = np.abs(height*width)
    return size


def get_bottom_left_corner(roi, img):
    _, xmax, ymin, _ = roi
    yx = np.array([ymin, xmax]) * img.T.shape
    return yx


def _plot_roi(img, xy, height, width):
    'roi is xmin, xmax, ymin, ymax like in prismatic'
    fig, ax = plt.subplots()
    img = ax.imshow(img)
    rect = patches.Rectangle(
        xy, width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.set_xlabel('Y-dimension')
    ax.set_ylabel('X-dimension')
    return fig


def plot_roi(cell, roi, probestep=0.15):
    img = projected_cell(cell, probestep)
    xy = get_bottom_left_corner(roi, img)
    height, width = _get_roi_height_width(roi, img)
    _plot_roi(img, xy, height, width)
