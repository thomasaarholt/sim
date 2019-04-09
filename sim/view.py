import numpy as np
from skimage.draw import circle
import matplotlib.pyplot as plt
from matplotlib import patches


def projected_cell(cell, probestep=0.15):
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


def projected_number(cell, probestep=0.15):
    cell_size = cell.cell.diagonal()[:2]
    xyshape = np.round(cell_size/probestep).astype(int)
    img = np.zeros(xyshape)

    for pos in cell.positions:
        x, y, _ = pos / probestep
        x, y = int(x), int(y)
        img[x, y] += 1
    return img


def plot_projected_number(cell, probestep=0.15):
    img = projected_number(cell, probestep)
    _plot_img(img)


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
    ymin, xmax = roi
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


def _plot_img(img):
    fig, ax = plt.subplots()
    img = ax.imshow(img)
    ax.set_xlabel('Y-dimension')
    ax.set_ylabel('X-dimension')


def plot_roi(cell, roi=(0.2, 0.8), probestep=0.15):
    img = projected_cell(cell, probestep)
    xy = get_bottom_left_corner(roi, img)
    height, width = _get_roi_height_width(roi, img)
    _plot_roi(img, xy, height, width)


def plot_inside_roi(cell, roi=(0.2, 0.8), probestep=0.15, nan=True):
    img = projected_cell(cell, probestep)
    lim1, lim2 = int(roi[0]), int(roi[1])
    img2 = img[lim1:lim2, lim1:lim2]
    if nan:
        img2[img2 == 0] = np.nan
    _plot_img(img2)


def annotate(ax):
    ax.annotate("In$_{Zn}$",
                xy=(87.5+2, 70+2), xycoords='data',
                xytext=(87+20, 120), textcoords='data',
                size=20, va="center", ha="center",
                bbox=dict(boxstyle="round4", fc="lightgrey"),
                arrowprops=dict(
                    fc="red"),
                )
    ax.annotate("V$_{Zn}$",
                xy=(69-2, 69.2-2), xycoords='data',
                xytext=(68.7-20, 20), textcoords='data',
                size=20, va="center", ha="center",
                bbox=dict(boxstyle="round4", fc="lightgrey"),
                arrowprops=dict(
                    fc="red"),
                )
