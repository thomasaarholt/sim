from sim.voronoi import integrate
import numpy as np
import matplotlib.pyplot as plt


def error(arr):
    return arr.std(ddof=1) / np.sqrt(len(arr))


def mean_and_std_error(arr):
    axis = []
    mean = []
    std = []

    for i in range(1, len(arr)):
        axis.append(i)
        mean.append(arr[:i+1].mean())
        std.append(error(arr[:i+1]))

    axis = np.asarray(axis)
    mean = np.asarray(mean)
    std = np.asarray(std)
    return axis, mean, std


def plot_standard_error_with_phonons(integrated_list, indium_index=27, vacancy_index=28, bulk_index=20):
    fig = plt.figure()

    axis, means, std = mean_and_std_error(integrated_list[indium_index])
    plt.errorbar(axis-0.1, means, std, fmt='', color='red',
                 ecolor='lightcoral', elinewidth=3, capsize=0, label='Indium')

    axis, means, std = mean_and_std_error(integrated_list[bulk_index])
    plt.errorbar(axis, means, std, fmt='', color='green',
                 ecolor='lightgreen', elinewidth=3, capsize=0, label='Bulk')

    axis, means, std = mean_and_std_error(integrated_list[vacancy_index])
    plt.errorbar(axis+0.1, means, std, fmt='', color='black',
                 ecolor='lightgray', elinewidth=3, capsize=0, label='Vacancy')
    plt.xlabel('Summed number of frozen phonons')
    plt.ylabel('Voronoi Intensity')
    plt.legend()
    return fig
