import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, mu, sigma, normalise=False):
    'Create gaussian around mu. Normalise will make area = 1 but height != 1.'
    gauss = np.exp((-(x-mu)**2)/(2*sigma**2))
    if normalise:
        gauss = gauss * 1/(sigma*np.sqrt(2*np.pi))  # normalise
    return gauss


def get_series_defocus_and_weight(ZLP=0.9, Voltage=3e5, Chromatic=1.6e-3):
    'Calculates sigma of probe defocus distribution due to Cc, in Å'
    E0, dE, Cc = sp.symbols('E0, dE, C_c, ')
    # *2*sp.sqrt(2*sp.log(2))# for fwhm isntead of sigma
    deltaZ_sigma = Cc*dE/E0

    f = sp.lambdify([Cc, dE, E0], deltaZ_sigma)

    sigma = f(Chromatic, ZLP, Voltage) / 1e-10  # Å
    defocus = np.array([-2*sigma, -sigma, 0, sigma, 2*sigma])
    mu = 0
    weighting = gaussian(defocus, mu, sigma)
    return sigma, defocus, weighting


def plot_defocus_distribution(sigma, defocus, weighting):
    'Plot '
    x = np.linspace(-4*sigma, +4*sigma, 1000)
    mu = 0
    gauss = gaussian(x, mu, sigma)

    fig, ax = plt.subplots()
    ax.plot(x, gauss, color='grey', zorder=-1, lw=4)
    plt.fill_between(x, gauss, color='lightgrey')

    defocus = np.array([-2*sigma, -sigma, 0, sigma, 2*sigma])
    plt.scatter(defocus, weighting, color='green')
    plt.xlabel('Temporal distribution in defocus / Å')
    plt.ylabel('Defocus weights')
    plt.ylim(0, None)

    data_coords = np.array([[0, i] for i in weighting])
    figure_coords = ax.transData.transform(data_coords)
    axes_coords = ax.transAxes.inverted().transform(figure_coords).T[1]
    for x0, ymax in zip(defocus, axes_coords):
        plt.axvline(x0, ymax=ymax, ls='--', color='green')
