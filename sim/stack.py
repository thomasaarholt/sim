import hyperspy.api as hs
from sim.fileio import readMRC
from sim.voronoi import integrate
from sim.stats import get_atom_indices, plot_standard_error_with_phonons
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from tqdm.auto import tqdm


def stack_and_save(simulation_folder='prism', add_atom_positions=False, save_hspy=True):
    plt.close('all')
    names = set([
        f.stem.split('_FP')[0] for f in
        Path("{}/".format(simulation_folder)).iterdir()
        if f.suffix == '.hspy' or f.suffix == '.mrc'
    ])

    groups = [sorted(glob.glob('{}/{}*'.format(simulation_folder, f)))
              for f in names]
    for files, name in tqdm(zip(groups, names), total=len(names)):
        tqdm.write('Stacking {}'.format(name))
        save(files, name, simulation_folder, add_atom_positions, save_hspy=save_hspy)


def read(filenames):
    data = []
    for filename in filenames:
        if filename.endswith('.mrc'):
            data.append(readMRC(filename))
    return np.asarray(data)


def save(files, name, simulation_folder='prism', add_atom_positions=True, save_hspy=True):
    if files[0].endswith('.mrc'):
        s = hs.signals.Signal2D(read(files)).as_signal2D((0, -1))
        s.metadata.add_node('Simulation')
        s.metadata.Simulation.Software = simulation_folder

        s.axes_manager[0].name = 'Acceptance Angle'
        s.axes_manager[0].units = 'mrad'
        s.axes_manager[0].offset = 0
        s.axes_manager[0].scale = 1

        s.axes_manager[1].name = 'Frozen Phonons'
        s.axes_manager[1].units = ''
        s.axes_manager[1].offset = 0
        s.axes_manager[1].scale = 1

        s.axes_manager[-2].name = 'X-Axis'
        s.axes_manager[-2].scale = 0.15
        s.axes_manager[-2].units = 'Å'

        s.axes_manager[-1].name = 'Y-Axis'
        s.axes_manager[-1].scale = 0.15
        s.axes_manager[-1].units = 'Å'

        haadf_series = s.inav[40.:].sum(0)
        haadf = haadf_series.sum()

    elif simulation_folder == 'multem':
        s = hs.load(files, stack=True).swap_axes(-1, -2)
        haadf = s.inav[1].sum()  # multem
        haadf.data = np.flip(haadf.data, axis=-1)
    tqdm.write('Begun saving!')

    Path('hyperspy/').mkdir(parents=True, exist_ok=True)
    if save_hspy:
        s.save("hyperspy/" + name + ".hspy", overwrite=True)

    fig, ax = plt.subplots(dpi=200)
    im = ax.imshow(haadf.data)
    saveimg("hyperspy/" + name + "_HAADF_sum.png", fig=fig)

    I, IM, PM = integrate(haadf, add_atom_positions)
    fig2, ax = plt.subplots(dpi=200)
    im2 = ax.imshow(IM.data)
    saveimg("hyperspy/" + name + "_voronoi.png", fig=fig2)

    if len(haadf_series.axes_manager.navigation_axes):
        I, IM, PM = integrate(haadf_series, add_atom_positions)
        error(I, name)


def error(I, name):
    indium_index, vacancy_index, bulk_index = get_atom_indices(I)
    fig = plot_standard_error_with_phonons(
        I, indium_index, vacancy_index, bulk_index)
    fig.savefig("hyperspy/" + name + "_error.png", dpi=200)


def saveimg(filepath, fig=None):
    '''Save the current image with no whitespace
    Example filepath: "myfig.png" or r"C:\myfig.pdf" 
    Based on answers from https://stackoverflow.com/questions/11837979/
    '''
    import matplotlib.pyplot as plt
    if not fig:
        fig = plt.gcf()

    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
    fig.savefig(filepath, pad_inches=0, bbox_inches='tight')
