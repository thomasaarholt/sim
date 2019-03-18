import hyperspy.api as hs
from sim.fileio import readMRC
from sim.voronoi import integrate
from sim.stats import get_atom_indices, plot_standard_error_with_phonons
from sim.defocus import get_series_defocus_and_weight
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
from pathlib import Path
from tqdm.auto import tqdm
from time import time
print('')
print('')
print('')


def stack_and_save_old(simulation_folder='prism', add_atom_positions=False, save_hspy=True):
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
        save(files, name, simulation_folder,
             add_atom_positions, save_hspy=save_hspy)


def stack_and_save(simulation_folder='prism', add_atom_positions=False, save_hspy=True):
    plt.close('all')

    sim_folder = Path("{}/".format(simulation_folder))
    sim_folder = get_files(sim_folder, ['.hspy', '.mrc'])
    names = sorted(set([get_stem(f) for f in sim_folder]))

    temperature = "RT" if "RT" in names[0] else "LN2"
    state = "DFT" if "DFT" in names[0] else "noDFT"
    firstpartname = names[0].split("_d")[0]
    corename = firstpartname + "_" + state + "_" + temperature

    depths, defoci = get_depth_and_defocus_lists(names)
    number_of_defoci = len(defoci)
    first_depth = depths[0]
    # just a random number so it works
    second_depth = depths[1] if len(depths) > 1 else first_depth + 1
    defect_delta_Z = 1.6218179
    depth_difference = second_depth - first_depth

    sigma, defocus_list, weighting_list = get_series_defocus_and_weight(
        ZLP=0.9, Voltage=3e5, Chromatic=1.6e-3)
    # indexing just for testing, harmless
    weighting_list = np.array(weighting_list)[:number_of_defoci]

    filename_structure = get_filename_structure(names[0])
    # depth_defocus_data = []
    for depth in tqdm(depths, desc="Reading depth"):
        defocus_data = []
        for defocus in tqdm(defoci, desc="Reading defocus"):
            filename = filename_structure.format(depth, defocus)
            filenames = get_sorted_filelist(filename, simulation_folder)
            defocus_data.append(read(filenames))
        data = np.asarray(defocus_data)

        s = hs.signals.Signal2D(data)
        s = s.as_signal2D((0, -1))

        s = s*weighting_list[:, None, None, None, None] / \
            weighting_list.sum() * number_of_defoci

        defocus_offset = defocus_list[0]
        defocus_scale = defocus_list[1] - defocus_list[0]

        s.metadata.add_node('Simulation')
        s.metadata.Simulation.Software = simulation_folder

        s.axes_manager[0].name = 'Acceptance Angle'
        s.axes_manager[0].units = 'mrad'
        s.axes_manager[0].offset = 0
        s.axes_manager[0].scale = 1

        s.axes_manager[1].name = 'Frozen Phonon'
        s.axes_manager[1].units = ''
        s.axes_manager[1].offset = 0
        s.axes_manager[1].scale = 1

        s.axes_manager[2].name = 'Defocus Series'
        s.axes_manager[2].units = 'Å'
        s.axes_manager[2].offset = defocus_offset
        s.axes_manager[2].scale = defocus_scale

        # s.axes_manager[3].name = 'Defect Depth Position'
        # s.axes_manager[3].units = 'Å'
        # s.axes_manager[3].offset = first_depth*defect_delta_Z
        # s.axes_manager[3].scale = defect_delta_Z*depth_difference

        s.axes_manager[-2].name = 'X-Axis'
        s.axes_manager[-2].scale = 0.15
        s.axes_manager[-2].units = 'Å'

        s.axes_manager[-1].name = 'Y-Axis'
        s.axes_manager[-1].scale = 0.15
        s.axes_manager[-1].units = 'Å'     

        save3(s, corename + "_d{:02}".format(depth), add_atom_positions)

def save3(s, name, add_atom_positions):
    tqdm.write("Saving {}".format(name))
    plt.close('all')
    Path('hyperspy/').mkdir(parents=True, exist_ok=True)
    number_of_defoci = s.axes_manager['Defocus Series'].size
    haadf_FP = HAADF(s).mean(1)  # sum over mrad, mean over defocus
    haadf_FP.axes_manager['Frozen Phonon'].offset = number_of_defoci
    haadf_FP.axes_manager['Frozen Phonon'].scale = number_of_defoci
    haadf = haadf_FP.mean(0)

    fig, ax = plt.subplots(dpi=200)
    im = ax.imshow(haadf.data)
    colorbar(im)
    saveimg("hyperspy/" + name + "_HAADF.png", fig=fig)

    I, IM, PM = integrate(haadf, add_atom_positions)
    fig2, ax = plt.subplots(dpi=200)
    im2 = ax.imshow(IM.data)
    colorbar(im2)
    saveimg("hyperspy/" + name + "_voronoi.png", fig=fig2)
    tqdm.write('\t' + 'Saving hspy file', end="")
    t = time()
    s.save("hyperspy/" + name + ".hspy", overwrite=True)
    tqdm.write('...ok, took {} seconds'.format(t - time()))
    
    if len(haadf_FP.axes_manager.navigation_axes):
        tqdm.write('\t' + 'Calculating error', end="")
        t = time()
        I, IM, PM = integrate(haadf_FP, add_atom_positions)
        error(I, name)
        tqdm.write('...ok, took {} seconds'.format(t - time()))

def save2(s, haadf_FP_depth_series, corename, depths, save_hspy=True, add_atom_positions=False):
    tqdm.write('Begun saving!')
    Path('hyperspy/').mkdir(parents=True, exist_ok=True)
    for i, depth in enumerate(depths):
        plt.close('all')
        name = corename + "_d" + str(depth)
        haadf_FP_series = haadf_FP_depth_series.inav[:, i]
        haadf = haadf_FP_series.mean()

        fig, ax = plt.subplots(dpi=200)
        im = ax.imshow(haadf.data)
        colorbar(im)
        saveimg("hyperspy/" + name + "_HAADF_sum.png", fig=fig)

        I, IM, PM = integrate(haadf, add_atom_positions)
        fig2, ax = plt.subplots(dpi=200)
        im2 = ax.imshow(IM.data)
        colorbar(im2)
        saveimg("hyperspy/" + name + "_voronoi.png", fig=fig2)

        if len(haadf_FP_series.axes_manager.navigation_axes):
            I, IM, PM = integrate(haadf_FP_series, add_atom_positions)
            error(I, name)
    if save_hspy:
        s.save("hyperspy/" + corename + ".hspy", overwrite=True)


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


def saveimg(filepath="image.png", fig=None):
    '''Save the current image with no whitespace
    Example filepath: "myfig.png" or r"C:\myfig.pdf" 
    Based on answers from https://stackoverflow.com/questions/11837979/
    '''
    import matplotlib.pyplot as plt
    if not fig:
        fig = plt.gcf()

    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    for ax in fig.axes:
        if ax.aname == 'colorbar':
            continue
        ax.axis('off')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    fig.savefig(filepath, pad_inches=0, bbox_inches='tight')


def colorbar(mappable):
    "mappable is img = plt.imshow()"
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.aname = 'colorbar'
    return fig.colorbar(mappable, cax=cax)


def get_depth_and_defocus_lists(names):
    depths = []
    defoci = []

    for name in names:
        depth, defocus = get_depth_and_defocus(name)
        depths.append(depth)
        defoci.append(defocus)

    depths = sorted(set(depths))
    defoci = sorted(set(defoci))
    return depths, defoci


def get_files(path, extensions=['.hspy', '.mrc']):
    all_files = []
    for ext in extensions:
        all_files.extend(path.glob("*" + ext))
    return all_files


def get_stem(filepath):
    f = Path(filepath)
    return f.stem.split('_FP')[0]


def get_filename_structure(first_name):
    keyleft = '_d'
    keyright = "defocus_"

    A, B = first_name.split(keyright)
    first, AB = A.split(keyleft)
    ABS = AB.split('_')
    second = "_" + "_".join(ABS[1:])

    BS = B.split('_')
    rest = "_" + "_".join(BS[1:])

    structure = first + keyleft + "{}" + second + keyright + "{}" + rest
    return structure


def get_depth_and_defocus(name):
    keyleft = '_d'
    keyright = "defocus_"

    A, B = name.split(keyright)
    first, AB = A.split(keyleft)
    ABS = AB.split('_')
    second = "_" + "_".join(ABS[1:])

    BS = B.split('_')
    rest = "_" + "_".join(BS[1:])

    depth = int(ABS[0])
    defocus = int(BS[0])

    return depth, defocus


def get_sorted_filelist(name, simulation_folder):
    filelist = glob.glob('{}/{}*'.format(simulation_folder, name))
    return sorted(filelist)


def HAADF(s, left=65., right=95.):
    return s.inav[left:right].sum(0)


def MAADF(s, left=35., right=45.):
    return s.inav[left:right].sum(0)
