import hyperspy.api as hs
from sim.fileio import readMRC
from sim.voronoi import integrate
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path


def stack_and_save(simulation_folder='prism'):
    names = set([
        f.stem.split('_FP')[0] for f in
        Path("{}/".format(simulation_folder)).iterdir()
        if f.suffix == '.hspy' or f.suffix == '.mrc'
    ])

    groups = [sorted(glob.glob('{}/{}*'.format(simulation_folder, f))) for f in names]

    def save(files, name):
        print(files)
        if files[0].endswith('.mrc'):
            def read(filenames):
                data = []
                for filename in filenames:
                    print(filename)
                    data.append(readMRC(filename))
                print('Next')
                return np.asarray(data)
            s = hs.signals.Signal2D(read(files)).as_signal2D((0, 3))
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

            s.axes_manager[2].name = 'X-Axis'
            s.axes_manager[2].scale = 0.15
            s.axes_manager[2].units = 'Å'

            s.axes_manager[3].name = 'Y-Axis'
            s.axes_manager[3].scale = 0.15
            s.axes_manager[3].units = 'Å'

            haadf = s.inav[40.:].sum()
        elif simulation_folder == 'multem':
            s = hs.load(files, stack=True).swap_axes(-1,-2)
            haadf = s.inav[1].sum() #multem
            haadf.data = np.flip(haadf.data, axis=-1)
        print('Begun saving!')

        fig, ax = plt.subplots(dpi=200)
        im = ax.imshow(haadf.data)
        print(haadf)

        fig2, ax = plt.subplots(dpi=200)
        im2 = ax.imshow(integrate(haadf).data)

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

        s.save("hyperspy/" + name + ".hspy", overwrite=True)
        saveimg("hyperspy/" + name + "_HAADF_sum.png", fig=fig)
        saveimg("hyperspy/" + name + "_voronoi.png", fig=fig2)

    for files, name in zip(groups, names):
        print('Stacking {}'.format(name))
        save(files, name)
