import pathlib
import hyperspy.api as hs
from sim.fileio import readMRC
from sim.voronoi import integrate
import numpy as np
import matplotlib.pyplot as plt

def stack_and_save():
    model_files = []
    for p in pathlib.Path('./Models').iterdir():
        if "_Prismatic" in p.name:
            model_files.append(p.stem)
            
    def key(entry):
        return int(entry.split('_d')[1].split('_')[0])


    sorted(model_files, key=key)
#    print(model_files)

    files = []
    for p in pathlib.Path('./prism').iterdir():
        if "_Prismatic" in p.name:
            files.append(p.name)

    headers = list(set([f.split('_FP')[0] for f in files]))

    group_of_files = []
    for header in headers:
        files = []
        for p in pathlib.Path('./prism').iterdir():
            if header in p.name and not p.is_dir() and p.suffix == ".mrc":
                files.append(p.name)
        files = sorted(files)
        group_of_files.append(files)

    def save(files, name):
        def read(filenames):
            data = []
            for filename in filenames:
                data.append(readMRC(filename))
            return np.asarray(data)
        s = hs.signals.Signal2D(read(files)).as_signal2D((0,3))
        s.metadata.add_node('Simulation')
        s.metadata.Simulation.Software = 'Prismatic'
        
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
        print('Begun saving!')

        haadf = s.inav[40.:].sum()
        fig, ax = plt.subplots(dpi=200)
        im = ax.imshow(haadf.data)

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
                    
            plt.subplots_adjust(0,0,1,1,0,0)
            for ax in fig.axes:
                    ax.axis('off')
                    ax.margins(0,0)
                    ax.xaxis.set_major_locator(plt.NullLocator())
                    ax.yaxis.set_major_locator(plt.NullLocator())
            fig.savefig(filepath, pad_inches = 0, bbox_inches='tight')



        s.save("hyperspy/" + name + ".hspy", overwrite=True)
        saveimg("hyperspy/" + name + "_HAADF_sum.png", fig=fig)
        saveimg("hyperspy/" + name + "_voronoi.png", fig=fig2)



    for files in group_of_files:
        load_files = sorted(["prism/" + f for f in files if "nothermal" not in f])
        if load_files:
            name = load_files[0].split("prism/")[1].split("_FP")[0]
            print('Stacking {}'.format(name))
            save(load_files, name)

        load_files = sorted(["prism/" + f for f in files if "nothermal" in f])
        if load_files:
            name = load_files[0].split("prism/")[1].split("_FP")[0]
            print('Stacking {}'.format(name))
            save(load_files, name + "_nothermal")