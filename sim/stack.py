import pathlib
import hyperspy.api as hs
from sim.fileio import readMRC
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

    def save(files, name):
        def read(filenames):
            data = []
            for name in filenames:
                data.append(readMRC(name))
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

        fig, ax = plt.subplots(dpi=200)
        im = ax.imshow(s.inav[40.:].sum().data)

        def save(filepath, fig=None):
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



        s.save("hyperspy/" + name, overwrite=True)
        save("hyperspy/" + name + "_HAADF_sum.png")



    for model in model_files:
        load_files = sorted(["prism/" + f for f in files if f.startswith(model) and "thermal" not in f and f.endswith('.mrc')])
        if load_files:
            print('Stacking {}'.format(model))
            save(load_files, model)

        load_files = sorted(["prism/" + f for f in files if f.startswith(model) and "thermal" in f and f.endswith('.mrc')])
        if load_files:
            print('Stacking {}'.format(model))
            save(load_files, model + "_thermal")