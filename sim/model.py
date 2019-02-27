import numpy as np
from ase.visualize import view
from sim.cp import copy_file
from pathlib import Path
import os


def central_XY(central_fraction, sidelength):
    XMin = (1-central_fraction/sidelength)/2
    XMax = (1-(1-central_fraction/sidelength)/2)
    return XMin, XMax


def create(specs, **kwargs):
    comment = specs['comment']
    central_fraction = specs['central_fraction']
    single_index = specs['single_index']

    final_depth = kwargs['final_depth']
    sidelength = kwargs['sidelength']

    model_folder_name = "{}_{}".format(
        kwargs['defect_type'].capitalize(), kwargs['temperature'])
    if comment:
        model_folder_name += "_" + comment
    defect_range = final_depth - 1 - \
        np.array(single_index) if single_index else range(4, final_depth - 2)

    kwargs['model_folder_name'] = model_folder_name
    cells = [
        create_and_save_model_by_stacking(
            defect_depth_position=defect_position, **kwargs,
        ) for defect_position in defect_range]

    print(central_XY(central_fraction, sidelength))

    target_folder = kwargs['main_path'] / model_folder_name
    copy_file(target_folder)


def create_cube(sidelength=3, defect_type='relaxed', main_path="", temperature='RT', comment="", central_fraction=2.0):
    from ase import io
    from ase.build import stack
    from pathlib import Path
    import os

    model_folder_name = "cube_{}_{}".format(
        defect_type.capitalize(), temperature)
    if comment:
        model_folder_name += "_" + comment
    defect, bulk, name = load_defect_bulk(defect_type, name="")

    dim1_defect = defect.copy()
    dim1_bulk = bulk.copy()
    for i in range(int((sidelength-1)/2)):
        dim1_defect = stack(dim1_defect, bulk, axis=0)
        dim1_defect = stack(bulk, dim1_defect, axis=0)
        dim1_bulk = stack(dim1_bulk, bulk, axis=0)
        dim1_bulk = stack(bulk, dim1_bulk, axis=0)

    dim2_defect = dim1_defect.copy()
    dim2_bulk = dim1_bulk.copy()

    for i in range(int((sidelength-1)/2)):
        dim2_defect = stack(dim2_defect, dim1_bulk, axis=1)
        dim2_defect = stack(dim1_bulk, dim2_defect, axis=1)
        dim2_bulk = stack(dim2_bulk, dim1_bulk, axis=1)
        dim2_bulk = stack(dim1_bulk, dim2_bulk, axis=1)

    dim3_defect = dim2_defect.copy()

    for i in range(int((sidelength-1)/2)):
        dim3_defect = stack(dim3_defect, dim2_bulk, axis=2)
        dim3_defect = stack(dim2_bulk, dim3_defect, axis=2)

    model = dim3_defect
    model.info['name'] = model_folder_name
    if main_path == "":
        raise ValueError(
            'main_path must be the parent folder of where I am running the simulation from')
    model_path = Path(main_path) / model_folder_name / 'Models'
    model_path.mkdir(parents=True, exist_ok=True)
    model_path = str(model_path)

    XMin, XMax = central_XY(central_fraction, sidelength)

    multem_cell = flip_for_MULTEM(model)
    multem_cell.info['name'] += '_Multem'
    target_folder = Path(main_path) / model_folder_name

    with open(str(Path(__file__).parent / 'run.py'), 'r') as myfile:
        run_contents = myfile.read()
    run_contents = run_contents.format(XMin, XMax)
    with open(str(target_folder / 'run.py'), 'a') as the_file:
        the_file.write(run_contents)
    copy_file(target_folder)

    save_cell(cell=model, name=model.info['name'], path=model_path,
              only_save='prismatic', temperature=temperature)
    save_cell(cell=model, name=model.info['name'],
              path=model_path, only_save='xyz', temperature=temperature)
    save_cell(cell=multem_cell, name=multem_cell.info['name'],
              path=model_path, only_save='multem', temperature=temperature)
    save_cell(cell=multem_cell, name=multem_cell.info['name'],
              path=model_path, only_save='xyz', temperature=temperature)


def flip_for_MULTEM(cell):
    cell2 = cell.copy()
    cell2.positions[:, 2] = cell2.cell.diagonal()[2] - cell2.positions[:, 2]
    return cell2


def load_defect_bulk(defect_type, name):
    from ase.io import read
    model_path = Path(__file__).parent / 'DFT Cells'
    bulk_filename = "ZnO_Unitcell_Orth_3x2x2.cif"
    bulk_path = model_path / bulk_filename
    bulk = read(bulk_path)

    # Set whether to use DFT in the variable above
    if defect_type == 'relaxed':
        defect_filename = "ZnO_Unitcell_Orth_3x2x2_InVZnO_Relaxed.cif"
        name += '_DFT'
    elif defect_type == 'bulk':
        defect_filename = bulk_filename
        name += '_bulkZnO'
    elif defect_type == 'indium':
        defect_filename = "ZnO_Unitcell_Orth_3x2x2_InVZnO_Static.cif"  # we modify this later
        name += '_indium'
    else:
        defect_filename = "ZnO_Unitcell_Orth_3x2x2_InVZnO_Static.cif"
        name += '_noDFT'
    defect = read(model_path / defect_filename)

    bulk = rotate90(bulk)
    defect = rotate90(defect)

    return defect, bulk, name


def create_defect_by_stacking(defect_depth_position, sidelength, final_depth, defect_type='relaxed', final_sidelength=None):
    '''Create defect at a certain layer position, 
    with a supercell sidelength (odd number), and the total final depth in layers
    defect_type:
    - 'relaxed'
    - 'static'
    - 'bulk'
    - 'indium'

    final_width should be either None or a length in Å describing the total volume of vacuum that the sample is placed in
    '''
    from ase import io
    from ase.build import cut, stack, make_supercell
    from pathlib import Path
    import os

    prismatic_equivalent_depth_position = final_depth - 1 - defect_depth_position

    name = 'ZnO_xy'+str(sidelength)+'_z'+str(final_depth) + \
        '_d'+str(prismatic_equivalent_depth_position) + '_rev'

    defect, bulk, name = load_defect_bulk(defect_type, name)

    total_number_of_layers_in_supercell = 6
    natural_defect_depth_position_in_supercell = 4

    x_, y_, z_ = bulk.cell.diagonal()
    layer_thickness = z_/total_number_of_layers_in_supercell

    def calculate_bulk_before_defect(defect_depth_position):
        return (defect_depth_position + 1) // total_number_of_layers_in_supercell

    def calculate_bulk_after_defect(defect_depth_position, final_depth):
        import math
        return math.ceil((final_depth - defect_depth_position) / total_number_of_layers_in_supercell)

    def layers_to_take_off_the_top(defect_depth_position):
        return (natural_defect_depth_position_in_supercell - defect_depth_position) % total_number_of_layers_in_supercell

    dim1_defect = defect.copy()
    dim1_bulk = bulk.copy()
    for i in range(int((sidelength-1)/2)):
        dim1_defect = stack(dim1_defect, bulk, axis=0)
        dim1_defect = stack(bulk, dim1_defect, axis=0)
        dim1_bulk = stack(dim1_bulk, bulk, axis=0)
        dim1_bulk = stack(bulk, dim1_bulk, axis=0)

    dim2_defect = dim1_defect.copy()
    dim2_bulk = dim1_bulk.copy()

    for i in range(int((sidelength-1)/2)):
        dim2_defect = stack(dim2_defect, dim1_bulk, axis=1)
        dim2_defect = stack(dim1_bulk, dim2_defect, axis=1)
        dim2_bulk = stack(dim2_bulk, dim1_bulk, axis=1)
        dim2_bulk = stack(dim1_bulk, dim2_bulk, axis=1)

    dim3_defect = dim2_defect.copy()

    number_of_bulk_before_defect = calculate_bulk_before_defect(
        defect_depth_position)
    if number_of_bulk_before_defect:
        bulk_before_defect_matrix = np.diag(
            [sidelength, sidelength, number_of_bulk_before_defect])
        bulk_before_defect = make_supercell(bulk, bulk_before_defect_matrix)
        dim3_defect = stack(bulk_before_defect, dim3_defect)

    number_of_bulk_after_defect = calculate_bulk_after_defect(
        defect_depth_position, final_depth)
    if number_of_bulk_after_defect:
        bulk_after_defect_matrix = np.diag(
            [sidelength, sidelength, number_of_bulk_after_defect])
        bulk_after_defect = make_supercell(bulk, bulk_after_defect_matrix)
        dim3_defect = stack(dim3_defect, bulk_after_defect)

    model = dim3_defect
    model.positions[:, 2] -= layers_to_take_off_the_top(
        defect_depth_position)*layer_thickness
    model.cell[2, 2] = final_depth*layer_thickness

    outside_cell = [atom.index for atom in model if (atom.position < [0, 0, 0]).any() or (
        atom.position >= model.cell.diagonal()).any() or (atom.position[2] + 0.0001 >= model.cell[2, 2]).any()]
    # we delete atoms - but this should only be in the z direction
    del model[outside_cell]

    model.info['name'] = name

    if defect_type == 'indium':
        # creates an indium column and a vacancy column
        indium_index = np.where(model.numbers == 49)
        x, y, z = model.positions[indium_index][0]

        Inx = np.logical_and(
            model.positions[:, 0] > x-0.1, model.positions[:, 0] < x+0.1)
        Iny = np.logical_and(
            model.positions[:, 1] > y-0.1, model.positions[:, 1] < y+0.1)

        model.numbers[Inx*Iny] = 49

        Vx = np.logical_and(model.positions[:, 0] > (
            (sidelength-1)/2+0.49)*x_, model.positions[:, 0] < ((sidelength-1)/2+0.51)*x_)
        Vy = np.logical_and(model.positions[:, 1] > (
            (sidelength-1)/2+0.41)*y_, model.positions[:, 1] < ((sidelength-1)/2+0.42)*y_)
        del model[Vx*Vy]

    if final_sidelength != None:
        assert final_sidelength > model.cell[0, 0] and final_sidelength > model.cell[1, 1], \
            'final_sidelength smaller than cell!: {} vs {}|{}'.format(
                final_sidelength, model.cell[0, 0], model.cell[1, 1])

        vacuum_x = (final_sidelength - model.cell[0, 0])/2
        vacuum_y = (final_sidelength - model.cell[1, 1])/2

        model.positions[:, 0] += vacuum_x
        model.positions[:, 1] += vacuum_y

        model.cell[0, 0], model.cell[1,
                                     1], = final_sidelength, final_sidelength
    return model


def create_and_save_model_by_stacking(
        defect_depth_position, sidelength, final_depth, defect_type='relaxed',
        temperature='LN2', final_sidelength=None, model_folder_name='Models', only_save=False, main_path=None):
    '''Create defect at a certain layer position, 
    with a supercell sidelength (odd number), and the total final depth in layers
    defect_type:
    - 'relaxed'
    - 'static'
    - 'bulk'
    - 'indium'

    temperature:
    - 'LN2'
    - 'RT'

    only_save:
    - False
    - 'prismatic'
    - 'multem'
    - 'xyz'
    - 'cif'
    - 'none'
    '''
    from pathlib import Path
    cell = create_defect_by_stacking(defect_depth_position=defect_depth_position, sidelength=sidelength,
                                     final_depth=final_depth, defect_type=defect_type, final_sidelength=final_sidelength)
    if main_path == None:
        main_path = Path.cwd()
    model_path = str(Path(main_path) / model_folder_name / 'Models')
    save_cell(cell, cell.info['name'], model_path, only_save, temperature)

    Path("./{}".format(model_folder_name)).mkdir(exist_ok=True)
    return cell


def rotate90(cell, axis1=0, axis2=2):
    cell2 = cell.copy()
    cell2.positions[:, axis1], cell2.positions[:,
                                               axis2] = cell2.positions[:, axis2], cell2.positions[:, axis1].copy()
    cell2.cell[axis1, axis1], cell2.cell[axis2,
                                         axis2] = cell2.cell[axis2, axis2], cell2.cell[axis1, axis1].copy()
    return cell2


def relativistic_wavelength(kV=300):
    'Returns relativistic wavelength in meters'
    import numpy as np
    from scipy.constants import h, c, electron_mass, e
    V = 1e3*kV
    top = h*c
    bottom = np.sqrt(e*V*(2*electron_mass*c**2 + e*V))
    wavelength = top / bottom
    return wavelength


def max_scattering_angle_mrad_prismatic(kV=300, potential_spacing=0.05*1e-10):
    'Prismatic-style maximum scattering angle for a given pixel size'
    wavelength = relativistic_wavelength(kV)
    amax = 0.5*wavelength/potential_spacing*1000
    return amax


def check_sample_size_big_enough(sample_length_Å=50, kV=300, alpha=20e-3):
    ratio = alpha/angular_resolution(sample_length_Å, kV)
    assert ratio > 10, 'Increase sample size. Alpha / Angular Res is currently {:.2f}, must be larger than 10'.format(
        ratio)


def angular_resolution(sample_length_Å=50, kV=300):
    'Returns in radians, not mrad'
    sample_length = sample_length_Å * 1e-10
    wavelength = relativistic_wavelength(kV)
    reciprocal_resolution_k = 1/sample_length
    ang_res = wavelength*reciprocal_resolution_k
    return ang_res


def check_simulation_parameters(sample_length_Å=50, potential_spacing=0.05*1e-10, kV=300,  alpha=20e-3,):
    print('Max scattering angle is {:.2f}'.format(
        max_scattering_angle_mrad_prismatic(kV, potential_spacing)))
    check_sample_size_big_enough(sample_length_Å, kV, alpha)
    print('Model passed health check')


def save_cell(cell, name, path="", only_save=False, temperature='LN2'):
    '''Save cell with name to path
    only_save:
    - False
    - 'prismatic'
    - 'multem'
    - 'xyz'
    - 'cif'
    - 'none'
    '''

    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)

    if only_save == 'prismatic':
        save_cell_for_prismatic(cell, name, path, temperature=temperature)
    elif only_save == 'multem':
        save_cell_multem(cell, name, path, temperature=temperature)
    elif only_save == 'xyz':
        save_cell_structure(cell, name, path, 'xyz')
    elif only_save == 'cif':
        save_cell_structure(cell, name, path, 'cif')
    elif only_save == 'none':
        pass
    else:
        save_cell_for_prismatic(cell, name, path, temperature=temperature)
        save_cell_multem(cell, name, path, temperature=temperature)
        save_cell_multem_txt(cell, name, path, temperature=temperature)
        save_cell_structure(cell, name, path, 'xyz')
        save_cell_structure(cell, name, path, 'cif')


def save_cell_structure(cell, name, path="", format="xyz"):
    import os

    if path == "":
        path = os.getcwd()
    path = os.path.join(path, "")

    cell.write(os.path.join(path, name + "." + format))


def save_cell_for_prismatic(sample, name, path="", temperature='RT'):
    "Auto-adds 'prismatic' to filename"
    import os
    import numpy as np

    if path == "":
        path = os.getcwd()
    name += "_Prismatic"
    name += "_" + temperature
    total = np.zeros(len(sample.numbers), dtype=[('Z', 'int8'), ("x", 'float32'), (
        "y", 'float32'), ("z", 'float32'), ('occ', 'int'), ('dw', 'float')])
    total['Z'] = sample.numbers
    total['x'], total['y'], total['z'] = sample.positions.T
    total["occ"] = 1

    O_sites = np.where(total['Z'] == 8)[0]
    Zn_sites = np.where(total['Z'] == 30)[0]
    In_sites = np.where(total['Z'] == 49)[0]

    O_rms = debye_waller('O', temperature)
    Zn_rms = debye_waller('Zn', temperature)
    In_rms = Zn_rms  # assume same as Zn

    total['dw'][O_sites] = round(O_rms, 3)
    total['dw'][Zn_sites] = round(Zn_rms, 3)
    total['dw'][In_sites] = round(In_rms, 3)

    positions = " ".join(format(x, "0.6f") for x in sample.cell.diagonal())

    header = "Test cell" + "\n\t" + positions
    footer = "-1"
    fmt = ['%d', '%0.9f', '%0.9f', '%0.9f', '%0.1f', '%0.3f']
    np.savetxt(os.path.join(path, name + ".xyz"), total, fmt=fmt,
               header=header, footer=footer, comments="",)  # encoding='utf-8')


def save_cell_multem(bigcell, name, path="", temperature='RT'):
    import os
    import numpy as np
    from scipy import io as out

    if path == "":
        path = os.getcwd()
    path = os.path.join(path, "")

    name += '_' + temperature

    atomic_number_to_atomic_symbol_dict = {
        8: 'O',
        30: 'Zn',
        49: 'In'
    }

    occ = [1]
    label = [0]
    charge = [0]
    data = [[n] + list(xyz) + [debye_waller(atomic_number_to_atomic_symbol_dict[n], temperature)] +
            occ + label + charge for n, xyz in zip(bigcell.numbers, bigcell.positions)]

    if bigcell.cell[0, 0] == 0:
        print("Detected that sample has been rotated without rotating the unit cell")
        sample_x = bigcell.cell[2, 0]
        sample_y = bigcell.cell[1, 1]
        sample_z = bigcell.cell[0, 2]
    else:
        sample_x = bigcell.cell[0, 0]
        sample_y = bigcell.cell[1, 1]
        sample_z = bigcell.cell[2, 2]

    matlabdict = {
        "spec_atoms": data,
        "spec_lx": sample_x,
        "spec_ly": sample_y,
        "spec_lz": sample_z,
        "spec_dz": 1.62182,  # d of 110
        "a": sample_x,  # changed to bigcell from ZnO
        "b": sample_y,
        "c": sample_z,
    }
    out.savemat(os.path.join(path, name), matlabdict)


def save_cell_multem_txt(bigcell, name, path="", temperature='RT'):
    #raise Exception('This needs to be updated for debye waller factors')
    import os
    import numpy as np

    from scipy import io as out
    if path == "":
        path = os.getcwd()
    path = os.path.join(path, "")

    if bigcell.cell[0, 0] == 0:
        print("Detected that sample has been rotated without rotating the unit cell")
        sample_x = bigcell.cell[2, 0]
        sample_y = bigcell.cell[1, 1]
        sample_z = bigcell.cell[0, 2]
    else:
        sample_x = bigcell.cell[0, 0]
        sample_y = bigcell.cell[1, 1]
        sample_z = bigcell.cell[2, 2]

    lx = [sample_x]
    ly = [sample_y]
    dz = [1.62182]  # d of 110

    header = [lx + ly + dz + 5*[0]]

    rms3d_atoms = 0.10070188209027682
    rms3d_carbon = 0.085

    occ = [1]
    label = [0]
    charge = [0]

    rms3d_list = [rms3d_carbon if Z ==
                  6 else rms3d_atoms for Z in bigcell.numbers]

    data = [[n] + list(xyz) + [rms3d] + occ + label + charge for n, xyz,
            rms3d in zip(bigcell.numbers, bigcell.positions, rms3d_list)]

    total = np.array(header + data)
    #total = [item for sublist in total for item in sublist]

    np.savetxt(path + name + ".txt", total,
               fmt='%.8f', newline="\n", delimiter=" ")


def debye_waller(element='Zn', temperature='RT'):
    '''Zn, In or O, RT or LN2'''
    if element == 'Zn' or element == 'In':
        if temperature == 'RT':
            dw = 0.10070188209027682
        if temperature == 'LN2':
            dw = 0.062474299060172774

    if element == 'O':
        if temperature == 'RT':
            dw = 0.09872241583012728
        if temperature == 'LN2':
            dw = 0.06435856712918735
    return dw
