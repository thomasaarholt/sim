import atomap.api as am
import numpy as np


def get_atom_positions_two_central_fraction():
    atom_positions = np.array([
        [2.57546548e+02, 2.27221296e+02],
        [2.22510578e+02, 2.27354287e+02],
        [1.87348631e+02, 2.27197441e+02],
        [1.52080260e+02, 2.27234089e+02],
        [1.17049824e+02, 2.27367256e+02],
        [8.18858489e+01, 2.27232049e+02],
        [4.69416842e+01, 2.27247358e+02],
        [1.16885489e+01, 2.27364756e+02],
        [2.69229005e+02, 1.94743480e+02],
        [2.34176408e+02, 1.94885949e+02],
        [1.98969013e+02, 1.94746557e+02],
        [1.63880562e+02, 1.94814344e+02],
        [1.28763319e+02, 1.94879874e+02],
        [9.36085077e+01, 1.94762515e+02],
        [5.84674009e+01, 1.94786390e+02],
        [2.33959295e+01, 1.94752613e+02],
        [2.57553676e+02, 1.62340393e+02],
        [2.22343404e+02, 1.62260080e+02],
        [1.87165674e+02, 1.62248089e+02],
        [1.52089724e+02, 1.62183018e+02],
        [1.17090820e+02, 1.62299524e+02],
        [8.19694680e+01, 1.62315403e+02],
        [4.67742214e+01, 1.62309402e+02],
        [1.17238747e+01, 1.62282356e+02],
        [2.69161409e+02, 1.29877076e+02],
        [2.34048944e+02, 1.29822311e+02],
        [1.98959122e+02, 1.29819361e+02],
        [1.63923043e+02, 1.29785176e+02],
        [1.28692311e+02, 1.29868469e+02],
        [9.36915138e+01, 1.29820038e+02],
        [5.85391628e+01, 1.29866258e+02],
        [2.34277021e+01, 1.29776432e+02],
        [2.57379101e+02, 9.74872628e+01],
        [8.19265444e+01, 9.75711894e+01],
        [1.17297405e+01, 9.74971135e+01],
        [2.22427153e+02, 9.73206000e+01],
        [1.87177634e+02, 9.74059960e+01],
        [1.52121721e+02, 9.73725284e+01],
        [1.16988028e+02, 9.73503644e+01],
        [4.68866059e+01, 9.74457352e+01],
        [2.69219370e+02, 6.49568926e+01],
        [2.34040706e+02, 6.49727120e+01],
        [1.99000141e+02, 6.49931479e+01],
        [1.63876862e+02, 6.49141812e+01],
        [1.28833181e+02, 6.49995526e+01],
        [9.36971024e+01, 6.49310904e+01],
        [5.85940694e+01, 6.48581152e+01],
        [2.34146759e+01, 6.49740960e+01],
        [2.22308972e+02, 3.24777550e+01],
        [1.16607498e+01, 3.25532470e+01],
        [2.57530430e+02, 3.23535984e+01],
        [1.87260616e+02, 3.24311879e+01],
        [1.52128615e+02, 3.24464478e+01],
        [1.17164279e+02, 3.24180014e+01],
        [8.19530558e+01, 3.24139516e+01],
        [4.69287909e+01, 3.24778312e+01],
        [2.33723498e+01, 2.10000000e-01],
        [5.83158294e+01, 2.10000000e-01],
        [9.34368268e+01, 2.10000000e-01],
        [1.28768909e+02, 2.10000000e-01],
        [1.63815021e+02, 2.10000000e-01],
        [1.98980356e+02, 2.10000000e-01],
        [2.33998143e+02, 2.10000000e-01],
        [2.68992993e+02, 2.10000000e-01],
        [2.30683065e+01, 2.59520000e+02],
        [5.81541066e+01, 2.59520000e+02],
        [9.32174170e+01, 2.59520000e+02],
        [1.28669146e+02, 2.59520000e+02],
        [1.63662874e+02, 2.59520000e+02],
        [1.98437229e+02, 2.59520000e+02],
        [2.33870400e+02, 2.59520000e+02],
        [2.68781989e+02, 2.59520000e+02]])
    return atom_positions


def get_indium_vacancy_positions_two_central_fraction():
    return (27, 28)


def integrate(s, use_two_central_positions=True, remove_edges=True):
    # x, y = am.get_atom_positions(s, int(0.75/s.axes_manager[-1].scale)).T
    image = s.sum()

    if use_two_central_positions:
        x, y = get_atom_positions_two_central_fraction().T
    else:
        atom_positions = am.get_atom_positions(
            image, int(1.0/s.axes_manager[-1].scale))

        sub = am.Sublattice(atom_positions, image)
        sub.find_nearest_neighbors()
        sub.refine_atom_positions_using_center_of_mass(show_progressbar=False)
        sub.refine_atom_positions_using_2d_gaussian(show_progressbar=False)
        x, y = sub.atom_positions.T

    I, IM, PM = am.integrate(s.data, x, y, show_progressbar=False)

    if remove_edges:
        I, IM, PM = remove_integrated_edge_cells(I, IM, PM)

    # voronoi_map = am.integrate(s, x, y, show_progressbar=False)[1]
    return I, IM, PM  # voronoi_map


def _border_elems(image, pixels=1):
    """
    Return the values of the edges along the border of the image, with
    border width `pixels`.
    """
    arr = np.ones_like(image, dtype=bool)
    arr[pixels:-1-(pixels-1), pixels:-1-(pixels-1)] = False
    return image[arr]


def remove_integrated_edge_cells(i_points, i_record, p_record,
                                 pixels=1, use_nans=True, inplace=False):
    """Removes any cells that touch within a number of pixels of
    the image border.

    Note on using use_nans: If this is used on a dataset with more than
    two dimensions, the resulting hyperspy i_record signal might be needed to
    be viewed with i_record.plot(navigator='slider'), since hyperspy may throw
    an error when plotting a dataset with only NaNs present.

    Parameters
    ----------
    i_points : NumPy array
        The output of the atomap integrate function or method
    i_record : NumPy array
        The output of the atomap integrate function or method
    p_record : NumPy array
        The output of the atomap integrate function or method

    Returns
    -------
    i_points : NumPy array
        Modified list of integrated intensities with either np.nan or 0
        on the removed values, which preserves the atom index.
    i_record : HyperSpy signal
        Modified integrated intensity record, with either np.nan or 0
        on the removed values, which preserves the atom index
    p_record : NumPy array, same size as image
        Modified points record, where removed areas have value = -1.

    Example
    -------
    points_x, points_y = am.get_atom_positions(s).T
    i, ir, pr = am.integrate(s, points_x, points_y, method='Voronoi')
    i2, ir2, pr2 = remove_integrated_edge_cells(i, ir, pr, pixels=5,
                                                use_nans=True)
    """

    if not inplace:
        i_points = i_points.copy()
        i_record = i_record.deepcopy()
        p_record = p_record.copy()

    border = _border_elems(p_record, pixels)
    border_indices = np.array(list(set(border)))
    indices = np.in1d(p_record, border_indices).reshape(p_record.shape)
    i_points[border_indices] = np.nan if use_nans else 0
    i_record.data[..., indices] = np.nan if use_nans else 0
    p_record[indices] = -1

    if not inplace:
        return i_points, i_record, p_record
