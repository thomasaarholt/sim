import atomap.api as am
import numpy as np


def integrate(s, add_atom_positions=True, remove_edges=True):
    # x, y = am.get_atom_positions(s, int(0.75/s.axes_manager[-1].scale)).T
    image = s.sum()
    atom_positions = am.get_atom_positions(
        image, int(0.75/s.axes_manager[-1].scale))

    # the following were added for this specific dataset,
    # using the gui tool in atomap
    edge_atom_positions = [
    [57.07284167768582, 19.179257269753712],
    [50.84383635437713, 1.759157636771782],
    [32.15682038445106, 1.8647339981837945],
    [13.68095713734901, 2.3926158052438495],
    [1.1173701293196148, 19.390409992577737],
    [13.469804414524985, 37.021662348383686],
    [31.945667661627034, 36.704933264147655],
    [51.16056543861317, 36.91608598697168]]
    if add_atom_positions:
        atom_positions = np.concatenate((atom_positions, edge_atom_positions))

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
