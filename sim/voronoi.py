import atomap.api as am
import numpy as np


def integrate(s, add_atom_positions=True):
    # x, y = am.get_atom_positions(s, int(0.75/s.axes_manager[-1].scale)).T
    atom_positions = am.get_atom_positions(
        s, int(0.75/s.axes_manager[-1].scale))

    # the following were added for this specific dataset,
    # using the gui tool in atomap
    edge_atom_positions = [
        [12.159726974134387, 0.7876094840564178],
        [31.67717599562063, 0.31679382783636356],
        [49.48256808539756, 0.31679382783636356],
        [67.9727829478582, 0.31679382783636356],
        [87.49023196934446, 1.0016166005200944],
        [106.32285821814696, 0.31679382783636356],
        [125.49789585329137, 1.0016166005200944],
        [144.33052210209388, 1.0016166005200944],
        [12.159726974134387, 137.62375975092377],
        [31.334764609278764, 137.2813483645819],
        [49.824979471739425, 137.2813483645819],
        [69.0000171068838, 137.2813483645819],
        [87.1478205830026, 137.2813483645819],
        [105.98044683180512, 137.62375975092377],
        [124.81307308060764, 136.93893697824004],
        [144.33052210209388, 137.62375975092377]]
    if add_atom_positions:
        atom_positions = np.concatenate((atom_positions, edge_atom_positions))

    sub = am.Sublattice(atom_positions, s)
    sub.find_nearest_neighbors()
    sub.refine_atom_positions_using_center_of_mass(show_progressbar=False)
    sub.refine_atom_positions_using_2d_gaussian(show_progressbar=False)

    x, y = sub.atom_positions
    I, IM, PM = am.integrate(s.data, x, y, show_progressbar=False)

    from sim.voronoi import remove_integrated_edge_cells
    I2, IM2, PM2 = remove_integrated_edge_cells(I, IM, PM)

    # voronoi_map = am.integrate(s, x, y, show_progressbar=False)[1]
    return IM2  # voronoi_map


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
    i_record.data[indices] = np.nan if use_nans else 0
    p_record[indices] = -1

    if not inplace:
        return i_points, i_record, p_record
