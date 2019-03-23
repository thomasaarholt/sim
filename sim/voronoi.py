import atomap.api as am
import numpy as np

def get_atom_positions_two_central_fraction():
    atom_positions = np.array([
        [137.32807007, 121.20097274],
        [118.59527918, 121.19310443],
        [ 99.85846114, 121.23258311],
        [ 81.14121135, 121.19251295],
        [ 62.43346371, 121.23178563],
        [ 43.70241186, 121.17577745],
        [ 25.00360544, 121.20428145],
        [  6.25953322, 121.21109704],
        [143.58514947, 103.90230741],
        [124.86136756, 103.90434222],
        [106.09021146, 103.86237618],
        [ 87.42625392, 103.91777283],
        [ 68.66914232, 103.89949767],
        [ 49.97202293, 103.90385764],
        [ 31.2234899 , 103.88921221],
        [ 12.46920808, 103.89483309],
        [137.32704071,  86.52934074],
        [118.63266652,  86.58627634],
        [ 99.89504886,  86.5770791 ],
        [ 81.12163312,  86.53453467],
        [ 62.41700128,  86.56872593],
        [ 43.70466519,  86.60362464],
        [ 24.94724398,  86.58597128],
        [  6.23183505,  86.55990797],
        [143.56440226,  69.25495177],
        [124.83464513,  69.26698011],
        [106.12263091,  69.2728027 ],
        [ 87.40849986,  69.26207382],
        [ 68.67779951,  69.28228929],
        [ 49.96189486,  69.23250855],
        [ 31.23581166,  69.20386461],
        [ 12.45498501,  69.25947329],
        [137.30260614,  51.91432786],
        [118.61194866,  51.97114171],
        [ 99.89213027,  51.94546504],
        [ 81.15240549,  51.93371159],
        [ 62.45625862,  51.93718111],
        [ 43.68296405,  51.95267197],
        [ 24.96990595,  51.9300639 ],
        [  6.25141868,  51.97046228],
        [143.56349753,  34.66946251],
        [124.83175512,  34.64735746],
        [106.13279093,  34.61800723],
        [ 87.38720936,  34.6237571 ],
        [ 68.64110958,  34.65477243],
        [ 49.91526436,  34.66555293],
        [ 31.2451763 ,  34.62713029],
        [ 12.49402476,  34.66094983],
        [137.31127594,  17.3295968 ],
        [118.58077809,  17.28444187],
        [ 99.88789027,  17.29324251],
        [ 81.15851156,  17.31809834],
        [ 62.4087769 ,  17.30078107],
        [ 43.69401902,  17.32852098],
        [ 24.96840218,  17.31407896],
        [  6.23650612,  17.31372092],
        [ 12.33945223,   1.73137242],
        [ 31.15848621,   1.75362791],
        [ 49.96048431,   1.74911722],
        [ 68.50593058,   1.7627292 ],
        [ 87.31381254,   1.75106082],
        [106.0669888 ,   1.7663879 ],
        [124.86625054,   1.73766195],
        [143.56533874,   1.74190191],
        [ 12.16963397, 136.25175019],
        [ 31.02630787, 136.30424282],
        [ 49.80600733, 136.2799493 ],
        [ 68.62918697, 136.27182203],
        [ 87.14911288, 136.30286664],
        [105.93874954, 136.25096892],
        [124.75436325, 136.29049215],
        [143.47306675, 136.2792313 ]
        ])
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
            image, int(0.75/s.axes_manager[-1].scale))

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
