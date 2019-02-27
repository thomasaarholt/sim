import atomap.api as am


def integrate(s):
    x, y = am.get_atom_positions(s, int(0.75/s.axes_manager[-1].scale)).T
    voronoi_map = am.integrate(s, x, y, show_progressbar=False)[1]
    return voronoi_map
