def sample_bz(a1, a2, grid_size=100):
    """
    Sample a dense grid of reciprocal‐space points inside the 1st Brillouin Zone (BZ)
    for a 2D lattice defined by real‐space vectors a1, a2.

    Parameters
    ----------
    a1, a2 : array‐like, shape (2,)
        Real‐space lattice vectors.
    grid_size : int, default=100
        Number of points along each reciprocal‐coordinate axis to try.

    Returns
    -------
    samples : ndarray, shape (M, 2)
        Array of reciprocal‐basis coordinates (c1, c2) whose Cartesian images lie
        inside the 1st BZ.
    """
    # real→reciprocal basis
    a1 = np.array(a1, float)
    a2 = np.array(a2, float)
    area = abs(a1[0]*a2[1] - a1[1]*a2[0])
    b1 = 2*np.pi * np.array([-a2[1],  a2[0]]) / area
    b2 = 2*np.pi * np.array([ a1[1], -a1[0]]) / area
    B = np.column_stack((b1, b2))

    # build first‐BZ polygon via Voronoi
    rec_pts = [i*b1 + j*b2
               for i in range(-2, 3) for j in range(-2, 3)]
    rec_pts = np.vstack(rec_pts)
    vor = Voronoi(rec_pts)
    # find region index for the origin
    idx = next(r for (pt, r) in zip(vor.points, vor.point_region)
               if np.allclose(pt, 0))
    verts = vor.vertices[vor.regions[idx]]
    bz_poly = Polygon(verts)

    # find bounds in reciprocal‐basis coords
    c_bounds = np.linalg.solve(B, verts.T)  # shape (2, Nverts)
    c1min, c1max = c_bounds[0].min(), c_bounds[0].max()
    c2min, c2max = c_bounds[1].min(), c_bounds[1].max()

    # sample grid in reciprocal‐basis
    c1_vals = np.linspace(c1min, c1max, grid_size)
    c2_vals = np.linspace(c2min, c2max, grid_size)
    samples = []
    for c1 in c1_vals:
        for c2 in c2_vals:
            xyz = B.dot([c1, c2])
            if bz_poly.contains(Point(xyz)):
                samples.append((c1, c2))

    return np.array(samples)


def write_kpoints(filename, kpoints, mode='line', coord_sys='reciprocal'):
    """
    Write a sequence of k-points to a VASP-style KPOINTS file in 'line' mode.

    Parameters
    ----------
    filename : str
        Output path for the KPOINTS file.
    kpoints : sequence of (kx, ky[, kz])
        List of points in reciprocal-basis coordinates.
    mode : {'line'}, default 'line'
        KPOINTS generation mode. Only 'line' supported.
    coord_sys : {'reciprocal', 'cartesian'}, default 'reciprocal'
        Coordinate system of the provided k-points.

    File format:
        auto
        2
        line
        {coord_sys}
        k1
        k2

        k2
        k3

        ...
    """
    if mode != 'line':
        raise ValueError("Only 'line' mode is supported.")
    header = ['auto', '2', 'line', coord_sys]
    with open(filename, 'w') as f:
        for line in header:
            f.write(f"{line}\n")
        # iterate pairs
        for i in range(len(kpoints)-1):
            p1 = kpoints[i]
            p2 = kpoints[i+1]
            f.write("{:.3f}  {:.3f}  {:.3f}\n".format(*p1))
            f.write("{:.3f}  {:.3f}  {:.3f}\n".format(*p2))
            f.write("\n")




# Example usage:
# define lattice and center
a1 = [2.45, 0.0]
a2 = [-1.225, 2.1217622395]


# sample k points in the 2d brilouine zone
all_pts = sample_bz(a1,a2,grid_size=4)


# add zeros to the third column of kpoints to make each point be 3d k vector
zeros = np.zeros((all_pts.shape[0], 1))
kpoints_to_write = np.hstack((all_pts,zeros))

#write those 3d k vectors to KPOINTS file
write_kpoints('KPOINTS',kpoints_to_write)
print('KPOINTS written')
