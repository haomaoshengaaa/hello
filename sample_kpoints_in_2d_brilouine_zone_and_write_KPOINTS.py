def sample_2d_brillouin_zone(a1, a2, grid_size=100, center_rec=(0.0,0.0), radius=None):
    """
    Sample a dense grid of reciprocal‐space points inside the 1st Brillouin Zone (BZ)
    for a 2D lattice defined by real‐space vectors a1, a2.

    Parameters
    ----------
    a1, a2 : array‐like, shape (2,)
        Real‐space lattice vectors.
    grid_size : int, default=100
        Number of points along each reciprocal‐coordinate axis to try.
    center_rec:tuple
        if radius is not None, any k-point outside of the circle centered around given
        center_rec with given radius will be excluded.
    radius:float
        see explanation for center_rec
        

    Returns
    -------
    samples : ndarray, shape (M, 2)
        Array of reciprocal‐basis coordinates (c1, c2) whose Cartesian images lie
        inside the 1st BZ.
    """

    if radius is not None and radius < 0:
        raise ValueError("radius must be non‐negative")


    # real→reciprocal basis
    a1 = np.array(a1, float)
    a2 = np.array(a2, float)
    area = abs(a1[0]*a2[1] - a1[1]*a2[0])
    b1 = 2*np.pi * np.array([-a2[1],  a2[0]]) / area
    b2 = 2*np.pi * np.array([ a1[1], -a1[0]]) / area
    B = np.column_stack((b1, b2))


    # ---- NEW: Cartesian center of exclusion circle ----
    if radius is not None:
        center_cart = B.dot(center_rec)

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
                # if radius is set, also enforce circle
                if radius is None or np.linalg.norm(xyz - center_cart) <= radius:
                    samples.append((c1, c2))


    return np.array(samples)

def write_kpoints(filename, kpoints, mode='line', coord_sys='reciprocal'):
    """
    Write k-points to one or more VASP-style KPOINTS files in 'line' mode.

    If the total number of kpoints exceeds 400, splits them into multiple
    files each containing at most 400 points. Filenames will be suffixed with
    an index (e.g., KPOINTS_1, KPOINTS_2, ...). If only one file is needed,
    it retains the given filename.
    """
    # Ensure only 'line' mode is supported
    if mode != 'line':
        raise ValueError("Only 'line' mode is supported.")

    # Determine how many sections we need (max 400 kpoints per section)
    total = len(kpoints)                        # total number of kpoints
    max_per_file = 400                          # maximum per KPOINTS file
    # calculate number of output files required
    sections = (total + max_per_file - 1) // max_per_file

    # Loop over each section and write separate KPOINTS file
    for idx in range(sections):
        # Determine slice indices for this section
        start = idx * max_per_file
        end = min(start + max_per_file, total)
        chunk = kpoints[start:end]               # subset of kpoints for this file

        # Determine output filename: add suffix only if multiple files
        if sections == 1:
            out_name = filename                 # single output file
        else:
            base, ext = os.path.splitext(filename)
            # If there's no extension, ext will be ''
            out_name = f"{base}_{idx+1}{ext}"

        # Open and write this section to its KPOINTS file
        with open(out_name, 'w') as f:
            # HEADER: auto generation, 2 lines, mode, and coordinate system
            f.write('auto\n')
            f.write('2\n')
            f.write('line\n')
            f.write(f"{coord_sys}\n")

            # Write each pair of consecutive kpoints with blank line separators
            for i in range(len(chunk) - 1):
                p1 = chunk[i]                     # first point
                p2 = chunk[i + 1]                 # second point
                # write k-point coordinates, formatted to 3 decimal places
                f.write("{:.3f}  {:.3f}  {:.3f}\n".format(*p1))
                f.write("{:.3f}  {:.3f}  {:.3f}\n".format(*p2))
                f.write('\n')                    # blank line between segments
        print(f"{out_name} written")

def sample_and_add_kz(kxy_array, kz_num):
    n = kxy_array.shape[0]
    kz_values = np.linspace(-0.5, 0.5, kz_num, endpoint=True)
    kz_column = np.repeat(kz_values, n)[:, np.newaxis]
    kxy_repeated = np.tile(kxy_array, (kz_num, 1))
    return np.hstack((kxy_repeated, kz_column))




# Example usage:
# define lattice and center
a1 = [2.45, 0.0]
a2 = [-1.225, 2.1217622395]
# sample k points in the 2d brilouine zone
all_pts = sample_bz(a1,a2,grid_size=150,center_rec=(0.0,0.5), radius=0.5)

kpoints_to_write = sample_and_add_kz(all_pts,10)

#write those 3d k vectors to KPOINTS file
write_kpoints('KPOINTS',kpoints_to_write)

