import numpy as np

def generate_gamma_centered_kmesh(N1, N2, N3):
    """
    Generate a Gamma-centered k-mesh with VASP-style shifting.
    
    Parameters:
    N1, N2, N3: int
        Number of k-points along reciprocal lattice vectors b1, b2, b3.
    
    Returns:
    kmesh: list of tuples
        List of k-points in fractional coordinates.
    """
    # Generate k-point grid for each direction with the shifted formula
    k1 = [(i - np.ceil(N1 / 2) + 1) / N1 for i in range(N1)]
    k2 = [(i - np.ceil(N2 / 2) + 1) / N2 for i in range(N2)]
    k3 = [(i - np.ceil(N3 / 2) + 1) / N3 for i in range(N3)]

    # Function to reorder the k-points in the desired order
    def reorder_k_points(k):
        k_pos = [k_val for k_val in k if k_val >= 0]  # Positive values (including 0)
        k_neg = [k_val for k_val in k if k_val < 0]   # Negative values
        return k_pos + k_neg  # First positive, then negative

    # Reorder each k-point list
    k1 = reorder_k_points(k1)
    k2 = reorder_k_points(k2)
    k3 = reorder_k_points(k3)

    # Generate all combinations of k-points with the order of z, y, and x
    kmesh = []
    for z in k3:
        for y in k2:
            for x in k1:
                kmesh.append((x, y, z))

    return kmesh

# Example usage for a 3x3x2 grid
N1, N2, N3 = 5, 5, 3
kmesh = generate_gamma_centered_kmesh(N1, N2, N3)

# Print the k-mesh in a format similar to VASP output
for k in kmesh:
    print(f"{k[0]: .6f} {k[1]: .6f} {k[2]: .6f}")

