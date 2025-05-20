import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

def plane_wave_eigenvalues(
    heights,
    num_intervals,
    planewave_shift=0.0,
    potential=lambda z: np.zeros_like(z),
):
    """
    Compute plane?wave eigenvalues for each height in `heights`.

    Parameters
    ----------
    heights : array?like of float
        Slab heights in ?ngstr?ms at which to solve the 1D Schr?dinger equation.
    num_intervals : int
        Number of discretization intervals N (so there are N+1 grid points).
    planewave_shift : float, optional
        Constant energy shift (in eV) to subtract from the Hamiltonian.
    potential : callable, optional
        A function potential(z) giving the potential energy (in joules) at each z.

    Returns
    -------
    eigvals_dict : dict
        A dictionary mapping each height to a 1D numpy array of eigenvalues (in eV).
    """
    # Physical constants
    hbar = 1.0545718e-34       # J?s
    m_e  = 9.10938356e-31      # kg
    joule_to_eV = 1.0 / 1.60218e-19

    eigvals_dict = {}

    # Discretization parameter
    N = num_intervals
    delta = 1.0 / N

    for H in heights:
        # Build real?space grid from 0 to H (? → m)
        z = np.linspace(0, H, N + 1) * 1e-10
        dz = z[1] - z[0]

        # Finite?difference kinetic prefactor
        fac = -hbar**2 / (2 * m_e * dz**2)

        # Diagonal: kinetic + potential
        V = potential(z[1:-1])  # exclude boundaries if desired
        diag = (-2 * fac + V) * joule_to_eV

        # Off?diagonal: kinetic coupling
        offdiag = (fac * np.ones(len(diag) - 1)) * joule_to_eV

        # Assemble tridiagonal Hamiltonian
        H_mat = np.diag(diag)
        for i in range(len(offdiag)):
            H_mat[i, i+1] = offdiag[i]
            H_mat[i+1, i] = offdiag[i]

        # Apply constant shift
        H_mat -= np.eye(len(diag)) * planewave_shift

        # Diagonalize
        vals, _ = eigh(H_mat)
        eigvals_dict[H] = vals

    return eigvals_dict

# heights in ansgtrom
heights = np.linspace(5, 1000, 1000)
eig_dict = plane_wave_eigenvalues(
    heights=heights,
    num_intervals=100,
    planewave_shift=1.0,          # subtract 1 eV if you like
    potential=lambda z: 0*z       # zero potential
)

for h,vals in eig_dict.items():
    x = [h for i in range(len(vals))]
    y = vals
    plt.scatter(x,y,s=0.1)
plt.show()
