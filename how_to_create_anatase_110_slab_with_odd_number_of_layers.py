from ase.spacegroup import crystal
from ase.build import surface
import numpy as np
from ase import Atom

# lattice parameters
a = 4.6
c = 2.95

# define the lattice
atoms = crystal(['Ti', 'O'], basis=[(0, 0, 0), (0.3, 0.3, 0.0)],
                spacegroup=136, cellpar=[a, a, c, 90, 90, 90])

# cut a surface based on the lattice
s1 = surface(lattice=atoms,indices=(1,1,0),layers=5, vacuum=5)

# there is a problem for the resulting atoms in s1, we have to adjust the positions of some atoms in the slab s1

# remove the uppermost atoms, which are redundant
maxz = np.max(s1.get_positions()) - 0.00001 #add a small number to avoid precision problem
s1 = s1[[atom.position[2] < maxz for atom in s1]]

# add a new O atom at the bottom of the slab
Ti_ref = s1[25] # reference atom for the calculation of the connecting vector
O_ref = s1[20] 
displace_vec = O_ref.position - Ti_ref.position

# the new O atom will be placed below this Ti_operate atom
Ti_operate = s1[1]
O_added = Atom("O",tuple(Ti_operate.position + displace_vec))
s1.append(O_added)

# save the 110 slab
s1.write("rutile_surface.vasp")
