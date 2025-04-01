import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from shapely.geometry import Polygon

# Define real-space lattice vectors for graphene.
#a1 = np.array([2.1390827473, -1.235])
#a2 = np.array([2.1390827473, 1.235])
a1 = np.array([2.45	,0.00])
a2 = np.array([-1.225,	2.1217622395])

# Calculate the area of the unit cell.
area = np.abs(a1[0]*a2[1] - a1[1]*a2[0])

# Compute reciprocal lattice vectors.
b1 = 2 * np.pi * np.array([-a2[1], a2[0]]) / area
b2 = 2 * np.pi * np.array([a1[1], -a1[0]]) / area

# Generate a grid of reciprocal lattice points.
rec_points = []
n_range = 2  # use a small range to cover nearby points
for i in range(-n_range, n_range+1):
    for j in range(-n_range, n_range+1):
        rec_points.append(i * b1 + j * b2)
rec_points = np.array(rec_points)

# Compute the Voronoi diagram of the reciprocal lattice.
vor = Voronoi(rec_points)

# Find the Voronoi region corresponding to the origin.
origin = np.array([0, 0])
region_index = None
for point_idx, region in enumerate(vor.point_region):
    if np.allclose(rec_points[point_idx], origin):
        region_index = region
        break

if region_index is None:
    raise ValueError("Origin not found in the list of reciprocal points!")

# Get and order the vertices of the Voronoi region.
vertices = vor.vertices[vor.regions[region_index]]
poly = Polygon(vertices)
ordered_vertices = np.array(poly.exterior.coords)

# Prepare the 2x2 matrix of reciprocal basis vectors (as columns)
B = np.column_stack((b1, b2))  # B is 2x2 with b1 and b2 as columns

# Plot the Brillouin zone.
plt.figure(figsize=(6,6))
plt.fill(ordered_vertices[:, 0], ordered_vertices[:, 1],
         color='lightblue', alpha=0.5, label="1st Brillouin Zone")
plt.plot(ordered_vertices[:, 0], ordered_vertices[:, 1], 'b-')
plt.plot(0, 0, 'ro', label="Origin")

# Plot and annotate the reciprocal lattice basis vectors.
plt.arrow(0, 0, b1[0], b1[1],
          head_width=0.15, head_length=0.15,
          fc='green', ec='green', length_includes_head=True)
plt.text(b1[0]*1.1, b1[1]*1.1, 'b1', color='green', fontsize=12)

plt.arrow(0, 0, b2[0], b2[1],
          head_width=0.15, head_length=0.15,
          fc='purple', ec='purple', length_includes_head=True)
plt.text(b2[0]*1.1, b2[1]*1.1, 'b2', color='purple', fontsize=12)

# Annotate each vertex with its reciprocal-basis coordinates.
n_unique = len(ordered_vertices) - 1  # exclude duplicate of first vertex
for v in ordered_vertices[:n_unique]:
    c = np.linalg.solve(B, v)  # Solve for [c1, c2] such that v = c1*b1 + c2*b2.
    label = f"({c[0]:.2f}, {c[1]:.2f})"
    plt.text(v[0], v[1], label, fontsize=10, color='red',
             ha='center', va='center')

# Compute and annotate midpoints between consecutive vertices.
for i in range(n_unique):
    v1 = ordered_vertices[i]
    v2 = ordered_vertices[(i+1) % n_unique]
    midpoint = (v1 + v2) / 2.0
    # Compute the edge vector and a perpendicular unit vector.
    edge_vec = v2 - v1
    norm_edge = np.linalg.norm(edge_vec)
    if norm_edge != 0:
        perp_vec = np.array([-edge_vec[1], edge_vec[0]])
        perp_vec = perp_vec / np.linalg.norm(perp_vec)
    else:
        perp_vec = np.array([0, 0])
    offset_mag = 0.1  # Adjust this for desired displacement.
    offset = offset_mag * perp_vec

    c_mid = np.linalg.solve(B, midpoint)
    label_mid = f"({c_mid[0]:.2f}, {c_mid[1]:.2f})"
    plt.plot(midpoint[0], midpoint[1], 'ko')
    plt.text(midpoint[0] + offset[0], midpoint[1] + offset[1],
             label_mid, fontsize=10, color='black', ha='center', va='center')

# === New Block: Plotting user-specified points ===
# Define user provided points in reciprocal-basis coordinates.
# For example, these three points are given relative to the reciprocal basis:
user_points_reciprocal = np.array([
[0,  0]
#  [0.33,  0.000000],
#  [0.33,  0.17],
#  [0.500000,  0.17]

])
# Convert user points to Cartesian coordinates:
user_points_cartesian = np.dot(user_points_reciprocal, B.T)
# Plot the user points and label them.
for pt_rec, pt_cart in zip(user_points_reciprocal, user_points_cartesian):
    plt.plot(pt_cart[0], pt_cart[1], 'm*', markersize=10)
    label_user = f"({pt_rec[0]:.2f}, {pt_rec[1]:.2f})"
    # Displace the text slightly (here we add 0.05 in both directions)
    plt.text(pt_cart[0] + 0.05, pt_cart[1] + 0.05,
             label_user, color='magenta', fontsize=10)

plt.title("2D Brillouin Zone for Graphene\nwith Reciprocal Basis Coordinates, Midpoints, and User Points")
plt.xlabel("$k_x$")
plt.ylabel("$k_y$")
plt.axis('equal')
plt.legend()
plt.show()
