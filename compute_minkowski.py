#-------------------------------------------------------------------------------
# Librairies
#-------------------------------------------------------------------------------

import numpy as np
import skimage

#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------

def compute_minkowski(M_bin):
    '''
    Compute the Minkowski functionals for a binary microstructure M_bin (0: pore, 1: solid).
    
    M0 is the porosity.
    M1 is the specific surface area.
    M2 is the mean curvature.
    M3 is the Euler characteristic.
    '''
    # extract size 
    dim_sample = M_bin.shape[0]

    # M0: porosity
    M0 = 1 - np.sum(M_bin)/dim_sample**3

    # M1: specific surface area
    # extraction of the surface
    verts, faces, normals, values = skimage.measure.marching_cubes(M_bin, level=0.5, spacing=(1.0, 1.0, 1.0))
    # compute of the surface
    surface_area = skimage.measure.mesh_surface_area(verts, faces)
    # normalize by the volume
    M1 = surface_area / (dim_sample**3)

    # M2: mean curvature
    # to do
    M2 = 0

    # M3: Euler characteristic 
    euler = skimage.measure.euler_number(M_bin, connectivity=1)
    # normalize by the volume
    M3 = euler / (dim_sample**3)

    return M0, M1, M2, M3