#-------------------------------------------------------------------------------
# Librairies
#-------------------------------------------------------------------------------

import numpy as np
import porespy as ps
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
    # compute the local thickness of the solid phase
    # method = 'dt' is faster but less accurate
    M_local_thickness = ps.filters.local_thickness(M_bin, method='imj')
    # compute the distribution of the local thickness
    data = ps.metrics.pore_size_distribution(M_local_thickness, bins=20, log=False)    
    # compute the average local thickness
    mean_local_thickness = np.sum(data.bin_centers*data.pdf)/np.sum(data.pdf)+data.bin_widths[-1]/2
    # normalize by the volume
    M2 = mean_local_thickness / (dim_sample**3)

    # M3: Euler characteristic 
    euler = skimage.measure.euler_number(M_bin, connectivity=1)
    # normalize by the volume
    M3 = euler / (dim_sample**3)

    return M0, M1, M2, M3