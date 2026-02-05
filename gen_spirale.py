#-------------------------------------------------------------------------------
# Librairies
#-------------------------------------------------------------------------------

import skfmm, pickle
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------

from numpy_to_vtk import write_vtk_structured_points
from fast_marching_tortuosity import compute_tortuosity_fast_marching
from compute_minkowski import compute_minkowski

#-------------------------------------------------------------------------------

def distance_to_spiral(t, x, y, z, R_spiral, N, dim_sample):
    '''
    Compute the distance of a point (x,y,z) to a spiral parameterized by t and defined by a radius R_spiral, the number of turns N, and a height of dim_sample.
    
    Several distances are obtained for different values of t. The minimum distance is returned.
    '''
    # init
    dist2 = []
    # iterate over t
    for ti in t:
        # compute the position along the spiral
        x_spiral = R_spiral*np.cos(ti) + dim_sample/2
        y_spiral = R_spiral*np.sin(ti) + dim_sample/2
        z_spiral = (dim_sample/(2*np.pi*N))*ti
        # compute the distance from the point to the spiral
        dist_i2 = (x - x_spiral)**2 + (y - y_spiral)**2 + (z - z_spiral)**2
        dist2.append(dist_i2)    

    return min(dist2)**0.5

#-------------------------------------------------------------------------------
# User
#-------------------------------------------------------------------------------

dim_sample = 250 # -
porosity = 0.2 # -
dim_interface = 4 # -
tortuosity_spiral = 1.1 # -

sample_id = '00'

#-------------------------------------------------------------------------------
# Generate the binary microstructure
#-------------------------------------------------------------------------------

# compute the linear size of the spiral
L_spiral = dim_sample*tortuosity_spiral

# iterate on the number of turns
controler = False
# number of turns of the spiral
N = 1
while not controler:
    # compute the radius of the spiral
    R_spiral = (L_spiral**2 - dim_sample**2)**0.5/2/np.pi/N
    # compute the section area of the spiral
    A_spiral = porosity*(dim_sample**3)/L_spiral 
    # compute thr radius of the section
    R_section = (A_spiral/np.pi)**0.5
    if R_section < dim_interface:
        raise ValueError('the tortuosity is too large')
    # control
    if 2*(R_spiral + R_section) < dim_sample:
        controler = True
    else:
        N += 1

# initialize the list of t
t = np.linspace(-2*np.pi, N*2*np.pi+2*np.pi, (N+1)*50)

# generate the sample
M_bin = np.ones((dim_sample, dim_sample, dim_sample))
for i_x in range(dim_sample):
    for i_y in range(dim_sample):
        for i_z in range(dim_sample):
            # user interface
            if i_z + i_y*dim_sample + i_x*dim_sample**2 % 1000 == 0:
                print(round((i_z + i_y*dim_sample + i_x*dim_sample**2)/dim_sample**3*100, 2), '% done')

            # compute the distance to the spiral
            dist = distance_to_spiral(t, i_x, i_y, i_z, R_spiral, N, dim_sample)
            # check if the point is inside the spiral section
            if dist <= R_section:
                M_bin[i_x, i_y, i_z] = 0    

# verify the porosity with the binary 
print(round(1-np.sum(M_bin)/(dim_sample**3),2), '/', porosity)

#-------------------------------------------------------------------------------
# Compute the sdf 
#-------------------------------------------------------------------------------

# Extension of the sample (considering the periodic condition)
M_bin_extended = np.zeros((dim_sample, dim_sample, dim_sample+2*dim_sample))
for h in range(3):
    M_bin_extended[:, :, h*dim_sample:(h+1)*dim_sample] = M_bin

# compute the sdf on the extended sample
M_sd_extended = skfmm.distance(M_bin_extended-0.5, dx = np.array([1, 1, 1]))

# extract the sdf for the original sample
M_sd = M_sd_extended[:, :, dim_sample:2*dim_sample]

#-------------------------------------------------------------------------------
# Compute the microstructure
#-------------------------------------------------------------------------------

Microstructure = np.zeros((dim_sample, dim_sample, dim_sample))
for i_x in range(dim_sample):
    for i_y in range(dim_sample):
        for i_z in range(dim_sample):
            if M_sd[i_x, i_y, i_z] > dim_interface/2: # inside the grain
                Microstructure[i_x, i_y, i_z] = 1
            elif M_sd[i_x, i_y, i_z] < -dim_interface/2: # outside the grain
                Microstructure[i_x, i_y, i_z] = 0
            else : # in the interface
                Microstructure[i_x, i_y, i_z] = 0.5 + M_sd[i_x, i_y, i_z]/dim_interface
                
# check the porosity
print(round(1-np.sum(Microstructure)/(dim_sample**3),2), '/', porosity)

#-------------------------------------------------------------------------------
# Output
#-------------------------------------------------------------------------------

# plot
#fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(16,9),num=1)
#ax1.imshow(Microstructure[int(dim_sample//2), :, :], cmap='Greys', vmin=0, vmax=1)
#ax1.set_xlabel('Z axis')
#ax1.set_ylabel('Y axis')
#ax2.imshow(Microstructure[:, int(dim_sample//2), :], cmap='Greys', vmin=0, vmax=1)
#ax2.set_xlabel('Z axis')
#ax2.set_ylabel('X axis')
#ax3.imshow(Microstructure[:, :, int(dim_sample//2)], cmap='Greys', vmin=0, vmax=1)
#ax3.set_xlabel('Y axis')
#ax3.set_ylabel('X axis')
#plt.savefig('plot_microstructure.png')
#plt.close()

# save
dict_fft = {'M_microstructure': Microstructure}
with open('fft/spiral_dict_fft_'+sample_id, 'wb') as handle:
    pickle.dump(dict_fft, handle, protocol=pickle.HIGHEST_PROTOCOL)

# vtk file
# change the array structure to verify the function
Microstructure_vtk = np.transpose(Microstructure, (2, 1, 0))
write_vtk_structured_points('vtk/spiral_'+sample_id+'.vtk', Microstructure_vtk, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), binary=False)  

#-------------------------------------------------------------------------------
# Minkowski functionals
#-------------------------------------------------------------------------------

M0, M1, M2, M3 = compute_minkowski(M_bin)

print(f'M0 (porosity) = {M0:.3f}, M1 (specific surface area) = {M1:.3f}, M3 (Euler characteristic) = {M3:.3f} \n')

#-------------------------------------------------------------------------------
# fmm
#-------------------------------------------------------------------------------

# compute the geometrical tortuosities 
print("Computing tortuosity on x for solid")
tau_x = compute_tortuosity_fast_marching(M_bin, extraction=[0, dim_sample, 0, dim_sample], dx=1.0, neighborhood=6) 
print("Computing tortuosity on y for solid")
tau_y = compute_tortuosity_fast_marching(np.transpose(M_bin, (1, 2, 0)), extraction=[0, dim_sample, 0, dim_sample], dx=1.0, neighborhood=6) 
print("Computing tortuosity on z for solid")
tau_z = compute_tortuosity_fast_marching(np.transpose(M_bin, (2, 0, 1)), extraction=[0, dim_sample, 0, dim_sample], dx=1.0, neighborhood=6) 
print(f"tau_x={tau_x:.3f}, tau_y={tau_y:.3f}, tau_z={tau_z:.3f}\n")

print("Computing tortuosity on z for pore")
tau_z = compute_tortuosity_fast_marching(np.transpose(1-M_bin, (2, 0, 1)), extraction=[0, dim_sample, 0, dim_sample], dx=1.0, neighborhood=6) 
print(f"tau_z={tau_z:.3f}")