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

#-------------------------------------------------------------------------------
# User
#-------------------------------------------------------------------------------

dim_sample = 250 # -
porosity = 0.2 # -
dim_interface = 4 # -

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------

# generate the solid (1)
M_bin = np.ones((dim_sample, dim_sample, dim_sample))

# determine the size to verify the porosity
dim_pore = int((porosity * dim_sample**2)**(1/2))
# determine the offset to centerized the pore
off_set = int((dim_sample - dim_pore)/2)

# generate the pore (0)
for i_x in range(off_set, off_set + dim_pore):
    for i_y in range(off_set, off_set + dim_pore):
        M_bin[i_x, i_y, :] = 0

#-------------------------------------------------------------------------------
# Compute the sdf 
#-------------------------------------------------------------------------------

M_sd_phi = skfmm.distance(M_bin-0.5, dx = np.array([1, 1, 1]))

#-------------------------------------------------------------------------------
# Compute the microstructure
#-------------------------------------------------------------------------------

Microstructure = np.zeros((dim_sample, dim_sample, dim_sample))
for i_x in range(dim_sample):
    for i_y in range(dim_sample):
        for i_z in range(dim_sample):
            if M_sd_phi[i_x, i_y, i_z] > dim_interface/2: # inside the grain
                Microstructure[i_x, i_y, i_z] = 1
            elif M_sd_phi[i_x, i_y, i_z] < -dim_interface/2: # outside the grain
                Microstructure[i_x, i_y, i_z] = 0
            else : # in the interface
                Microstructure[i_x, i_y, i_z] = 0.5 + M_sd_phi[i_x, i_y, i_z]/dim_interface
                
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
with open('fft/rect_incl_dict_fft_00', 'wb') as handle:
    pickle.dump(dict_fft, handle, protocol=pickle.HIGHEST_PROTOCOL)

# vtk file
# change the array structure to verify the function
Microstructure_vtk = np.transpose(Microstructure, (2, 1, 0))
write_vtk_structured_points('vtk/rect_incl_0.vtk', Microstructure_vtk, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), binary=False)  

#-------------------------------------------------------------------------------
# Minkowski functionals
#-------------------------------------------------------------------------------

# to do

#-------------------------------------------------------------------------------
# fmm
#-------------------------------------------------------------------------------

print("Computing tortuosity on x for solid")
tau_x = compute_tortuosity_fast_marching(M_bin, extraction=[0, dim_sample, 0, dim_sample], dx=1.0, neighborhood=6) 
print("Computing tortuosity on y for solid")
tau_y = compute_tortuosity_fast_marching(np.transpose(M_bin, (1, 2, 0)), extraction=[0, dim_sample, 0, dim_sample], dx=1.0, neighborhood=6) 
print("Computing tortuosity on z for solid")
tau_z = compute_tortuosity_fast_marching(np.transpose(M_bin, (2, 0, 1)), extraction=[0, dim_sample, 0, dim_sample], dx=1.0, neighborhood=6) 
print(f"tau_x={tau_x:.3f}, tau_y={tau_y:.3f}, tau_z={tau_z:.3f}\n")

print("Computing tortuosity on z for pore")
tau_z = compute_tortuosity_fast_marching(1-np.transpose(M_bin, (2, 0, 1)), extraction=[0, dim_sample, 0, dim_sample], dx=1.0, neighborhood=6) 
print(f"tau_z={tau_z:.3f}")
