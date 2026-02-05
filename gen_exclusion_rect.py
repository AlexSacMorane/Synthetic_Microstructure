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
# User
#-------------------------------------------------------------------------------

dim_sample = 250 # -
porosity = 0.2 # -
dim_interface = 8 # -

sample_id = '01'

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------

# generate the pore (0)
M_bin = np.zeros((dim_sample, dim_sample, dim_sample))

# determine the size of the solid inclusion
dim_solid = ((1-porosity) * dim_sample**2)**(1/2)
# determine the offset to centerized the solid
off_set = int((dim_sample - dim_solid)/2)

# generate the solid (1)
for i_x in range(off_set, off_set + int(dim_solid)):
    for i_y in range(off_set, off_set + int(dim_solid)):
        M_bin[i_x, i_y, :] = 1

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
with open('fft/rect_excl_dict_fft_'+sample_id, 'wb') as handle:
    pickle.dump(dict_fft, handle, protocol=pickle.HIGHEST_PROTOCOL)

# vtk file
# change the array structure to verify the function
Microstructure_vtk = np.transpose(Microstructure, (2, 1, 0))
write_vtk_structured_points('vtk/rect_excl_'+sample_id+'.vtk', Microstructure_vtk, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), binary=False)  

#-------------------------------------------------------------------------------
# Minkowski functionals
#-------------------------------------------------------------------------------

M0, M1, M2, M3 = compute_minkowski(M_bin)

print(f'M0 (porosity) = {M0:.3f}, M1 (specific surface area) = {M1:.3f}, M3 (Euler characteristic) = {M3:.3f} \n')

#-------------------------------------------------------------------------------
# fmm
#-------------------------------------------------------------------------------

print("Computing tortuosity on z for solid")
tau_z = compute_tortuosity_fast_marching(np.transpose(M_bin, (2, 0, 1)), extraction=[0, dim_sample, 0, dim_sample], dx=1.0, neighborhood=6) 
print(f"tau_z={tau_z:.3f}")


print("Computing tortuosity on x for pore")
tau_x = compute_tortuosity_fast_marching(1-M_bin, extraction=[0, dim_sample, 0, dim_sample], dx=1.0, neighborhood=6) 
print("Computing tortuosity on y for pore")
tau_y = compute_tortuosity_fast_marching(np.transpose(1-M_bin, (1, 2, 0)), extraction=[0, dim_sample, 0, dim_sample], dx=1.0, neighborhood=6) 
print("Computing tortuosity on z for pore")
tau_z = compute_tortuosity_fast_marching(np.transpose(1-M_bin, (2, 0, 1)), extraction=[0, dim_sample, 0, dim_sample], dx=1.0, neighborhood=6) 
print(f"tau_x={tau_x:.3f}, tau_y={tau_y:.3f}, tau_z={tau_z:.3f}\n")

