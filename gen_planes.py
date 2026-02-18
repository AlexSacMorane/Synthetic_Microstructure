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

#-------------------------------------------------------------------------------
# User
#-------------------------------------------------------------------------------

dim_sample = 100 # -
porosity = 0.5 # -
dim_interface = 50 # -

sample_id = '05'

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------

# generate the solid (1)
M_bin = np.ones((dim_sample, dim_sample, dim_sample))

# determine the size to verify the porosity
dim_pore = int(porosity * dim_sample)
# determine the offset to centerized the pore
off_set = int((dim_sample - dim_pore)/2)

# generate the pore (0)
for i_y in range(off_set, off_set + dim_pore):
    M_bin[:, i_y, :] = 0

#-------------------------------------------------------------------------------
# Compute the sdf 
#-------------------------------------------------------------------------------

M_sd = skfmm.distance(M_bin-0.5, dx = np.array([1, 1, 1]))

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
                
#-------------------------------------------------------------------------------
# Output
#-------------------------------------------------------------------------------

# plot
fig, (ax1) = plt.subplots(1, 1, figsize=(16,9),num=1)
ax1.plot(Microstructure[int(dim_sample/2), :, int(dim_sample/2)])
ax1.set_xlabel('y axis')
ax1.set_ylabel('pixel value')
plt.savefig('png/profile_planes_'+sample_id+'.png')
plt.close()

# save
dict_fft = {'M_microstructure': Microstructure}
with open('fft/planes/dict_fft_'+sample_id, 'wb') as handle:
    pickle.dump(dict_fft, handle, protocol=pickle.HIGHEST_PROTOCOL)

# vtk file
# change the array structure to verify the function
Microstructure_vtk = np.transpose(Microstructure, (2, 1, 0))
write_vtk_structured_points('vtk/planes/planes_'+sample_id+'.vtk', Microstructure_vtk, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), binary=False)  

print(np.sum(Microstructure)/Microstructure.size)