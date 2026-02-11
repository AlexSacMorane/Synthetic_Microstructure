#-------------------------------------------------------------------------------
# Librairies
#-------------------------------------------------------------------------------

import skfmm, pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import label

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
dim_interface = 4 # -
radius_snake = 1.5*dim_interface # -
number_snakes = 9 # -

sample_id = '01'

#-------------------------------------------------------------------------------
# Generate the binary microstructure
#-------------------------------------------------------------------------------

# generate the sample
M_bin = np.ones((dim_sample, dim_sample, dim_sample))
# init the position of the snake
L_center_snake = None
count = 0
# the snake crawls until the porosity is reached
while np.sum(M_bin)/(dim_sample**3) > 1-porosity:
    if L_center_snake is None:
        # generate a random position for the head of the snake
        L_center_snake = []
        for i_snake in range(number_snakes):
            if i_snake%3 == 0:
                # face x-
                center_snake = np.array([0, np.random.rand()*dim_sample, np.random.rand()*dim_sample])
            elif i_snake%3 == 1:
                # face y-
                center_snake = np.array([np.random.rand()*dim_sample, 0, np.random.rand()*dim_sample])
            else:
                # face z-
                center_snake = np.array([np.random.rand()*dim_sample, np.random.rand()*dim_sample, 0])
            center_snake = center_snake.astype(int)
            L_center_snake.append(center_snake)
    else:
        for i_snake in range(number_snakes):
            center_snake = L_center_snake[i_snake]
            noise = 3
            if i_snake%3 == 0:
                displacement = np.array([1+noise, np.random.randint(0,2*noise+1), np.random.randint(0,2*noise+1)])
            elif i_snake%3 == 1:
                displacement = np.array([np.random.randint(0,2*noise+1), 1+noise, np.random.randint(0,2*noise+1)])
            else:
                displacement = np.array([np.random.randint(0,2*noise+1), np.random.randint(0,2*noise+1), 1+noise])
            displacement.astype(int)
            displacement = displacement - np.array([noise, noise, noise])
            center_snake = center_snake + displacement
            # consider the periodic condition
            # (x-axis)
            if center_snake[0] >= dim_sample :
                center_snake[0] = center_snake[0] - dim_sample
            if center_snake[0] < 0 :
                center_snake[0] = center_snake[0] + dim_sample
            # (y-axis)
            if center_snake[1] >= dim_sample :
                center_snake[1] = center_snake[1] - dim_sample
            if center_snake[1] < 0 :
                center_snake[1] = center_snake[1] + dim_sample
            # (z-axis)
            if center_snake[2] >= dim_sample :
                center_snake[2] = center_snake[2] - dim_sample
            if center_snake[2] < 0 :
                center_snake[2] = center_snake[2] + dim_sample
            # update
            L_center_snake[i_snake] = center_snake

    # count the number of steps
    count += 1
    if count % 500 == 0:
        print("Step n", count, " Current porosity:", round(1-np.sum(M_bin)/(dim_sample**3),2),'/', porosity)
    # generate the pore
    for i_snake in range(number_snakes):
        center_snake = L_center_snake[i_snake]
        for i_x in range(int(center_snake[0]-radius_snake-1), int(center_snake[0]+radius_snake+2)):
            for i_y in range(int(center_snake[1]-radius_snake-1), int(center_snake[1]+radius_snake+2)):
                for i_z in range(int(center_snake[2]-radius_snake-1), int(center_snake[2]+radius_snake+2)):
                    # compute the distance
                    dist = np.sqrt((i_x-center_snake[0])**2 + (i_y-center_snake[1])**2 + (i_z-center_snake[2])**2)
                    # if the distance is lower than the radius, assign 0 to the voxel
                    if dist < radius_snake:
                        # periodic condition (x-axis)
                        if i_x >= dim_sample :
                            i_x = i_x - dim_sample
                        if i_x < 0 :
                            i_x = i_x + dim_sample
                        # periodic condition (y-axis)
                        if i_y >= dim_sample :
                            i_y = i_y - dim_sample
                        if i_y < 0 :
                            i_y = i_y + dim_sample
                        # periodic condition (z-axis)
                        if i_z >= dim_sample :
                            i_z = i_z - dim_sample
                        if i_z < 0 :
                            i_z = i_z + dim_sample
                        # assign
                        M_bin[i_x, i_y, i_z] = 0

#-------------------------------------------------------------------------------
# Compute the sdf 
#-------------------------------------------------------------------------------

print("Compute the sdf")

# Extension of the sample (considering the periodic condition)
# this operation is conducted on the 3 axes
M_bin_extended = np.zeros((dim_sample+2*dim_interface, dim_sample+2*dim_interface, dim_sample+2*dim_interface))
# main sample
M_bin_extended[dim_interface:dim_interface+dim_sample, dim_interface:dim_interface+dim_sample, dim_interface:dim_interface+dim_sample] = M_bin
# periodicity -x
M_bin_extended[:dim_interface, dim_interface:dim_interface+dim_sample, dim_interface:dim_interface+dim_sample] = M_bin[-dim_interface:, :, :]
# periodicity +x
M_bin_extended[dim_interface+dim_sample:, dim_interface:dim_interface+dim_sample, dim_interface:dim_interface+dim_sample] = M_bin[:dim_interface, :, :]
# periodicity -y
M_bin_extended[dim_interface:dim_interface+dim_sample, :dim_interface, dim_interface:dim_interface+dim_sample] = M_bin[:, -dim_interface:, :]
# periodicity +y
M_bin_extended[dim_interface:dim_interface+dim_sample, dim_interface+dim_sample:, dim_interface:dim_interface+dim_sample] = M_bin[:, :dim_interface, :]
# periodicity -z
M_bin_extended[dim_interface:dim_interface+dim_sample, dim_interface:dim_interface+dim_sample, :dim_interface] = M_bin[:, :, -dim_interface:]
# periodicity +z
M_bin_extended[dim_interface:dim_interface+dim_sample, dim_interface:dim_interface+dim_sample, dim_interface+dim_sample:] = M_bin[:, :, :dim_interface]

# periodicity -x-y-z
M_bin_extended[:dim_interface, :dim_interface, :dim_interface] = M_bin[-dim_interface:, -dim_interface:, -dim_interface:]
# periodicity -x-y+z
M_bin_extended[:dim_interface, :dim_interface, dim_interface+dim_sample:] = M_bin[-dim_interface:, -dim_interface:, :dim_interface]
# periodicity -x+y-z
M_bin_extended[:dim_interface, dim_interface+dim_sample:, :dim_interface] = M_bin[-dim_interface:, :dim_interface, -dim_interface:]
# periodicity -x+y+z
M_bin_extended[:dim_interface, dim_interface+dim_sample:, dim_interface+dim_sample:] = M_bin[-dim_interface:, :dim_interface, :dim_interface]
# periodicity +x-y-z
M_bin_extended[dim_interface+dim_sample:, :dim_interface, :dim_interface] = M_bin[:dim_interface, -dim_interface:, -dim_interface:]
# periodicity +x-y+z
M_bin_extended[dim_interface+dim_sample:, :dim_interface, dim_interface+dim_sample:] = M_bin[:dim_interface, -dim_interface:, :dim_interface]
# periodicity +x+y-z
M_bin_extended[dim_interface+dim_sample:, dim_interface+dim_sample:, :dim_interface] = M_bin[:dim_interface, :dim_interface, -dim_interface:]
# periodicity +x+y+z
M_bin_extended[dim_interface+dim_sample:, dim_interface+dim_sample:, dim_interface+dim_sample:] = M_bin[:dim_interface, :dim_interface, :dim_interface]

# compute the sdf on the extended sample
M_sd_extended = skfmm.distance(M_bin_extended-0.5, dx = np.array([1, 1, 1]))

# extract the sdf for the original sample
M_sd = M_sd_extended[dim_interface:dim_interface+dim_sample, dim_interface:dim_interface+dim_sample, dim_interface:dim_interface+dim_sample]

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
with open('fft/snake/dict_fft_'+sample_id, 'wb') as handle:
    pickle.dump(dict_fft, handle, protocol=pickle.HIGHEST_PROTOCOL)

# vtk file
# change the array structure to verify the function
Microstructure_vtk = np.transpose(Microstructure, (2, 1, 0))
write_vtk_structured_points('vtk/snake/snake_'+sample_id+'.vtk', Microstructure_vtk, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), binary=False)  

#-------------------------------------------------------------------------------
# Check the connectivity and extract the connected pores
#-------------------------------------------------------------------------------

print("Check the pore connectivity")

# extend the sample (considering the periodic condition)
# this operation is conducted on the 3 axes
M_bin_extended_x = np.zeros((dim_sample, dim_sample+2*dim_sample, dim_sample+2*dim_sample))
for c in range(3):
    for h in range(3):
        M_bin_extended_x[:, c*dim_sample:(c+1)*dim_sample, h*dim_sample:(h+1)*dim_sample] = M_bin
M_bin_extended_y = np.zeros((dim_sample+2*dim_sample, dim_sample, dim_sample+2*dim_sample))
for l in range(3):
    for h in range(3):
        M_bin_extended_y[l*dim_sample:(l+1)*dim_sample, :, h*dim_sample:(h+1)*dim_sample] = M_bin
M_bin_extended_z = np.zeros((dim_sample+2*dim_sample, dim_sample+2*dim_sample, dim_sample))
for l in range(3):
    for c in range(3):
        M_bin_extended_z[l*dim_sample:(l+1)*dim_sample, c*dim_sample:(c+1)*dim_sample, :] = M_bin

# label the pores in the extended samples
labelled_image_x, num_features = label(1-M_bin_extended_x)
labelled_image_y, num_features = label(1-M_bin_extended_y)
labelled_image_z, num_features = label(1-M_bin_extended_z)

# extract sample from the labelled images
extracted_labelled_image_x = labelled_image_x[:, dim_sample:2*dim_sample, dim_sample:2*dim_sample]
extracted_labelled_image_y = labelled_image_y[dim_sample:2*dim_sample, :, dim_sample:2*dim_sample]
extracted_labelled_image_z = labelled_image_z[dim_sample:2*dim_sample, dim_sample:2*dim_sample, :]

# extract the faces
face_xm = extracted_labelled_image_x[0, :, :]
face_xp = extracted_labelled_image_x[-1, :, :]
face_ym = extracted_labelled_image_y[:, 0, :]
face_yp = extracted_labelled_image_y[:, -1, :]
face_zm = extracted_labelled_image_z[:, :, 0]
face_zp = extracted_labelled_image_z[:, :, -1]

# read the unique values on each face (ignore the background 0)
# check if a common id is present on each pair of faces
# create a microstructure with only the connected pores
M_inv_bin_extended_connected_x = np.zeros_like(M_bin_extended_x)
connected_x = False
for id_m in np.unique(face_xm)[1:]:
    if id_m in np.unique(face_xp)[1:]:
        connected_x = True
        M_inv_bin_extended_connected_x = M_inv_bin_extended_connected_x + (labelled_image_x == id_m).astype(int)
M_inv_bin_extended_connected_y = np.zeros_like(M_bin_extended_y)
connected_y = False
for id_m in np.unique(face_ym)[1:]:
    if id_m in np.unique(face_yp)[1:]:
        connected_y = True
        M_inv_bin_extended_connected_y = M_inv_bin_extended_connected_y + (labelled_image_y == id_m).astype(int)
M_inv_bin_extended_connected_z = np.zeros_like(M_bin_extended_z)
connected_z = False
for id_m in np.unique(face_zm)[1:]:
    if id_m in np.unique(face_zp)[1:]:
        connected_z = True  
        M_inv_bin_extended_connected_z = M_inv_bin_extended_connected_z + (labelled_image_z == id_m).astype(int)
if not(connected_x and connected_y and connected_z):
    print('not connected pores in all directions')

#-------------------------------------------------------------------------------
# Check the connectivity and extract the connected solids
#-------------------------------------------------------------------------------

# label the solids in the extended samples
labelled_image_x, num_features = label(M_bin_extended_x)
labelled_image_y, num_features = label(M_bin_extended_y)
labelled_image_z, num_features = label(M_bin_extended_z)

# extract sample from the labelled images
extracted_labelled_image_x = labelled_image_x[:, dim_sample:2*dim_sample, dim_sample:2*dim_sample]
extracted_labelled_image_y = labelled_image_y[dim_sample:2*dim_sample, :, dim_sample:2*dim_sample]
extracted_labelled_image_z = labelled_image_z[dim_sample:2*dim_sample, dim_sample:2*dim_sample, :]

# extract the faces
face_xm = extracted_labelled_image_x[0, :, :]
face_xp = extracted_labelled_image_x[-1, :, :]
face_ym = extracted_labelled_image_y[:, 0, :]
face_yp = extracted_labelled_image_y[:, -1, :]
face_zm = extracted_labelled_image_z[:, :, 0]
face_zp = extracted_labelled_image_z[:, :, -1]

# read the unique values on each face (ignore the background 0)
# check if a common id is present on each pair of faces
# create a microstructure with only the connected solids
M_bin_extended_connected_x = np.zeros_like(M_bin_extended_x)
connected_x = False
for id_m in np.unique(face_xm)[1:]:
    if id_m in np.unique(face_xp)[1:]:
        connected_x = True
        M_bin_extended_connected_x = M_bin_extended_connected_x + (labelled_image_x == id_m).astype(int)
M_bin_extended_connected_y = np.zeros_like(M_bin_extended_y)
connected_y = False
for id_m in np.unique(face_ym)[1:]:
    if id_m in np.unique(face_yp)[1:]:
        connected_y = True
        M_bin_extended_connected_y = M_bin_extended_connected_y + (labelled_image_y == id_m).astype(int)
M_bin_extended_connected_z = np.zeros_like(M_bin_extended_z)
connected_z = False
for id_m in np.unique(face_zm)[1:]:
    if id_m in np.unique(face_zp)[1:]:
        connected_z = True  
        M_bin_extended_connected_z = M_bin_extended_connected_z + (labelled_image_z == id_m).astype(int)
if not(connected_x and connected_y and connected_z):
    print('not connected solids in all directions')

#-------------------------------------------------------------------------------
# Minkowski functionals
#-------------------------------------------------------------------------------

print("\nCompute the Minkowski functionals")

M0, M1, M2, M3 = compute_minkowski(M_bin)

print(f'M0 (porosity) = {M0:.3f}, M1 (specific surface area) = {M1:.3e}, M2 (mean grain size) = {M2:.3e}, M3 (Euler characteristic) = {M3:.3e} \n')

#-------------------------------------------------------------------------------
# fmm
#-------------------------------------------------------------------------------

# compute the geometrical tortuosities 
print("Computing tortuosity on x for solid")
tau_x = compute_tortuosity_fast_marching(M_bin_extended_connected_x, extraction=[dim_sample, 2*dim_sample, dim_sample, 2*dim_sample], dx=1.0, neighborhood=6) 
print("Computing tortuosity on y for solid")
tau_y = compute_tortuosity_fast_marching(np.transpose(M_bin_extended_connected_y, (1, 2, 0)), extraction=[dim_sample, 2*dim_sample, dim_sample, 2*dim_sample], dx=1.0, neighborhood=6) 
print("Computing tortuosity on z for solid")
tau_z = compute_tortuosity_fast_marching(np.transpose(M_bin_extended_connected_z, (2, 0, 1)), extraction=[dim_sample, 2*dim_sample, dim_sample, 2*dim_sample], dx=1.0, neighborhood=6) 
print(f"tau_x={tau_x:.3f}, tau_y={tau_y:.3f}, tau_z={tau_z:.3f}\n")

print("Computing tortuosity on x for pore")
tau_x = compute_tortuosity_fast_marching(M_inv_bin_extended_connected_x, extraction=[dim_sample, 2*dim_sample, dim_sample, 2*dim_sample], dx=1.0, neighborhood=6) 
print("Computing tortuosity on y for pore")
tau_y = compute_tortuosity_fast_marching(np.transpose(M_inv_bin_extended_connected_y, (1, 2, 0)), extraction=[dim_sample, 2*dim_sample, dim_sample, 2*dim_sample], dx=1.0, neighborhood=6) 
print("Computing tortuosity on z for pore")
tau_z = compute_tortuosity_fast_marching(np.transpose(M_inv_bin_extended_connected_z, (2, 0, 1)), extraction=[dim_sample, 2*dim_sample, dim_sample, 2*dim_sample], dx=1.0, neighborhood=6) 
print(f"tau_x={tau_x:.3f}, tau_y={tau_y:.3f}, tau_z={tau_z:.3f}")