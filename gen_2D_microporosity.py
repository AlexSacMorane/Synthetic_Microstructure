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

dim_sample = 1800 # -
dim_interface = 4 # -
micro_porosity = 0.8 # -
f_microporosity = 20

sample_id = '00'

#-------------------------------------------------------------------------------
# Generate the binary microstructure
#-------------------------------------------------------------------------------

# generate the sample
M_bin = np.ones((dim_sample, dim_sample))

# compute the size of the pore
dim_pore = dim_interface*75

# generate the macro porosity
M_bin[:int(M_bin.shape[1]/2-dim_pore/2), 2*dim_interface:2*dim_interface+dim_pore] = 0
M_bin[-int(M_bin.shape[1]/2-dim_pore/2):, -dim_pore-2*dim_interface:-2*dim_interface] = 0
M_bin[int(M_bin.shape[1]/2-dim_pore/2):-int(M_bin.shape[1]/2-dim_pore/2), 2*dim_interface:-2*dim_interface] = 0

# compute the macro porosity
macro_porosity = 1-np.sum(M_bin)/M_bin.size
print('macro por', round(macro_porosity,2))
      
# plot
fig, (ax1) = plt.subplots(1,1, figsize=(16,9),num=1)
ax1.imshow(M_bin[:, :], cmap='Greys', vmin=0, vmax=1)
plt.savefig('vtk/2D_microporosity/'+sample_id+'.png')
plt.close()

#-------------------------------------------------------------------------------
# Compute the sdf 
#-------------------------------------------------------------------------------

print("Compute the sdf")

# Extension of the sample (considering the periodic condition)
# this operation is conducted on the 2 axes
M_bin_extended = np.zeros((dim_sample+2*dim_interface, dim_sample+2*dim_interface))
# main sample
M_bin_extended[dim_interface:dim_interface+dim_sample, dim_interface:dim_interface+dim_sample] = M_bin
# periodicity -x
M_bin_extended[:dim_interface, dim_interface:dim_interface+dim_sample] = M_bin[-dim_interface:, :]
# periodicity +x
M_bin_extended[dim_interface+dim_sample:, dim_interface:dim_interface+dim_sample] = M_bin[:dim_interface, :]
# periodicity -y
M_bin_extended[dim_interface:dim_interface+dim_sample, :dim_interface] = M_bin[:, -dim_interface:]
# periodicity +y
M_bin_extended[dim_interface:dim_interface+dim_sample, dim_interface+dim_sample:] = M_bin[:, :dim_interface]


# compute the sdf on the extended sample
M_sd_extended = skfmm.distance(M_bin_extended-0.5, dx = np.array([1, 1]))

# extract the sdf for the original sample
M_sd = M_sd_extended[dim_interface:dim_interface+dim_sample, dim_interface:dim_interface+dim_sample]

#-------------------------------------------------------------------------------
# Compute the microstructure
#-------------------------------------------------------------------------------

Microstructure = np.zeros((dim_sample, dim_sample))
for i_x in range(dim_sample):
    for i_y in range(dim_sample):
        if M_sd[i_x, i_y] > dim_interface/2: # inside the grain
            Microstructure[i_x, i_y] = 1
        elif M_sd[i_x, i_y] < -dim_interface/2: # outside the grain
            Microstructure[i_x, i_y] = 0
        else : # in the interface
            Microstructure[i_x, i_y] = 0.5 + M_sd[i_x, i_y]/dim_interface

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
with open('fft/2D_microporosity/dict_fft_'+sample_id, 'wb') as handle:
    pickle.dump(dict_fft, handle, protocol=pickle.HIGHEST_PROTOCOL)

#-------------------------------------------------------------------------------
# Check the connectivity and extract the connected pores
#-------------------------------------------------------------------------------

print("Check the pore connectivity")

# extend the sample (considering the periodic condition)
# this operation is conducted on the 2 axes
M_bin_extended_x = np.zeros((dim_sample, dim_sample+2*dim_sample))
for c in range(3):
    M_bin_extended_x[:, c*dim_sample:(c+1)*dim_sample] = M_bin
M_bin_extended_y = np.zeros((dim_sample+2*dim_sample, dim_sample))
for l in range(3):
    M_bin_extended_y[l*dim_sample:(l+1)*dim_sample, :] = M_bin

# label the pores in the extended samples
labelled_image_x, num_features = label(1-M_bin_extended_x)
labelled_image_y, num_features = label(1-M_bin_extended_y)

# extract sample from the labelled images
extracted_labelled_image_x = labelled_image_x[:, dim_sample:2*dim_sample]
extracted_labelled_image_y = labelled_image_y[dim_sample:2*dim_sample, :]

# extract the faces
face_xm = extracted_labelled_image_x[0, :]
face_xp = extracted_labelled_image_x[-1, :]
face_ym = extracted_labelled_image_y[:, 0]
face_yp = extracted_labelled_image_y[:, -1]

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
if not(connected_x and connected_y):
    print('not connected pores in all directions')

#-------------------------------------------------------------------------------
# Check the connectivity and extract the connected solids
#-------------------------------------------------------------------------------

print("Check the solid connectivity")

# label the solids in the extended samples
labelled_image_x, num_features = label(M_bin_extended_x)
labelled_image_y, num_features = label(M_bin_extended_y)

# extract sample from the labelled images
extracted_labelled_image_x = labelled_image_x[:, dim_sample:2*dim_sample]
extracted_labelled_image_y = labelled_image_y[dim_sample:2*dim_sample, :]

# extract the faces
face_xm = extracted_labelled_image_x[0, :]
face_xp = extracted_labelled_image_x[-1, :]
face_ym = extracted_labelled_image_y[:, 0]
face_yp = extracted_labelled_image_y[:, -1]

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
if not(connected_x and connected_y):
    print('not connected solids in all directions')

#-------------------------------------------------------------------------------
# Compute minkovski and tortuosity in 2D
#-------------------------------------------------------------------------------

# TO DO
