#-------------------------------------------------------------------------------
# Librairies
#-------------------------------------------------------------------------------

import skfmm, pickle, scipy
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import porespy as ps

import SimpleITK as sitk

from scipy.ndimage import label

#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------

from fast_marching_tortuosity import compute_tortuosity_fast_marching
from numpy_to_vtk import write_vtk_structured_points
from compute_minkowski import compute_minkowski

#-------------------------------------------------------------------------------

def load_itk(filename):
    '''
    This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.

    # Source - https://stackoverflow.com/a/42594949
    # Posted by savfod
    # Retrieved 2026-03-11, License - CC BY-SA 3.0
    '''
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing

#-------------------------------------------------------------------------------
# User
#-------------------------------------------------------------------------------

#database available
#https://www.imperial.ac.uk/earth-science/research/research-groups/pore-scale-modelling/micro-ct-images-and-networks/ 

dim_sample = 150 # -
dim_interface = 4 # -

namefile = 'Bentheimer_1000c_3p0035um'
sample_id = '00'

#-------------------------------------------------------------------------------
# pp scans
#-------------------------------------------------------------------------------

# modify the microsctructure to obtain a given porosity
pp = True 

if pp:
    porosity = 0.3

#-------------------------------------------------------------------------------
# Read ct-scans
#-------------------------------------------------------------------------------

M_bin, origin, spacing = load_itk('Imperial/'+namefile+'.mhd')

print(M_bin.shape)

#-------------------------------------------------------------------------------
# Extract and Plot
#-------------------------------------------------------------------------------

if not pp:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(16,9), num=1)
    ax1.imshow(M_bin[:, :, int(M_bin.shape[2]/2)])
    ax2.imshow(M_bin[:, int(M_bin.shape[2]/2), :])
    ax3.imshow(M_bin[int(M_bin.shape[2]/2), :, :])
    plt.savefig('Imperial/plot_'+namefile+'.png')
    plt.close()

#-------------------------------------------------------------------------------
# Reduce the size of the array for a 150x150x150
#-------------------------------------------------------------------------------

reduction_factor = min(M_bin.shape[0]/150, M_bin.shape[1]/150, M_bin.shape[2]/150)
reduction_factor = int(reduction_factor)

i_start = int((M_bin.shape[0]/reduction_factor-150)/2)
j_start = int((M_bin.shape[1]/reduction_factor-150)/2)
k_start = int((M_bin.shape[1]/reduction_factor-150)/2)

M_bin_cond = np.zeros((150, 150, 150), dtype='float')
for i in range(150):
    for j in range(150):
        for k in range(150):
            M_extract = M_bin[i_start+i*reduction_factor:i_start+(i+1)*reduction_factor,
                              j_start+j*reduction_factor:j_start+(j+1)*reduction_factor,
                              k_start+k*reduction_factor:k_start+(k+1)*reduction_factor]
            if np.mean(M_extract) > 0.5:
                M_bin_cond[i, j, k] = 1
            else: 
                M_bin_cond[i, j, k] = 0 

if not pp:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(16,9), num=1)
    ax1.imshow(M_bin_cond[:, :, int(M_bin_cond.shape[2]/2)])
    ax2.imshow(M_bin_cond[:, int(M_bin_cond.shape[2]/2), :])
    ax3.imshow(M_bin_cond[int(M_bin_cond.shape[2]/2), :, :])
    plt.savefig('Imperial/plot_cond_'+namefile+'.png')
    plt.close()

print(M_bin_cond.shape)

#-------------------------------------------------------------------------------
# Read the binary microstructure
#-------------------------------------------------------------------------------

M_bin = M_bin_cond.copy()

#-------------------------------------------------------------------------------
# Apply erosion/dilation algorithm to reach a given porosity
#-------------------------------------------------------------------------------

if pp:
    # define the elementary structure
    struct = np.array(np.ones((2,2,2)))
    counter = 0

    # erosion
    if 1 - np.sum(M_bin)/dim_sample**3 < porosity:
        while 1 - np.sum(M_bin)/dim_sample**3 < porosity:
            M_bin_eroded = scipy.ndimage.binary_erosion(M_bin, structure=struct, border_value=1).astype(M_bin.dtype)
            M_available = M_bin-M_bin_eroded # map of available seed 
            i_x, i_y, i_z = np.where(M_available==1) # extract the seed
            L_i = np.random.choice(np.arange(len(i_x)), size=100,replace=False) # pick randomly a seed
            for i in L_i:
                M_bin[i_x[i], i_y[i], i_z[i]] = 0 # erode
            counter = counter + 1
            if counter % 500 == 0:
                print("Erosion operations:", counter, " Current porosity:", round(1-np.sum(M_bin)/(dim_sample**3),2),'/', porosity)
        print(counter, 'erosion operations to reach the porosity')
    # dilation
    else :
        while porosity < 1 - np.sum(M_bin)/dim_sample**3:
            M_bin_dilated = scipy.ndimage.binary_dilation(M_bin, structure=struct).astype(M_bin.dtype)
            M_available = M_bin_dilated-M_bin # map of available seed 
            i_x, i_y, i_z = np.where(M_available==1) # extract the seed
            L_i = np.random.choice(np.arange(len(i_x)), size=100,replace=False) # pick randomly a seed
            for i in L_i:
                M_bin[i_x[i], i_y[i], i_z[i]] = 1 # dilate
            counter = counter + 1
            if counter % 500 == 0:
                print("Dilation operations:", counter, " Current porosity:", round(1-np.sum(M_bin)/(dim_sample**3),2),'/', porosity)
            counter = counter + 1
        print(counter, 'dilation operations to reach the porosity')
    
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

if not pp:
    # save
    dict_fft = {'M_microstructure': Microstructure}
    with open('fft/Imperial/dict_fft_'+ sample_id, 'wb') as handle:
        pickle.dump(dict_fft, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # vtk file
    # change the array structure to verify the function
    Microstructure_vtk = np.transpose(Microstructure, (2, 1, 0))
    write_vtk_structured_points('vtk/Imperial/imp_'+ namefile +'_' + sample_id + '.vtk', Microstructure_vtk, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), binary=False)  

else:
    # save
    dict_fft = {'M_microstructure': Microstructure}
    with open('fft/Imperial_pp/dict_fft_'+ sample_id, 'wb') as handle:
        pickle.dump(dict_fft, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # vtk file
    # change the array structure to verify the function
    Microstructure_vtk = np.transpose(Microstructure, (2, 1, 0))
    write_vtk_structured_points('vtk/Imperial_pp/imp_pp_'+ namefile +'_' + sample_id + '.vtk', Microstructure_vtk, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), binary=False)  

#-------------------------------------------------------------------------------
# Check the connectivity and extract the connected pores
#-------------------------------------------------------------------------------

print("Check the pore connectivity")

# Extension of the sample (considering the periodic condition)
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

print("Check the solid connectivity")

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

print("Computing the Minkowski functionals")

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
