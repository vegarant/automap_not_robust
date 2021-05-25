"""
This script read the data produced by the scripts 
'Demo_test_automap_error_map.py' and 'Demo_test_lasso_error_map.py'
and produce error maps for the reconstructed images. 
"""


from PIL import Image
from scipy.io import loadmat
import numpy as np
from os.path import join
import os, glob
from adv_tools_PNAS.adversarial_tools import l2_norm_of_tensor, cut_to_01, scale_to_01

err_func = lambda x: l2_norm_of_tensor(x)/np.sqrt(np.prod(x.shape))

HCP_nbr1 = 1002
HCP_nbr2 = 'dataset1'
HCP_nbr3 = 1004

im_nbrs1 = [49, 116, 150] 
im_nbrs2 = [49]
im_nbrs3 = []

HCP = [HCP_nbr1]*len(im_nbrs1) + [HCP_nbr2]*len(im_nbrs2) + [HCP_nbr3]*len(im_nbrs3)
im_nbrs = im_nbrs1 + im_nbrs2 + im_nbrs3

N = 128

src_data = 'data_error_map'
dest_plots = 'plots_error_map'


if not (os.path.isdir(dest_plots)):
    os.mkdir(dest_plots)

for f in glob.glob(join(dest_plots, '*')):
    os.remove(f)

data1_automap = loadmat(join(src_data, f'im_rec_automap_HCP_{HCP_nbr1}.mat'))
data2_automap = loadmat(join(src_data, f'im_rec_automap_{HCP_nbr2}.mat'))
data3_automap = loadmat(join(src_data, f'im_rec_automap_HCP_{HCP_nbr3}.mat'))
data1_lasso = loadmat(join(src_data, f'im_rec_lasso_HCP_{HCP_nbr1}.mat'))
data2_lasso = loadmat(join(src_data, f'im_rec_lasso_{HCP_nbr2}.mat'))
data3_lasso = loadmat(join(src_data, f'im_rec_lasso_HCP_{HCP_nbr3}.mat'))

mri_data1 =  np.abs(data1_automap['mri_data']).astype(np.float64)
mri_data2 =  np.abs(data2_automap['mri_data']).astype(np.float64)
mri_data3 =  np.abs(data3_automap['mri_data']).astype(np.float64)

im_rec_auto1 = np.abs(data1_automap['im_rec']).astype(np.float64)
im_rec_auto2 = np.abs(data2_automap['im_rec']).astype(np.float64)
im_rec_auto3 = np.abs(data3_automap['im_rec']).astype(np.float64)
print(im_rec_auto2.shape)

im_rec_lasso1 = data1_lasso['lasso_im_rec']
im_rec_lasso2 = data2_lasso['lasso_im_rec']
im_rec_lasso3 = data3_lasso['lasso_im_rec']

number_of_images = len(im_nbrs1) + len(im_nbrs2) + len(im_nbrs3)
all_images = np.zeros([number_of_images, N, N])
all_recs_lasso = np.zeros([number_of_images, N, N])
all_recs_auto = np.zeros([number_of_images, N, N])

for i in range(len(im_nbrs1)):
    print(np.amax(mri_data1[im_nbrs1[i]]))
    all_images[i, :, :] = mri_data1[im_nbrs1[i], :,:]
    all_recs_lasso[i, :, :] = cut_to_01(im_rec_lasso1[im_nbrs1[i], :,:])
    all_recs_auto[i, :, :] = im_rec_auto1[im_nbrs1[i], :,:]

for i in range(len(im_nbrs2)):
    print(np.amax(mri_data2[im_nbrs2[i]]))
    all_images[i+len(im_nbrs1), :, :] = mri_data2[im_nbrs2[i], :,:]
    all_recs_lasso[i+len(im_nbrs1), :, :] = cut_to_01(im_rec_lasso2[im_nbrs2[i], :, :])
    all_recs_auto[i+len(im_nbrs1), :, :] = im_rec_auto2[im_nbrs2[i], :,:]

for i in range(len(im_nbrs3)):
    print(np.amax(mri_data3[im_nbrs3[i]]))
    all_images[i+len(im_nbrs1)+len(im_nbrs2), :, :] = mri_data3[im_nbrs3[i], :,:]
    all_recs_lasso[i+len(im_nbrs1)+len(im_nbrs2), :, :] = cut_to_01(im_rec_lasso3[im_nbrs3[i], :, :])
    all_recs_auto[i+len(im_nbrs1)+len(im_nbrs2), :, :] = im_rec_auto3[im_nbrs3[i], :,:]

max_diff = []
for i in range(number_of_images):
    diff_automap = np.abs(all_recs_auto[i] - all_images[i])
    diff_lasso   = np.abs(all_recs_lasso[i] - all_images[i])



    max_diff_auto = np.amax(diff_automap)
    max_diff_lasso = np.amax(diff_lasso);
    max_diff.append(max_diff_auto)
    max_diff.append(max_diff_lasso)

print(max_diff)
max_err = max(max_diff)
for i in range(number_of_images):
    print(np.amax(all_recs_auto[i]))
    diff_automap = np.abs(all_recs_auto[i] - all_images[i])
    diff_lasso   = np.abs(all_recs_lasso[i] - all_images[i])
    err_auto = err_func(diff_automap)
    err_lasso = err_func(diff_lasso)

    MSE_auto = (l2_norm_of_tensor(diff_automap)**2)/np.prod(diff_automap.shape);
    MSE_lasso = (l2_norm_of_tensor(diff_lasso)**2)/np.prod(diff_lasso.shape);
    psnr_auto = 10*np.log10(1/MSE_auto)
    psnr_lasso = 10*np.log10(1/MSE_lasso)
    print(f'HCP_nbr: {HCP[i]}, im nbr: {im_nbrs[i]}, psnr_auto: {psnr_auto:5f}, psnr_lasso: {psnr_lasso:5f}, RMSE auto: {err_auto:5f}, RMSE lasso: {err_lasso:5f}')
    diff_im_automap = 1 - (diff_automap/max_err);
    diff_im_lasso   = 1 - (diff_lasso/max_err);

    pil_diff_im_true  = Image.fromarray(np.uint8(255*np.abs(all_images[i])));
    pil_diff_im_auto  = Image.fromarray(np.uint8(255*diff_im_automap));
    pil_diff_im_lasso = Image.fromarray(np.uint8(255*diff_im_lasso));
    
    pil_diff_im_true.save(join(dest_plots, f'true_im_HCP_{HCP[i]}_{im_nbrs[i]}.png'));
    pil_diff_im_auto.save(join(dest_plots, f'error_map_automap_HCP_{HCP[i]}_{im_nbrs[i]}.png'));
    pil_diff_im_lasso.save(join(dest_plots, f'error_map_lasso_HCP_{HCP[i]}_{im_nbrs[i]}.png'));

