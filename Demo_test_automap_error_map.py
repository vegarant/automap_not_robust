"""
This script uses the AUTOMAP method to reconstruct images. The resulting images
are stored in datasets that can be read by the script
'Demo_read_and_plot_error_maps.py'.

Run the scripts in this order
1. Demo_test_automap_error_map.py
2. Demo_test_lasso_error_map.py
3. Demo_read_and_plot_error_maps.py

In the first two scripts, it is necessary to run the scripts multiple times to
generate reconstructions from all the data. Change the `HCP_nbr` and 'use_HCP',
to apply the script to all the data.
"""

import tensorflow as tf;
import scipy.io;
import h5py
from os.path import join;
import os;
import os.path;
import _2fc_2cnv_1dcv_L1sparse_64x64_tanhrelu_upg as arch
import matplotlib.image as mpimg;
import numpy as np;
from adv_tools_PNAS.automap_config import src_weights, src_data;
from adv_tools_PNAS.automap_tools import read_automap_k_space_mask, compile_network, hand_f, hand_dQ;
from adv_tools_PNAS.adversarial_tools import l2_norm_of_tensor, scale_to_01
from PIL import Image
from scipy.io import loadmat, savemat

use_gpu = False
compute_node = 3
if use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]= "%d" % (compute_node)
    print('Compute node: {}'.format(compute_node))
else: 
    os.environ["CUDA_VISIBLE_DEVICES"]= "-1"

# Turn on soft memory allocation
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.log_device_placement = False
sess = tf.compat.v1.Session(config=tf_config)


k_mask_idx1, k_mask_idx2 = read_automap_k_space_mask();

N = 128
size_zoom = 80

HCP_nbr = 1002 # Change the HCP_nbr to pick a different dataset
use_HCP = True # Set to False to use the fine tuned dataset
if use_HCP:
    data = scipy.io.loadmat(join(src_data, f'HCP_mgh_{HCP_nbr}_T2_subset_N_128.mat'));
else:
    data = scipy.io.loadmat(join(src_data, f'dataset1.mat'));

mri_data = data['im'];
#new_im1 = mpimg.imread(join(src_data, 'brain1_128_anonymous.png'))
#new_im2 = mpimg.imread(join(src_data, 'brain2_128_anonymous.png'))

batch_size = mri_data.shape[0]

for i in range(batch_size):
    mri_data[i, :,:] = scale_to_01(mri_data[i,:,:]);

dest_plots = './plots_error_map'
dest_data = './data_error_map'

if not (os.path.isdir(dest_plots)):
    os.mkdir(dest_plots)
if not (os.path.isdir(dest_data)):
    os.mkdir(dest_data)

sess = tf.compat.v1.Session()

raw_f, _ = compile_network(sess, batch_size)

f  = lambda x: hand_f(raw_f, x, k_mask_idx1, k_mask_idx2)

print('mri_data.shape: ', mri_data.shape);


fx_no_noise = f(mri_data)

im_rec = np.zeros(mri_data.shape, mri_data.dtype);


for i in range(batch_size):
    im_rec[i,:,:]  = scale_to_01(fx_no_noise[i]);
    
if use_HCP:
    savemat(join(dest_data, f'im_rec_automap_HCP_{HCP_nbr}.mat'), {'mri_data': mri_data, 'im_rec': im_rec});
else:
    savemat(join(dest_data, f'im_rec_automap_dataset1.mat'), {'mri_data': mri_data, 'im_rec': im_rec});

sess.close();



