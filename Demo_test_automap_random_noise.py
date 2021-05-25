"""
This script reads an image provided by GE Healthcare and samples the image with
and without random noise added to the measurements. The two reconstructed
images are stored as png images and in a .mat file.    

In the paper, only the reconstructed image from noiseless images is used. 
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
from adv_tools_PNAS.automap_tools import read_automap_k_space_mask, compile_network, hand_f, sample_image
from adv_tools_PNAS.adversarial_tools import l2_norm_of_tensor, scale_to_01
from PIL import Image

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

new_im1 = scale_to_01(mpimg.imread(join(src_data, 'brain1_128_anonymous.png')))

mri_data = np.zeros([1, N,N], dtype='float32')
mri_data[0, :, :] = new_im1
print('max_im_1: ', np.amax(mri_data));
print('mri_data.shape: ', mri_data.shape)

batch_size = mri_data.shape[0]

dest_plots = './plots_random'
dest_data = './data_random'

if not (os.path.isdir(dest_plots)):
    os.mkdir(dest_plots)
if not (os.path.isdir(dest_data)):
    os.mkdir(dest_data)

sess = tf.compat.v1.Session()

raw_f, _ = compile_network(sess, batch_size)

sample_im = lambda x: sample_image(x, k_mask_idx1, k_mask_idx2)

f  = lambda x: hand_f(raw_f, x, k_mask_idx1, k_mask_idx2)



k_space_data_no_noise = sample_im(mri_data)
print('k_space_data_no_noise.shape: ',  k_space_data_no_noise.shape)
# Create noise, mri_data.shape = [1,N,N]
noise_gauss = np.float32(np.random.normal(loc=0, scale=1, size=k_space_data_no_noise.shape))

# Scale the noise
norm_k_space_data_no_noise = l2_norm_of_tensor(k_space_data_no_noise)
norm_noise_gauss    = l2_norm_of_tensor(noise_gauss)

p = 0.06;
noise_gauss *= (p*norm_k_space_data_no_noise/norm_noise_gauss);

# Save noise
fname_data = 'noise_gauss_%d_automap.mat' % (round(1000*p));


k_space_data_gauss = k_space_data_no_noise + noise_gauss

fx_no_noise = scale_to_01(raw_f(k_space_data_no_noise))
fx_noise_gauss = scale_to_01(raw_f(k_space_data_gauss))

scipy.io.savemat(join(dest_data, fname_data), {'mri_data': mri_data, 
                                               'noise_gauss': noise_gauss,
                                               'rec_no_noise': fx_no_noise,
                                               'rec_gauss_noise': fx_noise_gauss});

image_no_noise = mri_data;

for i in range(batch_size):
    # Save reconstruction with noise
    image_data_no_noise = np.uint8(255*fx_no_noise[i])
    image_data_gauss = np.uint8(255*fx_noise_gauss[i])

    image_rec_no_noise = Image.fromarray(image_data_no_noise)
    image_rec_gauss = Image.fromarray(image_data_gauss)

    image_rec_no_noise.save(join(dest_plots, 'im_rec_auto_no_noise_nbr_%d.png' % (i)));
    image_rec_gauss.save(join(dest_plots, 'im_rec_auto_gauss_p_%d_nbr_%d.png' % (round(p*1000), i)));

    # Save original image with noise
    image_orig_no_noise = Image.fromarray(np.uint8(255*(scale_to_01(image_no_noise[i,:,:]))));

    image_orig_no_noise.save(join(dest_plots, 'im_org_no_noise_nbr_%d.png' % (i)));

    # Create zoomed crops, reconstructions 
    image_rec_no_noise_zoom = Image.fromarray(image_data_no_noise[:size_zoom, -size_zoom:]);
    image_rec_gauss_zoom = Image.fromarray(image_data_gauss[:size_zoom, -size_zoom:]);

    image_rec_no_noise_zoom.save(join(dest_plots, 'im_rec_auto_no_noise_nbr_%d_zoom.png' % (i)));
    image_rec_gauss_zoom.save(join(dest_plots, 'im_rec_auto_gauss_p_%d_nbr_%d_zoom.png' % (round(p*1000), i)));

    # Create zoomed crops, images with noise 
    image_orig_no_noise_zoom = Image.fromarray(np.uint8(255*(scale_to_01(image_no_noise[i, :size_zoom, -size_zoom:]))));

    image_orig_no_noise_zoom.save(join(dest_plots, 'im_no_noise_nbr_%d_zoom.png' % (i)));

sess.close();

