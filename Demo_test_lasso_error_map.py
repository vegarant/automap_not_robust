"""
This script uses the LASSO method to reconstruct images. The resulting images are stored in datasets that can be read by the script 'Demo_read_and_plot_error_maps.py'.

Run the scripts in this order
1. Demo_test_automap_error_map.py
2. Demo_test_lasso_error_map.py
3. Demo_read_and_plot_error_maps.py

In the first two scripts, it is necessary to run the scripts multiple times to generate reconstructions from all the data. Change the `HCP_nbr` and 'use_HCP', to apply the script to all the data.
"""

import time
import tensorflow as tf
import numpy as np
import h5py
import scipy.io
from os.path import join 
import os.path

from optimization.gpu.operators import MRIOperator
from optimization.gpu.proximal import WeightedL1Prox, SQLassoProx2
from optimization.gpu.algorithms import SquareRootLASSO
from optimization.utils import estimate_sparsity, generate_weight_matrix
from tfwavelets.dwtcoeffs import get_wavelet
from tfwavelets.nodes import idwt2d
from PIL import Image
import matplotlib.image as mpimg;

from adv_tools_PNAS.automap_config import src_data;
from adv_tools_PNAS.adversarial_tools import l2_norm_of_tensor, cut_to_01
from adv_tools_PNAS.Runner import Runner;
from scipy.io import savemat

src_im_rec = 'data_error_map';

N = 128
wavname = 'db2'
levels = 3
use_gpu = True
compute_node = 3
dtype = tf.float64;
sdtype = 'float64';
scdtype = 'complex128';
cdtype = tf.complex128
wav = get_wavelet(wavname, dtype=dtype);
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


dest_data = 'data_error_map';
dest_plots = 'plots_error_map';

if not (os.path.isdir(dest_data)):
    os.mkdir(dest_data);

if not (os.path.isdir(dest_plots)):
    os.mkdir(dest_plots);

# Parameters for the CS-algorithm
n_iter = 1000
tau = 0.6
sigma = 0.6
lam = 0.0001

# Parameters for CS algorithm
pl_sigma = tf.compat.v1.placeholder(dtype, shape=(), name='sigma')
pl_tau   = tf.compat.v1.placeholder(dtype, shape=(), name='tau')
pl_lam   = tf.compat.v1.placeholder(dtype, shape=(), name='lambda')

# Build Primal-dual graph
tf_im = tf.compat.v1.placeholder(cdtype, shape=[N,N,1], name='image')
tf_samp_patt = tf.compat.v1.placeholder(tf.bool, shape=[N,N,1], name='sampling_pattern')

# For the weighted l^1-norm
pl_weights = tf.compat.v1.placeholder(dtype, shape=[N,N,1], name='weights')

tf_input = tf_im

op = MRIOperator(tf_samp_patt, wav, levels, dtype=dtype)
measurements = op.sample(tf_input)

tf_adjoint_coeffs = op(measurements, adjoint=True)
adj_real_idwt = idwt2d(tf.math.real(tf_adjoint_coeffs), wav, levels)
adj_imag_idwt = idwt2d(tf.math.imag(tf_adjoint_coeffs), wav, levels)
tf_adjoint = tf.complex(adj_real_idwt, adj_imag_idwt)

prox1 = WeightedL1Prox(pl_weights, pl_lam*pl_tau, dtype=dtype)
prox2 = SQLassoProx2(dtype=dtype)

alg = SquareRootLASSO(op, prox1, prox2, measurements, sigma=pl_sigma, tau=pl_tau, lam=pl_lam, dtype=dtype)

initial_x = op(measurements, adjoint=True)

result_coeffs = alg.run(initial_x)

real_idwt = idwt2d(tf.math.real(result_coeffs), wav, levels)
imag_idwt = idwt2d(tf.math.imag(result_coeffs), wav, levels)
tf_recovery = tf.complex(real_idwt, imag_idwt)

samp = np.swapaxes(np.fft.fftshift(np.array(h5py.File(join(src_data, 'k_mask.mat'), 'r')['k_mask']).astype(np.bool)), 0,1)
samp = np.expand_dims(samp, -1)


HCP_nbr = 1004
use_HCP = True
if use_HCP:
    data = scipy.io.loadmat(join(src_im_rec, f'im_rec_automap_HCP_{HCP_nbr}.mat'));
else:
    data = scipy.io.loadmat(join(src_im_rec, f'im_rec_automap_dataset1.mat'));

mri_data = data['mri_data'];
automap_im_rec = data['im_rec'];

batch_size = mri_data.shape[0];

print(f'Number of images: {batch_size}')

with tf.compat.v1.Session() as sess:

    sess.run(tf.compat.v1.global_variables_initializer())
    weights = np.ones([128,128,1], dtype=sdtype);

    lasso_im_rec = np.zeros([batch_size, N, N], dtype=sdtype);

    for i in range(batch_size):
        print(f'Reconstructing image: {i}')
        _image = mri_data[i,:,:];
        _image = np.expand_dims(_image, -1)
        _rec = sess.run(tf_recovery, feed_dict={ 'tau:0': tau,
                                                 'lambda:0': lam,
                                                 'sigma:0': sigma,
                                                 'weights:0': weights,
                                                 'n_iter:0': n_iter,
                                                 'image:0': _image,
                                                 'sampling_pattern:0': samp})
        lasso_im_rec[i,:,:] = cut_to_01(_rec[:,:,0]);

if use_HCP:
    savemat(join(dest_data, f'im_rec_lasso_HCP_{HCP_nbr}.mat'), {'lasso_im_rec': lasso_im_rec});
else:
    savemat(join(dest_data, f'im_rec_lasso_dataset1.mat'), {'lasso_im_rec': lasso_im_rec});

#        bd = 5
#        im_out = np.ones([2*N+bd, 2*N+bd]);
#
#        diff_automap = np.abs(automap_im_rec[i] - mri_data[i])
#        diff_lasso = np.abs(lasso_im_rec[i] - mri_data[i])
#
#        max_diff_auto = np.amax(diff_automap)
#        max_diff_lasso = np.amax(diff_lasso);
#        max_err = max(max_diff_auto, max_diff_lasso);
#        
#        diff_im_automap = 1 - (diff_automap/max_err);
#        diff_im_lasso   = 1 - (diff_lasso/max_err);
#
#        im_out[:N,:N] = automap_im_rec[i]
#        im_out[:N,N+bd:] = lasso_im_rec[i]
#        im_out[N+bd:,:N] = diff_im_automap
#        im_out[N+bd:,N+bd:] = diff_im_lasso
#
#        pil_im = Image.fromarray(np.uint8(255*im_out));
#        pil_im.save(join(dest_plots, f'error_map_HCP_{HCP_nbr}_{i:03d}.png'));

