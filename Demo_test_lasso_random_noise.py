"""
This script reads the random noise generated in the script
'Demo_test_automap_random_noise.py' and the unperturbed image. It samples the
image with and without noise and reconstructs the two images with the LASSO
method.

Only the image reconstructed from the noiseless measurements is used in the paper.
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
import scipy.io

from adv_tools_PNAS.automap_config import src_data;
from adv_tools_PNAS.adversarial_tools import l2_norm_of_tensor, cut_to_01
from adv_tools_PNAS.automap_tools import read_automap_k_space_mask
from adv_tools_PNAS.Runner import Runner;
from utils import convert_automap_samples_to_tf_samples_in_image_domain

src_noise = 'data_random';

N = 128
wavname = 'db2'
levels = 3
use_gpu = True
compute_node = 1
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


dest_data = 'data_random';
dest_plots = 'plots_random';

if not (os.path.isdir(dest_data)):
    os.mkdir(dest_data);

if not (os.path.isdir(dest_plots)):
    os.mkdir(dest_plots);

# Parameters for the CS-algorithm
n_iter = 1000
tau = 0.6
sigma = 0.6
lam = 0.0001

############################################################################
###                     Build Tensorflow Graph                           ###
############################################################################

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

k_mask_idx1, k_mask_idx2 = read_automap_k_space_mask();

p = 0.06
fname_data = 'noise_gauss_%d_automap.mat' % (round(1000*p))
data = scipy.io.loadmat(join(src_noise, fname_data))
image = np.squeeze(data['mri_data'])
image = np.expand_dims(image, -1)
print(data['noise_gauss'].shape);
noise = convert_automap_samples_to_tf_samples_in_image_domain(data['noise_gauss'], 
                                                              k_mask_idx1,
                                                              k_mask_idx2)

mri_data = np.zeros([2, N, N], dtype=np.complex128);
mri_data[0,:,:] = np.squeeze(image.copy())
mri_data[1,:,:] = np.squeeze(image.copy()) + np.squeeze(noise)

batch_size = mri_data.shape[0];
zoom_size = 80;

with tf.compat.v1.Session() as sess:

    sess.run(tf.compat.v1.global_variables_initializer())
    weights = np.ones([128,128,1], dtype=sdtype);

    np_im_rec = np.zeros([batch_size, N, N], dtype=scdtype);

    for i in range(batch_size):

        _image = mri_data[i,:,:];
        _image = np.expand_dims(_image, -1)
        _rec = sess.run(tf_recovery, feed_dict={ 'tau:0': tau,
                                                 'lambda:0': lam,
                                                 'sigma:0': sigma,
                                                 'weights:0': weights,
                                                 'n_iter:0': n_iter,
                                                 'image:0': _image,
                                                 'sampling_pattern:0': samp})
        np_im_rec[i,:,:] = _rec[:,:,0];

    np_im_rec = cut_to_01(np_im_rec);

    im_no_noise = np_im_rec[0,:,:];
    im_gauss = np_im_rec[1,:,:];


    fname_data = 'noise_gauss_%d_lasso.mat' % (round(1000*p))
    scipy.io.savemat(join(dest_data, fname_data), {'mri_data': mri_data, 
                                                   'noise_gauss': noise,
                                                   'rec_no_noise': im_no_noise,
                                                   'rec_gauss_noise': im_gauss});

    fname_no_noise = f'im_rec_lasso_no_noise';
    fname_gauss   = f'im_rec_lasso_gauss_p_{round(1000*p)}';

    Image_im_no_noise = Image.fromarray(np.uint8(255*np.abs(im_no_noise)));
    Image_im_gauss = Image.fromarray(np.uint8(255*np.abs(im_gauss)));

    Image_im_no_noise_zoom = Image.fromarray(np.uint8(255*np.abs(im_no_noise[:zoom_size, -zoom_size:])));
    Image_im_gauss_zoom = Image.fromarray(np.uint8(255*np.abs(im_gauss[:zoom_size, -zoom_size:])));

    Image_im_no_noise.save(join(dest_plots, fname_no_noise + '.png'));
    Image_im_gauss.save(join(dest_plots, fname_gauss + '.png'));

    Image_im_no_noise_zoom.save(join(dest_plots, fname_no_noise + '_zoom.png'));
    Image_im_gauss_zoom.save(join(dest_plots, fname_gauss + '_zoom.png'));









