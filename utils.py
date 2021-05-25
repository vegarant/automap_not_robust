import numpy as np

def convert_automap_samples_to_tf_samples_in_image_domain(samples, k_idx1, k_idx2):
    ''' Converts AUTOMAP samples adjusted to LASSO in Tensorflow

Arguments
---------
samples (np.ndarray) : Array with AUTOMAP samples. Shape (1,2*9855)
k_idx1 (np.ndarray): Row indices for the AUTOMAP Fourier samples
k_idx2 (np.ndarray): Column indices for the AUTOMAP Fourier samples

Returns
-------
X (np.ndarray): The measurements in image corresponding to the samples in 
                sampling domain for automap

Note: This function convert samples in k-space sampled for AUTOMAP using the
function `adv_tools_PNAS.automap_tools.sample_image` to the corresponding
samples used for the LASSO method. This is done by inverting the scaling from
done by function `sample_image`, and insert these samples in a zero-padded
image and apply the adjoint. When we sample this adjoint, we get the
corresponding measurements for the LASSO method. This approach only works
because AA* = Identity, where A is a subsampled (normalized) discrete Fourier
Transform (DCT) matrix.
'''

    size = samples.shape
    if max(size) != 2*9855 and (len(size) != 2 or len(size) != 2):
        raise IndexError('samples must have size (1,2*9855)');

    N = 128
    nbr_samples = 9855
    const = 4096*N*(0.0075/(2*4096)) # The extra factor N is compensating for 
                                     # not dividing by N when running the 
                                     # fft2 call. 
    samples = np.squeeze(samples)
    samples_real = samples[0:nbr_samples]
    samples_imag = samples[nbr_samples:]
    
    samples_complex = np.expand_dims(samples_real - 1j*samples_imag,-1);
    
    Y = np.zeros([N,N], dtype=np.complex128);
    Y[k_idx1, k_idx2] = samples_complex/const # Samples in sampling domain

    X = np.fft.ifft2(Y)*N;
    X = np.expand_dims(X, -1)

    return X

if __name__ == "__main__":
    """ This test code check that the noise-measurement ratio |e|/|y|, is the 
    same for both the AUTOMAP and LASSO sampling operator.
    """

    from adv_tools_PNAS.automap_config import src_weights, src_data
    from adv_tools_PNAS.automap_tools import read_automap_k_space_mask, sample_image
    from adv_tools_PNAS.adversarial_tools import l2_norm_of_tensor, scale_to_01
    from optimization.gpu.operators import MRIOperator
    from os.path import join
    import tensorflow as tf
    import numpy as np
    import scipy.io
    import h5py

    N = 128

    k_mask_idx1, k_mask_idx2 = read_automap_k_space_mask()
    HCP_nbr = 1033
    data = scipy.io.loadmat(join(src_data, f'HCP_mgh_{HCP_nbr}_T2_subset_N_128.mat'))
    mri_data = data['im']
    image = mri_data[2,:,:]
    image = np.expand_dims(image, 0)

    y_automap = sample_image(image, k_mask_idx1, k_mask_idx2)

    e = np.random.normal(loc=0, scale=0.01, size=y_automap.shape);

    n_e_auto = l2_norm_of_tensor(e)
    n_y_auto = l2_norm_of_tensor(y_automap)
    print(f'AUTOMAP: |e|/|y|: {n_e_auto/n_y_auto}')

   #########################################################################
   ###        Compute the same for tensorflow sampling operator          ###
   #########################################################################
    image = np.expand_dims(np.squeeze(image), -1);
    dtype = tf.float64
    cdtype = tf.complex128
    samp = np.swapaxes(np.fft.fftshift(np.array(h5py.File(join(src_data, 'k_mask.mat'), 'r')['k_mask']).astype(np.bool)),0,1)
    samp = np.expand_dims(samp, -1)

    X = convert_automap_samples_to_tf_samples_in_image_domain(e, k_mask_idx1, k_mask_idx2)
    
    # Build Primal-dual graph
    tf_im = tf.compat.v1.placeholder(cdtype, shape=[N,N,1], name='image')
    tf_samp_patt = tf.compat.v1.placeholder(tf.bool, shape=[N,N,1], name='sampling_pattern')
    
    wav = []
    levels = 1
    op = MRIOperator(tf_samp_patt, wav, levels, dtype=dtype)
    measurements = op.sample(tf_im)


    with tf.compat.v1.Session() as sess:
        y_tf = sess.run(measurements, feed_dict={ 'image:0': image,
                                                  'sampling_pattern:0': samp})
        

        e_tf = sess.run(measurements, feed_dict={ 'image:0': X,
                                                  'sampling_pattern:0': samp})

        n_y_tf = l2_norm_of_tensor(y_tf)
        n_e_tf = l2_norm_of_tensor(e_tf)

        print(f'TF: |e|/|y|: {n_e_tf/n_y_tf}')












