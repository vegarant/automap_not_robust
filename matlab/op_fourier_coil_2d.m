% Function handle to perform vector times matrix multiplicaiton with 
% parallel MRI matrix
%
% Arguments
% ---------
% x (vector) : Vector to multiply with the matrix.
% mode (int) : if 1, we multiply with the matrix A, otherwise we multiply with 
%              its transpose.
% N (int) : The dimension of the image is N x N.
% idx (vector) : Linear indices of the samples we would like to sample. A 
%                subset of {1, ... , N^2}.
% coil_sens (array) : Array of size [N,N,c] with sensitivity profile of each 
%                     of c coils
function y=op_fourier_coil_2d(x, mode, N, idx, coil_sens)

    if (~isvector(x))
        error('Input is not a vector');
    end

    nbr_coils = size(coil_sens, 3);
    nbr_samples = length(idx);
    idx_full = zeros([nbr_samples*nbr_coils,1]);

    for i = 1:nbr_coils
        idx_full((i-1)*nbr_samples+1:i*nbr_samples) = (i-1)*N*N + idx;
    end

    if or(mode == 1, strcmp(mode,'notransp')) 
        X = reshape(x, [N,N]);
        XC = X.*coil_sens;
        YC = fftshift( fftshift( fft2(XC), 1), 2)/N;
        y = YC(idx_full);
    else % Transpose
        Z = zeros([N, N, nbr_coils]);
        Z(idx_full) = x;
        Z = ifft2( ifftshift( ifftshift(Z, 1), 2) )*N;
        Z = conj(coil_sens).*Z;
        Z = sum(Z, 3);
        y = reshape(Z, [N*N,1]);
    end
end
