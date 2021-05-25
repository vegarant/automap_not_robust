clear('all') ; close('all');
load('cilib_defaults.mat') % load font size, line width, etc.

% create destination for the plots
dest = 'plots';
if (exist(dest) ~= 7) 
    mkdir(dest);
end

N = 128; % resolution
nbr_coils = 15; % Number of coils 
acc = 4;

load(sprintf('../samp_patt/masks/mask_N_%d_acc_%d_equispaced.mat', N, acc)); % mask
idx = find(mask);

nbr_samples = length(idx);

% Construct coil sensitivites
FOV = 0.256; % FOV width
mxsize = N*[1 1];
pixelsize = FOV./mxsize;

coils = 2:2:20
for i = 1:length(coils)
    nbr_coils = coils(i);
    nbr_coils
    coil_sens = GenerateSensitivityMap( FOV, pixelsize, nbr_coils, .09, .18);

    opA = @(x, tflag) op_fourier_coil_2d(x, tflag, N, idx, coil_sens);

    tic
    sing_vals = svds(opA, [nbr_samples*nbr_coils,N*N], N*N);
    toc

    fname = sprintf('data/sing_val_N_%d_acc_%d_coil_%d.mat', N, acc, nbr_coils);
    save(fname, 'sing_vals', 'coil_sens', 'idx');

end
