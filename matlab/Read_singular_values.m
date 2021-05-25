clear('all') ; close('all');
load('cilib_defaults.mat') % load font size, line width, etc.

src_mask = '/mn/sarpanitu/ansatte-u4/vegarant/storage_matters_arising_final/fastMRI_masks';
src = 'data';
% Create destination for the plots
dest = 'plots';
if (exist(dest) ~= 7) 
    mkdir(dest);
end

disp_plots = 'off';


N = 128;
coils = [2:2:12, 16, 20];
acc = 8;

fig = figure('visible', disp_plots);
label = cell(length(coils), 1);

load(fullfile(src_mask, sprintf('mask_N_%d_acc_%d_equispaced.mat', N, acc))); % mask
idx = find(mask);
nbr_samples = length(idx);

eff_rank = zeros([length(coils), 1]);

fprintf('Number of coils & Rank, acc: %d \n', acc);

for i = 1:length(coils)
    nbr_coils = coils(i); 
    fname_core = sprintf('sing_val_N_%d_acc_%d_coil_%d', N, acc, nbr_coils);
    load(fullfile(src,[fname_core, '.mat']));
    label{i} = sprintf('coils: %2d', nbr_coils);
    semilogy(sing_vals, 'linewidth', 2);
    hold('on');
    tol = max([nbr_samples*nbr_coils, N*N])*eps(single(sing_vals(1)));
    eff_rank(i) = sum(sing_vals > tol);
    
    fprintf('%2d & %5d \n', nbr_coils, eff_rank(i));
end
axis([1, N*N, 10^(-7), 2])
legend(label, 'location', 'southwest');
set(gca, 'FontSize', cil_dflt.font_size);
set(gca,'LooseInset',get(gca,'TightInset'));
fname = sprintf('sing_val_N_%d_acc_%d', N, acc);
saveas(fig,fullfile(dest, fname), cil_dflt.plot_format);
saveas(fig,fullfile(dest, fname), cil_dflt.image_format);

