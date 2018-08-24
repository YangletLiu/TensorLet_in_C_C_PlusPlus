function [cal_nor, error] = TUCKER(time, ten_dim1, ten_dim2, ten_dim3, core_dim1, core_dim2, core_dim3)
tic

addpath(genpath(pwd));

%Read Tensor from Python
size = ['[' num2str(ten_dim1) ', ' num2str(ten_dim2)  ', ' num2str(ten_dim3) ']'];
read_path = fullfile('..', 'save', size);
read_name = fullfile(read_path,  'tucker_origin_tensor.npy');
%T = readNPY(read_name);
if ~ (exist(read_path))
	mkdir(read_path);
end


% Generate pseudorandom LMLRA (U,S) and associated full tensor T.
size_tens = [ten_dim1 ten_dim2 ten_dim3];
size_core = [core_dim1, core_dim2, core_dim3];
[U,S] = lmlra_rnd(size_tens,size_core);
T = lmlragen(U,S);
writeNPY(T, read_name);

% Compute an LMLRA of a noisy version of T with 20dB SNR.
[Uhat,Shat] = lmlra(noisy(T,20), size_core);

tucker_reconstruction = tmprod(Shat,Uhat,1:3);
error = tucker_reconstruction - T;

cal_nor = 0;
for i = 1:ten_dim3
    cal_nor = cal_nor + norm(error(:, :, i));
end

%part1 = num2str(ten_dim1);
%part2 = num2str(ten_dim2);
%part3 = num2str(ten_dim3);
%size = strcat('(',part1, ',', part2, ',', part3,')');
%folder_name = strcat('./', fullfile('save', size));
%if ~ (exist(folder_name))
%	mkdir(folder_name)
%end

file_name = strcat('ml_TUCKER_rec_', num2str(time), '.npy');
file_name = fullfile(read_path, file_name);
writeNPY(tucker_reconstruction, file_name);

toc

%save myfile.mat
%load myfile.mat