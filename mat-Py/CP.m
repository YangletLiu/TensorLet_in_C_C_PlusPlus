function [cal_nor, error] = CPD(time, dim1, dim2, dim3, R)
tic

addpath(genpath(pwd));

% Generate pseudorandom factor matrices U0 and their associated full tensor T.
size_tens = [dim1 dim2 dim3];

%Read Tensor from Python
size = ['[' num2str(dim1) ', ' num2str(dim2)  ', ' num2str(dim3) ']'];
read_path = fullfile('..', 'save', size);
read_name = fullfile(read_path,  'cp_origin_tensor.npy');
%T = readNPY(read_name);
if ~ (exist(read_path))
	mkdir(read_path);
end

U = cpd_rnd(size_tens,R);
T = cpdgen(U);
writeNPY(T, read_name);

% Compute the CPD of the full tensor T.
Uhat = cpd(T,R);

cp_reconstruction = zeros(size_tens);
for i = 1:R
	cp_reconstruction = cp_reconstruction + outprod(Uhat{1}(:, i), Uhat{2}(:, i), Uhat{3}(:, i));
end
error = cp_reconstruction - T;

cal_nor = 0;
for i = 1:dim3
    cal_nor = cal_nor + norm(error(:, :, i));
end

%part1 = num2str(dim1);
%part2 = num2str(dim2);
%part3 = num2str(dim3);
%size = strcat(part1, ',', part2, ',', part3);
%folder_name = fullfile('..', 'save', size)
%if ~ (exist(folder_name))
%	mkdir(folder_name)
%end

file_name = strcat('ml_CP_rec_', num2str(time), '.npy');
file_name = fullfile(read_path, file_name);
writeNPY(cp_reconstruction, file_name);

toc