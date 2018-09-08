time = 2; ten_dim1 = 10; ten_dim2 = 10; ten_dim3 = 32; core_dim1 = 5; core_dim2 = 5; core_dim3 = 16;
tic

addpath(genpath(pwd));

% Generate pseudorandom LMLRA (U,S) and associated full tensor T.
size_tens = [ten_dim1 ten_dim2 ten_dim3];
size_core = [core_dim1, core_dim2, core_dim3];
[U,S] = lmlra_rnd(size_tens,size_core);
T = lmlragen(U,S);



size = ['[' num2str(ten_dim1) ', ' num2str(ten_dim2)  ', ' num2str(ten_dim3) ']'];
read_path = fullfile('..', 'save', size);
read_name = fullfile(read_path,  'origin_tensor.npy');
%T = readNPY(read_name);

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

file_name = strcat(num2str(time), '_TUCKER_rec_ml.npy');
file_name = fullfile(read_path, file_name);
writeNPY(tucker_reconstruction, file_name);

toc

%save myfile.mat
%load myfile.mat