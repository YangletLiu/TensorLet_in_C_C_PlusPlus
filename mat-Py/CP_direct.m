dim1 = 7;
dim2 = 7;
dim3 = 8;
R = 4;
tic

addpath(genpath(pwd));
% Generate pseudorandom factor matrices U0 and their associated full tensor T.
size_tens = [dim1 dim2 dim3];
U = cpd_rnd(size_tens,R);
T = cpdgen(U);

% Compute the CPD of the full tensor T.
Uhat = cpd(T,R);

cp_reconstruction = zeros(size_tens);
for i = 1:R
	cp_reconstruction = cp_reconstruction + outprod(Uhat{1}(:, i), Uhat{2}(:, i), Uhat{3}(:, i));
end
error = cp_reconstruction - T;

toc