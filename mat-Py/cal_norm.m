function result = cal_norm(origin, reconstruction)

result = 0;
error = origin -reconstruction;
[dim1, dim2, dim3] = size(origin);
for i = 1:dim3
    result = result + norm(error(:, :, i));
end