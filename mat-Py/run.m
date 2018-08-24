R = 4; time = 1;
core_dim1 = 5; core_dim2 = 5; core_dim3 = 16;
for dim1 = 10:10:50
    dim2 = dim1;
    for dim3 = 32:32:64
        [cp, error] = CP(time, dim1, dim2, dim3, R);
        [tu, error] = TUCKER(time, dim1, dim2, dim3, core_dim1, core_dim2, core_dim3);
    end
end
