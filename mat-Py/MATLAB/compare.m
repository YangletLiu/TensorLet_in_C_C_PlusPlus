addpath(genpath(pwd));
save_path = fullfile('..', 'save');
list = dir(save_path);
name = cell(1, 6);
ml_cp_list = [];
ml_tucker_list = [];
py_cp_list = [];
py_tucker_list = [];

%under save folder
for i = 3:length(list)
    %under specific size
    sub = list(i).name;
    sub_path = fullfile(save_path, sub);
    sub_name = fullfile(sub_path, '*.npy');
    sub_list = dir(sub_name);
    for j = 1:length(sub_list)
        name{j} = sub_list(j).name;
    end

    
    %cp
    %get saved tensor
    expression = 'ml_CP.+';
    ml_CP = read_name(sub_path, name, expression);

    expression = 'py_CP.+';
    py_CP = read_name(sub_path, name, expression);

    expression = 'cp_origin.+';
    cp_origin = read_name(sub_path, name, expression);
    
    %calculate norm
    ml_cp_error = cal_norm(cp_origin, ml_CP);
    ml_cp_list = [ml_cp_list ml_cp_error];
    py_cp_error = cal_norm(cp_origin, py_CP);
    py_cp_list = [py_cp_list py_cp_error];
    

    %tucker
    %get saved tensor
    expression = 'ml_TUCKER.+';
    ml_TUCKER = read_name(sub_path, name, expression);

    expression = 'py_TUCKER.+';
    py_TUCKER = read_name(sub_path, name, expression);

    expression = 'tucker_origin.+';
    tucker_origin = read_name(sub_path, name, expression);

    %calculate norm
    ml_tucker_error = cal_norm(tucker_origin, ml_TUCKER);
    ml_tucker_list = [ml_tucker_list ml_tucker_error];
    py_tucker_error = cal_norm(tucker_origin, py_TUCKER);
    py_tucker_list = [py_tucker_list py_tucker_error];
end