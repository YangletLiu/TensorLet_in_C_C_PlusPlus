function result = read_name(path, name, expression)

match_str = regexp(name, expression, 'match');
%celldisp(match_str);
for i = 1:length(match_str)
    if ~ isempty(match_str{i})
        sub_name = fullfile(path, match_str{i}{1});
        result = readNPY(char(sub_name));
    end
end