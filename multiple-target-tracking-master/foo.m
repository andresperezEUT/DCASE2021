function [] = foo(input_file_path)
%FOO Summary of this function goes here
%   Detailed explanation goes here
    input_file_path
    Y = csvread(input_file_path)';
    [filepath,name,~] = fileparts(input_file_path);
    output_file = strcat(filepath,'/',name,'.mat');

end

