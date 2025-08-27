% Read the data from the file
current_file_path = mfilename('fullpath');
[parent_folder, ~, ~] = fileparts(fileparts(current_file_path));
data_path = fullfile(parent_folder, 'TMS.xlsx');

% Check if the file exists
if ~exist(data_path, 'file')
    error('The file TMS.xlsx does not exist in the specified path: %s', data_path);
end

% Load the data into a table
data = readtable(data_path);

% Perform linear and polynomial analysis for TMS = 0 (without TMS)
fprintf('\nAnalysis for TMS = 0 (Without TMS):\n');
analyze_polynomial(data, 0);

% Perform linear and polynomial analysis for TMS = 1 (with TMS)
fprintf('\nAnalysis for TMS = 1 (With TMS):\n');
analyze_polynomial(data, 1);

% Overall Conclusions:
% The low R2-value indicates that the model is unsuitable.
%  While polynomial regression slightly improves the fit, the improvement is insufficient to justify its use. 
% This suggests that ED Duration depends on factors beyond the setup, which alone cannot explain its behavior.
% The residual plot indicates that the model is unsuitable, as significant outliers deviate from the red reference lines.