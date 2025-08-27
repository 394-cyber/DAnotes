% Group40Ex8
warning('off', 'all');

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

% Extract data with TMS
data_with_TMS = data(data.TMS == 1, :);
data_with_TMS = removevars(data_with_TMS, 'Spike');

% Ensure all columns are numeric
columns_to_convert = {'Setup', 'Stimuli', 'Intensity', 'Frequency', 'CoilCode', 'preTMS'};
for i = 1:length(columns_to_convert)
    column = columns_to_convert{i};
    if iscell(data_with_TMS.(column))
        data_with_TMS.(column) = cellfun(@str2double, data_with_TMS.(column));
    end
end

% Select variables
indepedent_vars = table2array(data_with_TMS(:, columns_to_convert));
EDduration = data_with_TMS.EDduration;

% Centering the data
mx = mean(indepedent_vars);
X = indepedent_vars - mx;
X = zscore(X);
my = mean(EDduration);
y = EDduration - my;
fprintf('Model with preTMS variables\n');
compare_models(X, y);

%% Add postTMS as an independent variable
% Select variables
columns_to_convert = {'Setup', 'Stimuli', 'Intensity', 'Frequency', 'CoilCode', 'preTMS', 'postTMS'};
for i = 1:length(columns_to_convert)
    column = columns_to_convert{i};
    if iscell(data_with_TMS.(column))
        data_with_TMS.(column) = cellfun(@str2double, data_with_TMS.(column));
    end
end

indepedent_vars = table2array(data_with_TMS(:, columns_to_convert));
EDduration = data_with_TMS.EDduration;

% Center the data
mx = mean(indepedent_vars);
X = indepedent_vars;
X = zscore(X);
my = mean(EDduration);
y = EDduration;
fprintf('Model with preTMS and postTMS variables\n');

compare_models(X, y);

% Stepwise model achieved the highest adjusted R^2 and lowest MSE, proving its balance between simplicity and accuracy.
% PCR, along with LASSO model, performed the worst, indicating that dimensionality reduction through PCR is less effective for this dataset.

% Comparison with Task 5
% Task 5 showed that TMS presence improved EDduration predictability, with higher R^2 values across all polynomial models.
% Similarly, in Task 8, adding postTMS significantly boosted adjusted R^2, especially in the Stepwise model.

% Model with preTMS and postTMS variables
% Including the postTMS variable drastically improved performance. 
% - Full and Stepwise models achieved nearly perfect fits (Adjusted R^2 = 1).
% - LASSO and PCR models also improved but remained less effective than Stepwise.
% These results confirm postTMS as a crucial predictor of EDduration, enhancing model accuracy and capturing complex relationships in the data.
