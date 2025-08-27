% Group40Ex7
% Initialization
lambda_task6 = 0.1302;
warning('off', 'all');

% Load the data
current_file_path = mfilename('fullpath');
[parent_folder, ~, ~] = fileparts(fileparts(current_file_path));
data_path = fullfile(parent_folder, 'TMS.xlsx');
if ~exist(data_path, 'file')
    error('The file TMS.xlsx does not exist in the specified path: %s', data_path);
end

% Read data
data = readtable(data_path);
data_with_TMS = data(data.TMS == 1, :);

% Remove Spike variable
data_with_TMS_NoSpike = removevars(data_with_TMS, 'Spike');

% Ensure all columns are numeric
columns_to_convert = {'Setup', 'Stimuli', 'Intensity', 'Frequency', 'CoilCode'};
for i = 1:length(columns_to_convert)
    column = columns_to_convert{i};
    if iscell(data_with_TMS_NoSpike.(column))
        data_with_TMS_NoSpike.(column) = cellfun(@str2double, data_with_TMS_NoSpike.(column));
    end
end

% Select variables
indepedent_vars = table2array(data_with_TMS_NoSpike(:, columns_to_convert));
EDduration = data_with_TMS_NoSpike.EDduration;

% Split data into training and testing sets
rng(1); % For reproducibility
cv = cvpartition(height(data_with_TMS), 'HoldOut', 0.3);
train_idx = training(cv);
test_idx = test(cv);

X_train = indepedent_vars(train_idx, :);
Y_train = EDduration(train_idx);
X_test = indepedent_vars(test_idx, :);
Y_test = EDduration(test_idx);

% Normalization
mx_train = mean(X_train);
X_train = X_train - mx_train;
Y_train = Y_train - mean(Y_train);
X_test = X_test - mx_train;
Y_test = Y_test - mean(Y_test);

mx_full = mean(indepedent_vars);
X_full = indepedent_vars - mx_full;
Y_full = EDduration - mean(EDduration);

%Model Training and Evaluation
% Full Model
lm_full = fitlm(X_full, Y_full);
Y_pred_full = predict(lm_full, X_test);
mse_full = mean((Y_test - Y_pred_full).^2);

% Stepwise Regression
lm_stepwise_train = stepwiselm(X_full, Y_full,'Verbose', 0);
Y_pred_stepwise_train = predict(lm_stepwise_train, X_test);
mse_stepwise_train = mean((Y_test - Y_pred_stepwise_train).^2);

% LASSO Model
[beta, fitinfo] = lasso(X_full, Y_full, 'CV', 10);
lambda_optimal = lambda_task6;
[~, ilmin] = min(abs(fitinfo.Lambda - lambda_optimal));
bLASSOV = beta(:, ilmin);
mxV = mean(X_test);
my = mean(Y_test);
bLASSOV = [my - mxV * bLASSOV; bLASSOV];
yfitLASSOV = [ones(length(Y_test), 1) X_test] * bLASSOV;
mse_lasso = mean((Y_test - yfitLASSOV).^2);

% Display Results
results_table = table({'Full Model'; 'Stepwise Model'; 'LASSO Model'}, ...
                      [mse_full; mse_stepwise_train; mse_lasso], ...
                      'VariableNames', {'Model', 'MSE'});
disp('Model Comparison Results (Initial Variable Selection):');
disp(results_table);

%Model Evaluation with Variable Selection in Training Set
% Stepwise Model with Training Selection
lm_stepwise_train = stepwiselm(X_train, Y_train, 'Verbose', 0);
Y_pred_stepwise_train = predict(lm_stepwise_train, X_test);
mse_stepwise_train = mean((Y_test - Y_pred_stepwise_train).^2);

% LASSO Model with Training Selection
[beta_train, fitinfo_train] = lasso(X_train, Y_train, 'CV', 10);
lambda_optimal = lambda_task6;
[~, ilmin] = min(abs(fitinfo_train.Lambda - lambda_optimal));
bLASSOV = beta_train(:, ilmin);
bLASSOV = [my - mxV * bLASSOV; bLASSOV];

yfitLASSOV = [ones(length(Y_test), 1) X_test] * bLASSOV;
mse_lasso_refit = mean((Y_test - yfitLASSOV).^2);

% Display Results
results_table_refit = table({'Stepwise Model (Refit)'; 'LASSO Model (Refit)'}, ...
                             [mse_stepwise_refit; mse_lasso_refit], ...
                             'VariableNames', {'Model', 'MSE'});
disp('Model Comparison Results (Refit in Training Set):');
disp(results_table_refit);

% Conclusions:
% - Based on the MSE results, the Stepwise model demonstrates better fit in both scenarios. When trained on the entire dataset, it outperforms LASSO significantly.
% - However, when tested on unseen data, both Stepwise and LASSO exhibit comparable performance. 
% - The higher MSE in the second scenario is expected, reflecting the challenges of generalization to data not used during training.