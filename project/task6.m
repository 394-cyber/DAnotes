% Group40Ex6
warning('off', 'all');
seed_value=1;

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

% Keep only rows with non-missing Spike values
data_with_TMS_Spike = data_with_TMS(~ismissing(data_with_TMS.Spike), :);

% Ensure all columns are numeric for Spike analysis
columns_to_convert = {'Setup', 'Stimuli', 'Intensity', 'Spike', 'Frequency', 'CoilCode'};
for i = 1:length(columns_to_convert)
    column = columns_to_convert{i};
    if iscell(data_with_TMS_Spike.(column))
        data_with_TMS_Spike.(column) = cellfun(@(x) str2double(x), data_with_TMS_Spike.(column), 'UniformOutput', false);
        data_with_TMS_Spike.(column) = cell2mat(data_with_TMS_Spike.(column));
    end
end

% Select variables
indepedent_vars = table2array(data_with_TMS_Spike(:, {'Setup', 'Stimuli', 'Intensity', 'Spike', 'Frequency', 'CoilCode'}));
EDduration = data_with_TMS_Spike.EDduration;

% centering the data
mx = mean(indepedent_vars);
X = indepedent_vars - mx;
y = EDduration - mean(EDduration);

% === Analysis with Spike ===
% --- Full Model ---
lm_full = fitlm(X, y);
y_pred_full = predict(lm_full, X);
mse_full = mean((y - y_pred_full).^2);
r2_full = lm_full.Rsquared.Adjusted;

% --- Stepwise Regression ---
lm_stepwise = stepwiselm(X, y, 'Verbose', 0);
y_pred_stepwise = predict(lm_stepwise, X);
mse_stepwise = mean((y - y_pred_stepwise).^2);
% mse_stepwise_manual = lm_stepwise.MSE;
% fprintf('MSE Stepwise Manual: %.4f, MSE Stepwise: %.4f\n', mse_stepwise_manual, mse_stepwise);
r2_stepwise_adj = lm_stepwise.Rsquared.Adjusted;

% --- LASSO Model ---
rng(seed_value); 
[beta, fitinfo] = lasso(X, y, 'CV', 10);
lambda_optimal = fitinfo.LambdaMinMSE;
disp(lambda_optimal);
[~, ilmin] = min(abs(fitinfo.Lambda - lambda_optimal));
bLASSOV = beta(:, ilmin);
mxV = mean(X);
my = mean(y);
bLASSOV = [my - mxV*bLASSOV; bLASSOV];
yfitLASSOV = [ones(length(y), 1) X] * bLASSOV;

% Calculate MSE and R^2 for LASSO
resLASSOV = yfitLASSOV - y;
mse_lasso = mean(resLASSOV .^ 2);
RSSLASSO = sum(resLASSOV .^ 2);
TSS = sum((y - mean(y)) .^ 2);
r2_LASSO = 1 - RSSLASSO / TSS;
r2_lasso_adj = 1 - (1 - r2_LASSO) * (length(y) - 1) / (length(y) - size(X, 2) - 1);
fprintf('R^2 LASSO: %.4f, Adjusted R^2 LASSO: %.4f\n', r2_LASSO, r2_lasso_adj);

% Results comparison
results_table = table({'Full Model'; 'Stepwise Model'; 'LASSO Model'}, ...
                      [mse_full; mse_stepwise; mse_lasso], ...
                      [r2_full; r2_stepwise_adj; r2_lasso_adj], ...
                      'VariableNames', {'Model', 'MSE', 'Adjusted_R2'});
disp('Model Comparison Results:');
disp(results_table);

% === Analysis without Spike ===
data_with_TMS_NoSpike = removevars(data_with_TMS, 'Spike');

% Ensure all columns are numeric for no-Spike analysis
columns_to_convert = {'Setup', 'Stimuli', 'Intensity', 'Frequency', 'CoilCode'};
for i = 1:length(columns_to_convert)
    column = columns_to_convert{i};
    if iscell(data_with_TMS_NoSpike.(column))
        data_with_TMS_NoSpike.(column) = cellfun(@(x) str2double(x), data_with_TMS_NoSpike.(column), 'UniformOutput', false);
        data_with_TMS_NoSpike.(column) = cell2mat(data_with_TMS_NoSpike.(column));
    end
end

% Select independent variables and dependent variable
indepedent_vars_NoSpike = table2array(data_with_TMS_NoSpike(:, {'Setup', 'Stimuli', 'Intensity', 'Frequency', 'CoilCode'}));
EDduration_NoSpike = data_with_TMS_NoSpike.EDduration;

%Normalization
mx_no_spike = mean(indepedent_vars_NoSpike);
X_no_spike = indepedent_vars_NoSpike - mx_no_spike;
y_no_spike = EDduration_NoSpike - mean(EDduration_NoSpike);

% --- Full Model without Spike ---
lm_full_no_spike = fitlm(X_no_spike, y_no_spike);
y_pred_full_no_spike = predict(lm_full_no_spike, X_no_spike);
mse_full_no_spike = mean((y_no_spike - y_pred_full_no_spike).^2);
r2_full_no_spike_adj = lm_full_no_spike.Rsquared.Adjusted;

% --- Stepwise Regression without Spike ---
lm_stepwise_no_spike = stepwiselm(X_no_spike, y_no_spike, 'Verbose', 0);
%mse_stepwise_no_spike_manual = mean((y_no_spike - predict(lm_stepwise_no_spike, X_no_spike)).^2);
mse_stepwise_no_spike = lm_stepwise_no_spike.MSE;
%fprintf('MSE Stepwise Manual: %.4f, MSE Stepwise: %.4f\n', mse_stepwise_no_spike_manual, mse_stepwise_no_spike);
r2_stepwise_no_spike_adj = lm_stepwise_no_spike.Rsquared.Adjusted;

% --- LASSO Model without Spike ---
rng(seed_value); 
[beta_no_spike, fitinfo_no_spike] = lasso(X_no_spike, y_no_spike, 'CV', 10);
lambda_optimal_no_spike = fitinfo_no_spike.LambdaMinMSE;
disp('Optimal Lambda without Spike:');
disp(lambda_optimal_no_spike);

[~, ilmin_no_spike] = min(abs(fitinfo_no_spike.Lambda - lambda_optimal_no_spike));
bLASSOV_no_spike = beta_no_spike(:, ilmin_no_spike);
mxV_no_spike = mean(X_no_spike);
my_no_spike = mean(y_no_spike);
bLASSOV_no_spike = [my_no_spike - mxV_no_spike * bLASSOV_no_spike; bLASSOV_no_spike];
yfitLASSOV_no_spike = [ones(length(y_no_spike), 1) X_no_spike] * bLASSOV_no_spike;

% Calculate MSE and R^2 for LASSO without Spike
mse_lasso_no_spike = mean((y_no_spike - yfitLASSOV_no_spike).^2);
r2_lasso_no_spike = 1 - sum((y_no_spike - yfitLASSOV_no_spike).^2) / sum((y_no_spike - mean(y_no_spike)).^2);
r2_lasso_no_spike_adj = 1 - (1 - r2_lasso_no_spike) * (length(y_no_spike) - 1) / (length(y_no_spike) - size(X_no_spike, 2) - 1);


% Results comparison without Spike
results_table_no_spike = table({'Full Model'; 'Stepwise Model'; 'LASSO Model'}, ...
                               [mse_full_no_spike; mse_stepwise_no_spike; mse_lasso_no_spike], ...
                               [r2_full_no_spike_adj; r2_stepwise_no_spike_adj; r2_lasso_no_spike_adj], ...
                               'VariableNames', {'Model', 'MSE', 'Adjusted_R2'});

disp('Model Comparison Results without Spike:');
disp(results_table_no_spike); 

% --- With Spike ---
% Note: For the analysis with Spike, we used only rows with valid Spike values, reducing the dataset.
% - The Full Model has the lowest MSE (58.241) but a low adjusted R^2 (0.0407), indicating it does not fit the data well despite lower errors.
% - The Stepwise Model has a slightly higher MSE (61.078) and a slightly lower adjusted R^2 (0.0363), showing it is less suitable than the Full Model.
% - The LASSO Model has the highest MSE (61.725) and the lowest adjusted R^2, indicating it is less effective compared to the other two models.

% --- Without Spike ---
% Note: For the analysis without Spike, we used the full dataset but removed the Spike variable.
% - The Full Model has a much higher MSE (106.26) and a modest adjusted R^2 (0.2425), showing a decline in its performance without Spike.
% - The Stepwise Model demonstrates the best performance with the lowest MSE (83.753) and the highest adjusted R^2 (0.4281), making it the most suitable model.
% - The LASSO Model has the same MSE as the Full Model (106.26) but achieves a slightly higher adjusted R^2 (0.2682), making it better than the Full Model but less effective than the Stepwise Model.

% --- Final Recommendation ---
% - Use the Full Model when Spike is included, as it provides the lowest MSE, despite its low adjusted R^2.
% - Use the Stepwise Model without Spike, as it provides the best adjusted R^2 and lowest MSE, making it the most effective choice.
% - The LASSO Model does not perform as well in either case, making it a less suitable choice overall.
