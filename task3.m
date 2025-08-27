% Group40Exe3

% Read the data from the file
warning('off', 'all');
current_file_path = mfilename('fullpath');
[parent_folder, ~, ~] = fileparts(fileparts(current_file_path));
data_path = fullfile(parent_folder, 'TMS.xlsx');
if ~exist(data_path, 'file')
    error('The file TMS.xlsx does not exist in the specified path: %s', data_path);
end
data = readtable(data_path);

% Extract ED duration data for both TMS and no TMS
ED_without_TMS = data.EDduration(data.TMS == 0);
ED_with_TMS = data.EDduration(data.TMS == 1);

% Calculate the mean ED duration for both cases (without TMS and with TMS)
mu_without_TMS = mean(ED_without_TMS);
mu_with_TMS = mean(ED_with_TMS);
sigma_without_TMS = std(ED_without_TMS);
sigma_with_TMS = std(ED_with_TMS);

num_resamples = 1000;

% Without TMS
useTMS = 0;
[ci_without_TMS, bootstrap_means_without_TMS, p_values_without_TMS] = calculate_confidence_intervals(data, num_resamples, useTMS);

% With TMS
useTMS = 1;
[ci_with_TMS, bootstrap_means_with_TMS, p_values_with_TMS] = calculate_confidence_intervals(data, num_resamples, useTMS);

results_table = table('Size', [6, 4], 'VariableTypes', {'cell', 'string', 'cell', 'string'}, 'VariableNames', {'CI_Without_TMS', 'Accepted_Without_TMS', 'CI_With_TMS', 'Accepted_With_TMS'});

% Setup the results table
for setup_num=1:6
    % Populate the table with the confidence intervals and acceptance status
    results_table.CI_Without_TMS{setup_num} = ci_without_TMS{setup_num};
    results_table.CI_With_TMS{setup_num} = ci_with_TMS{setup_num};

    % Check if the mean ED duration is within the confidence intervals
    if mu_without_TMS >= ci_without_TMS{setup_num}(1) && mu_without_TMS <= ci_without_TMS{setup_num}(2)
        results_table.Accepted_Without_TMS(setup_num) = "Yes";
    else
        results_table.Accepted_Without_TMS(setup_num) = "No";
    end

    if mu_with_TMS >= ci_with_TMS{setup_num}(1) && mu_with_TMS <= ci_with_TMS{setup_num}(2)
        results_table.Accepted_With_TMS(setup_num) = "Yes";
    else
        results_table.Accepted_With_TMS(setup_num) = "No";
    end

end

% Display the table with the results
fprintf('Mean ED Duration without TMS: %.2f seconds\n', mu_without_TMS);
fprintf('Mean ED Duration with TMS: %.2f seconds\n', mu_with_TMS);
disp(results_table);


% Plot the Chi-square Goodness-of-Fit Test p-values for both "Without TMS" and "With TMS"
figure;
bar([p_values_without_TMS; p_values_with_TMS]', 'grouped');
title('Chi-square Goodness-of-Fit Test p-values');
set(gca, 'xticklabel', {'Setup 1', 'Setup 2', 'Setup 3', 'Setup 4', 'Setup 5', 'Setup 6'});
ylabel('p-value');
ylim([0 1]);
legend({'No TMS', 'With TMS'});

% -------------------- Plotting --------------------

% 1. Histograms for ED durations (with and without TMS)
figure;
subplot(2, 2, 1);
histogram(ED_without_TMS, 'Normalization', 'pdf');
hold on;
x = linspace(min(ED_without_TMS), max(ED_without_TMS), 100);
plot(x, normpdf(x, mu_without_TMS, sigma_without_TMS), 'r', 'LineWidth', 2);
title('ED Duration without TMS');
xlabel('Duration (seconds)');
ylabel('Probability Density');
legend('Data', 'Normal Fit');

subplot(2, 2, 2);
histogram(ED_with_TMS, 'Normalization', 'pdf');
hold on;
plot(x, normpdf(x, mu_with_TMS, sigma_with_TMS), 'r', 'LineWidth', 2);
title('ED Duration with TMS');
xlabel('Duration (seconds)');
ylabel('Probability Density');
legend('Data', 'Normal Fit');

% 2. Bootstrap Resampling Results for Confidence Intervals
subplot(2, 2, 3);
boxplot(bootstrap_means_without_TMS, 'Labels', {'Setup 1', 'Setup 2', 'Setup 3', 'Setup 4', 'Setup 5', 'Setup 6'});
title('Bootstrap Means (No TMS)');
ylabel('Mean ED Duration');
xlabel('Setup');

subplot(2, 2, 4);
boxplot(bootstrap_means_with_TMS, 'Labels', {'Setup 1', 'Setup 2', 'Setup 3', 'Setup 4', 'Setup 5', 'Setup 6'});
title('Bootstrap Means (With TMS)');
ylabel('Mean ED Duration');
xlabel('Setup');

% -------------------- Conclusion --------------------
% General Agreement:
% - Setups 1, 2, 3, 5, and 6 accept H0 in both conditions (with and without TMS),
%   indicating that their mean values do not significantly differ from μ0.
% - Setup 4 rejects H0 for the TMS condition, suggesting a potential effect of TMS.

% Variance Differences:
% - Setups 1, 3 and 6 have exceptionally large confidence intervals,
%   which reduces the reliability of the conclusions.
% - This is likely due to small sample sizes or high variability in the data.

% Overall TMS Effect:
% - TMS does not appear to significantly affect the mean ED durations in most setups

% Data Issues:
% - Negative bounds in some Setups indicate that the analysis might require more data.
% - With more data, the fit may improve, leading to smaller variance.
