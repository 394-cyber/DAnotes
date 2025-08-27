%Group40Exe4

% Read the data from the file
current_file_path = mfilename('fullpath');
[parent_folder, ~, ~] = fileparts(fileparts(current_file_path));
data_path = fullfile(parent_folder, 'TMS.xlsx');
if ~exist(data_path, 'file')
    error('The file TMS.xlsx does not exist in the specified path: %s', data_path);
end
data = readtable(data_path);  

setup_numbers = 1:6;
alpha = 0.05;

% Initialize variables for result collection
true_r = zeros(length(setup_numbers), 1);
p_parametric = zeros(length(setup_numbers), 1);
p_randomization = zeros(length(setup_numbers), 1);
num_random_samples = 1000;

% Loop through each setup number
fprintf('\n===========================\n');
fprintf('Parametric and Randomization Test Results:\n');
fprintf('===========================\n');
for i = 1:length(setup_numbers)
    setup_num = setup_numbers(i);
    
    % Extracting the data for the current setup
    preTMS = data.preTMS(data.Setup == setup_num);
    postTMS = data.postTMS(data.Setup == setup_num);

    % Remove NaN values
    preTMS = preTMS(~isnan(preTMS));
    postTMS = postTMS(~isnan(postTMS));

    % Perform parametric test
    [r, p] = corr(preTMS, postTMS, 'Type', 'Pearson');
    p_parametric(i) = p;
    true_r(i) = r;

    % Randomization test
    results = randomization(num_random_samples, preTMS, postTMS);
    % Calculate the empirical p-value
    p_randomization(i) = mean(abs(results.r) >= abs(r));

    % Display results in an easy-to-read format
    fprintf('\nSetup %d:\n', setup_num);
    fprintf('  Pearson Correlation (r): %.4f\n', r);
    fprintf('  Parametric p-value     : %.4f\n', p_parametric(i));
    fprintf('  Randomization p-value  : %.4f\n', p_randomization(i));
    fprintf('---------------------------\n');

    % Plot histogram of randomization results
    figure;
    histogram(results.r, 'NumBins', round(sqrt(num_random_samples)));
    title(sprintf('Randomization Test Results for Setup %d', setup_num));
    hold on;
    y_limits = ylim;
    plot([r, r], y_limits, 'r', 'LineWidth', 2);
    lower_bound = prctile(results.r, alpha/2 * 100);
    upper_bound = prctile(results.r, (1 - alpha/2) * 100);
    plot([lower_bound, lower_bound], y_limits, 'g--', 'LineWidth', 2);
    plot([upper_bound, upper_bound], y_limits, 'g--', 'LineWidth', 2);
    hold off;
    xlabel('r Value');
    ylabel('Frequency');
    legend('Random r Values', 'True r Value', 'Alpha 0.05 Interval');
end

% COMMENTS ON RESULTS

% Setup 1:
% Pearson r = 0.4548, Parametric p = 0.1599, Randomization p = 0.1470
% There is no statistically significant correlation (p > 0.05), but the moderate 
% r-value suggests a potential positive trend.

% Setup 2:
% Pearson r = -0.1693, Parametric p = 0.3235, Randomization p = 0.2980
% Weak negative correlation

% Setup 3:
% Pearson r = 0.0111, Parametric p = 0.9619, Randomization p = 0.964
% Practically no correlation; no evidence of a relationship between preTMS and postTMS.

% Setup 4:
% Pearson r = 0.3404, Parametric p = 0.0610, Randomization p = 0.0610
% Close to the significance threshold (p = 0.05). Moderate positive correlation.

% Setup 5:
% Pearson r = -0.1483, Parametric p = 0.6455, Randomization p = 0.5740
% Very weak negative correlation

% Setup 6:
% Pearson r = -0.2755, Parametric p = 0.5090, Randomization p = 0.4140
% Weak negative correlation

% SUMMARY:
% - None of the setups exhibit statistically significant correlation, based on p-value critiria (p>0.05). 
% - Pvalue is a matrix of p-values for testing the hypothesis of no correlation against the alternative that there is a non-zero correlation. 
% - Setup 4 shows the strongest indication of correlation.
% - Parametric and randomization p-values are consistent, suggesting agreement between the two methods.
% - For these small sample sizes, the randomization test is more reliable.
