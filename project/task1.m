%Group40Exe1

% Read the data from the file
current_file_path = mfilename('fullpath');
[parent_folder, ~, ~] = fileparts(fileparts(current_file_path));
data_path = fullfile(parent_folder, 'TMS.xlsx');
if ~exist(data_path, 'file')
    error('The file TMS.xlsx does not exist in the specified path: %s', data_path);
end
data = readtable(data_path);  

% Extract relevant columns for ED duration with and without TMS
ED_with_TMS = data.EDduration(data.TMS == 1);
ED_without_TMS = data.EDduration(data.TMS == 0);

% Remove missing values to be safe
ED_with_TMS = ED_with_TMS(~isnan(ED_with_TMS));
ED_without_TMS = ED_without_TMS(~isnan(ED_without_TMS));


% List of distributions to test
dist_names = {'BirnbaumSaunders', 'Burr', 'Exponential', 'Extreme Value', 'Gamma', 'Generalized Extreme Value', 'Generalized Pareto', 'Half Normal', 'InverseGaussian', 'Logistic', 'Loglogistic', 'Lognormal', 'Nakagami', 'Normal', 'Poisson', 'Rayleigh', 'Rician', 'tLocationScale', 'Weibull'};

warning('off', 'all'); % Suppress warnings for better output readability
% Perform goodness of fit test for ED duration with TMS
[best_fit_with, ~, ~] = test_goodness_of_fit(ED_with_TMS, dist_names);
[best_fit_without, ~, ~] = test_goodness_of_fit(ED_without_TMS, dist_names);


num_bins = round(sqrt(length(ED_with_TMS))); % Number of bins (rule of thumb)
bin_edges = linspace(min([ED_with_TMS; ED_without_TMS]), max([ED_with_TMS; ED_without_TMS]), num_bins);

ED_with_TMS_struct = struct('data', ED_with_TMS, 'dataname', 'ED with TMS', 'color', 'r');
ED_without_TMS_struct = struct('data', ED_without_TMS, 'dataname', 'ED without TMS', 'color', 'b');

figure;
plot_with_best_fit(ED_with_TMS_struct, bin_edges, best_fit_with);
plot_with_best_fit(ED_without_TMS_struct, bin_edges, best_fit_without);

xlabel('Time(s)');
ylabel('Probability Density');
title('Histogram and Best Fit Distribution for ED Duration');
legend('show')
hold off;

% Conclusion
% The best fit appears to be the Exponential distribution for both the TMS and the Without TMS data, as determined by the p-value criteria. 
% The empirical probability density functions (PDFs) in the plotted diagram show minimal deviation between the two distributions.
% In addition, the best-fit distribution appears to be identical for both datasets