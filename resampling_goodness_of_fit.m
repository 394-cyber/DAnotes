% RESAMPLING_GOODNESS_OF_FIT Resamples data and computes the chi-squared goodness of fit.
% 
%   Inputs:
%       data        - A vector of observed data points.
%       lambda      - The lambda parameter for the exponational distribution.
%       num_samples - The number of resampled datasets to generate.
% 
%   Outputs:
%       chi2_resampled - A vector of chi-squared statistics for each resampled dataset.

function chi2_resampled = resampling_goodness_of_fit(data, lambda, num_samples)

    % Resampling
    chi2_resampled = zeros(1, num_samples);
    for i = 1:num_samples
        resampled_data = exprnd(1/lambda, size(data));
        [~, ~, stats] = chi2gof(resampled_data, 'CDF', @(x)expcdf(x, 1/lambda), 'nparams', 1);
        chi2_resampled(i) = stats.chi2stat;
    end
end