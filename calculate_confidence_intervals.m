function [ci, bootstrap_means, p_values] = calculate_confidence_intervals(data, num_resamples, useTMS)
    % Initialize variables
    ci = cell(1, 6);
    p_values = zeros(1, 6);
    bootstrap_means = zeros(num_resamples, 6);

    % Split the data into 6 samples based on the Setup variable
    ED_samples = cell(1, 6);

    % Loop through all 6 setups and calculate confidence intervals
    for setup_num = 1:6
        ED_samples{setup_num} = data.EDduration(data.TMS == useTMS & data.Setup == setup_num);
        % Test for normality in each setup
        % Fit a normal distribution to the data
        pd = fitdist(ED_samples{setup_num}, 'Normal');
        fprintf('%d. length of data: %d\n', setup_num, length(ED_samples{setup_num}));
        mu_setup = pd.mu;
        sigma_setup = pd.sigma;
        norm_cdf = @(x) cdf(pd, x);

        [hypothesis, p_values(setup_num)] = chi2gof(ED_samples{setup_num}, 'CDF', norm_cdf, 'Alpha', 0.05);


        % If the data is not normally distributed, perform bootstrap resampling
        if hypothesis == 1
            % Bootstrap for ED without TMS
            for i = 1:num_resamples
                bootstrap_means(i, setup_num) = mean(datasample(ED_samples{setup_num}, length(ED_samples{setup_num})));
            end
            % Confidence intervals for ED without TMS
            ci{setup_num} = prctile(bootstrap_means(:, setup_num), [2.5, 97.5]);
        else
            fprintf('The data for Setup %d is normally distributed.\n', setup_num);
            % If the data is normally distributed, calculate the confidence intervals directly
            ci{setup_num} = norminv([0.025, 0.975], mu_setup, sigma_setup);
        end

    end
end
