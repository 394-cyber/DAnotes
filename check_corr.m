function check_corr(preTMS, postTMS, alpha)
    % Check if the preTMS and postTMS data is related
    % H0: The preTMS and postTMS data is not related
    % H1: The preTMS and postTMS data is related


    % Calculate the correlation coefficient
    [r, p] = corr(preTMS, postTMS, 'Type', 'Pearson');
    % t_stat follows student distribution with n-2 degrees of freedom
    t_stat = r * sqrt((length(preTMS) - 2) / (1 - r^2));

    % p_value can be taken directly from the p
    % Alternatively, we can calculate it using the formula: p = 2 * (1 - F(|t|))
    % where F is the cumulative distribution function of the student distribution
    p_value = 2 * (1 - tcdf(abs(t_stat), length(preTMS) - 2));

    % Verify p_value is the same as the one calculated by corr
    fprintf('p_value = %.4f and p = %.4f\n', p_value, p);

    % Reject the null hypothesis if p_value < alpha
    if p_value < alpha
        fprintf('The preTMS and postTMS data is correlated\n');
    else
        fprintf('The preTMS and postTMS data is not correlated\n');
    end
end
