% Function for stratified random sampling
function sample_days = stratified_random_sample(demand_matrix, holiday_flags, h1, h2, n)
    valid_days = get_valid_days(demand_matrix, h1, h2);
    
    % Separate holiday and non-holiday days
    holiday_valid = valid_days(holiday_flags(valid_days) == 1);
    non_holiday_valid = valid_days(holiday_flags(valid_days) == 0);
    
    % Calculate proportions
    p1 = length(holiday_valid) / length(valid_days);
    n1 = round(n * p1);
    n0 = n - n1;
    
    % Adjust if strata are insufficient
    n1 = min(n1, length(holiday_valid));
    n0 = min(n0, length(non_holiday_valid));
    
    if n1 + n0 < n
        n0 = n - n1; % Adjust n0 to meet sample size
    end
    
    % Sample from each stratum
    sample_holiday = randsample(holiday_valid, n1, false);
    sample_non_holiday = randsample(non_holiday_valid, n0, false);
    sample_days = [sample_holiday; sample_non_holiday];
end