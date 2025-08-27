% Function for simple random sampling
function sample_days = simple_random_sample(demand_matrix, h1, h2, n)
    valid_days = get_valid_days(demand_matrix, h1, h2);
    sample_days = randsample(valid_days, n, false);
end