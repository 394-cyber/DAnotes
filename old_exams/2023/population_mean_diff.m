
% Function to compute population mean difference
function pop_diff = population_mean_diff(demand_matrix, h1, h2)
    valid_days = get_valid_days(demand_matrix, h1, h2);
    pop_diff = mean(demand_matrix(valid_days, h1+1) - demand_matrix(valid_days, h2+1));
end
