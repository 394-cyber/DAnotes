% Function to get valid days for a pair of hours
function valid_days = get_valid_days(demand_matrix, h1, h2)
    valid = ~(isnan(demand_matrix(:, h1+1)) | isnan(demand_matrix(:, h2+1)));
    valid_days = find(valid);
end