function plot_with_best_fit(data_struct, bin_edges, best_fit_dist)
    data = data_struct.data;
    data_name = data_struct.dataname;
    color = data_struct.color;

    % Plot histogram of the data
    histogram(data, 'BinEdges', bin_edges, 'Normalization', 'pdf', 'DisplayName', data_name, 'FaceAlpha', 0.8);
    hold on;
    
    % Generate x values for the best fit distribution
    x_values = linspace(min(data), max(data), 100);
    
    % Evaluate the best fit distribution at the x values
    y_values = pdf(best_fit_dist, x_values);
    
    % Plot the best fit distribution
    plot(x_values, y_values, 'LineWidth', 2, 'DisplayName', ['Best Fit - ' data_name], 'Color', color);
    
    % Doesn't release the hold
end