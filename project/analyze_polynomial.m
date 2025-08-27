% Polynomial analysis
function analyze_polynomial(data, TMS_value)
    % Filter the data for the specified TMS value
    data_filtered = data(data.TMS == TMS_value, :);
    
    % Extract EDduration and convert Setup to numeric form
    EDduration = data_filtered.EDduration;
    Setup_numeric = double(categorical(data_filtered.Setup));
    
    % Prepare polynomial degrees for analysis
    degrees = [1, 2, 3]; % quadratic and cubic models
    R2_values = zeros(length(degrees), 1); % Store R^2 for each model
    
    for i = 1:length(degrees)
        degree = degrees(i);
        
        % Create polynomial design matrix
        xM = ones(length(Setup_numeric), 1); % Start with intercept
        for d = 1:degree
            xM = [xM, Setup_numeric.^d];
        end
        
        % Perform regression
        [b, ~, r, ~, stats] = regress(EDduration, xM);
        R2_values(i) = stats(1); % Store R^2
        
        % Compute predicted values
        yhat = xM * b;
        
        % Standardize residuals
        r_standardized = r / std(r);
        
        % Create figure with subplots
        figure;
        
        % First subplot: Data and fitted polynomial model
        subplot(2, 1, 1);
        scatter(Setup_numeric, EDduration, 'filled');
        hold on;
        plot(sort(Setup_numeric), sort(yhat), '-r', 'LineWidth', 1.5);
        xlabel('Setup (numeric)');
        ylabel(sprintf('EDduration (TMS = %d)', TMS_value));
        title(sprintf('Polynomial Regression (Degree %d) for TMS = %d', degree, TMS_value));
        legend('Data points', 'Fitted polynomial');
        grid on;
        hold off;
        
        % Second subplot: Residuals
        subplot(2, 1, 2);
        plot(r_standardized, 'o');
        hold on;
        yline(2, '--r');
        yline(-2, '--r');
        title(sprintf('Residuals for Polynomial Model (Degree %d) for TMS = %d', degree, TMS_value));
        xlabel('Observation');
        ylabel('Standardized Residual');
        grid on;
        hold off;
    end
    
    
    % Display R^2 values for all polynomial degrees
    fprintf('Polynomial Models - R^2 values for TMS = %d:\n', TMS_value);
    for i = 1:length(degrees)
        fprintf('  Degree %d: R^2 = %.3f\n', degrees(i), R2_values(i));
    end
end