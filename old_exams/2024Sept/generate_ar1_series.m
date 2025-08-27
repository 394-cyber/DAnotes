function time_series = generate_ar1_series(n, phi, sigma, x0)
% GENERATE_AR1_SERIES Generate AR(1) time series
%   n: length of final time series
%   phi: autoregressive coefficient (-1 < phi < 1)
%   sigma: standard deviation of noise
%   x0: initial value
    
    % Generate extra observations to discard initial transient
    total_obs = n + 20;
    
    % Initialize time series
    x = zeros(total_obs, 1);
    x(1) = x0;
    
    % Generate noise terms
    epsilon = sigma * randn(total_obs, 1);
    
    % Generate AR(1) process
    for t = 2:total_obs
        x(t) = phi * x(t-1) + epsilon(t);
    end
    
    % Discard first 20 observations
    time_series = x(21:end);
end