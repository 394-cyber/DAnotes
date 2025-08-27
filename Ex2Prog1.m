% Load the data
data = load('DataEx2.dat');

% Extract magnitude and date columns
M = data(:,4);
years = data(:,1);
months = data(:,2);
days = data(:,3);

% Total number of earthquakes
n = length(M);

% (α') Calculate N(M)
tab = tabulate(M);
unique_M = tab(:,1);
counts = tab(:,2);

% Sort unique magnitudes ascending
[sorted_M, idx] = sort(unique_M);
sorted_counts = counts(idx);

% Calculate complementary cumulative frequency N(M) = number of earthquakes with magnitude >= M
N_of_M = flipud(cumsum(flipud(sorted_counts)));

% (β') Linear regression on model log10(N) = a - b*M using regress function
logN = log10(N_of_M);
X = [ones(length(sorted_M),1), -sorted_M];
[coeffs,~,residuals,~,stats] = regress(logN, X);
a = coeffs(1);
b = coeffs(2);

% Model estimates
logN_pred = X * coeffs;

% Coefficient of determination R^2 from regress output
R2 = stats(1);

% Normalized residuals
norm_res = residuals / std(residuals);

% Plot N vs M with regression curve
figure;
scatter(sorted_M, N_of_M, 'filled');
hold on;
plot(sorted_M, 10.^logN_pred, 'r', 'LineWidth', 2);
xlabel('Magnitude M');
ylabel('Number of earthquakes N(M)');
title('Number of earthquakes vs Magnitude and Regression Fit');
grid on;

% Plot normalized residuals vs M
figure;
scatter(sorted_M, norm_res);
xlabel('Magnitude M');
ylabel('Normalized Residuals');
title('Normalized Residuals of Regression vs Magnitude');
grid on;

% Hypothesis test for b=1 (two-sided t-test)
SE_b = sqrt(stats(4)); % estimate of variance times covariance matrix diag element for b
t_stat = (b - 1) / SE_b;
df = length(sorted_M)-2;
p_val = 2 * (1 - tcdf(abs(t_stat), df));

fprintf('a = %f\n', a);
fprintf('b = %f\n', b);
fprintf('R^2 = %f\n', R2);
fprintf('t-statistic for b=1: %f\n', t_stat);
fprintf('p-value for test b=1: %f\n', p_val);

% (γ') Multiple linear regression with previous k earthquake magnitudes using regress function
k_list = [5, 10, 20];

% Create datetime array for time ordering
dates = datetime(years, months, days);

% Sort data by date (stable sorting preserves order within same date)
[sorted_dates, sort_idx] = sort(dates);
M_sorted = M(sort_idx);

% Define training period 2006-2008 and test period 2009
train_mask = (sorted_dates >= datetime(2006,1,1)) & (sorted_dates <= datetime(2008,12,31));
test_mask = (sorted_dates >= datetime(2009,1,1)) & (sorted_dates <= datetime(2009,12,31));

y_train = M_sorted(train_mask);
y_test = M_sorted(test_mask);

for i = 1:length(k_list)
    k = k_list(i);
    
    % Prepare design matrices for training and testing
    X_train = zeros(length(y_train)-k, k);
    X_test = zeros(length(y_test)-k, k);
    y_train_sub = y_train(k+1:end);
    y_test_sub = y_test(k+1:end);
    
    for j = 1:k
        X_train(:,j) = y_train(k+1-j:end-j);
        X_test(:,j) = y_test(k+1-j:end-j);
    end
    
    % Add intercept term
    X_train_intercept = [ones(size(X_train,1),1), X_train];
    X_test_intercept = [ones(size(X_test,1),1), X_test];
    
    % Full model linear regression
    [coeffs_full,~,residuals_train_full,~,stats_train_full] = regress(y_train_sub, X_train_intercept);
    y_train_pred_full = X_train_intercept * coeffs_full;
    
    % Adjusted R squared for train full model
    p_full = size(X_train_intercept,2);
    adjR2_full_train = 1 - (1 - stats_train_full(1))*(length(y_train_sub) - 1)/(length(y_train_sub) - p_full);
    
    % Sparse model using LASSO with 10-fold cross-validation
    [B, FitInfo] = lasso(X_train, y_train_sub, 'CV', 10);
    idxLambda1SE = FitInfo.Index1SE;
    coeffs_sparse = B(:, idxLambda1SE);
    intercept_sparse = FitInfo.Intercept(idxLambda1SE);
    
    y_train_pred_sparse = X_train * coeffs_sparse + intercept_sparse;
    residuals_train_sparse = y_train_sub - y_train_pred_sparse;
    
    % Adjusted R squared for train sparse model
    SS_res_sparse = sum(residuals_train_sparse.^2);
    SS_tot_sparse = sum((y_train_sub - mean(y_train_sub)).^2);
    p_sparse = nnz(coeffs_sparse) + 1; % nonzero coeff + intercept
    adjR2_sparse_train = 1 - (SS_res_sparse/(length(y_train_sub)-p_sparse)) / (SS_tot_sparse/(length(y_train_sub)-1));
    
    % Predict test data
    y_test_pred_full = X_test_intercept * coeffs_full;
    y_test_pred_sparse = X_test * coeffs_sparse + intercept_sparse;
    residuals_test_full = y_test_sub - y_test_pred_full;
    residuals_test_sparse = y_test_sub - y_test_pred_sparse;
    
    SS_res_test_full = sum(residuals_test_full.^2);
    SS_res_test_sparse = sum(residuals_test_sparse.^2);
    SS_tot_test = sum((y_test_sub - mean(y_test_sub)).^2);
    
    adjR2_full_test = 1 - (SS_res_test_full/(length(y_test_sub)-p_full)) / (SS_tot_test/(length(y_test_sub)-1));
    adjR2_sparse_test = 1 - (SS_res_test_sparse/(length(y_test_sub)-p_sparse)) / (SS_tot_test/(length(y_test_sub)-1));
    
    % Plot normalized residuals for full model (train)
    figure;
    scatter(y_train_sub, residuals_train_full/std(residuals_train_full));
    xlabel('Earthquake Magnitude y');
    ylabel('Normalized Residuals (Full Model)');
    title(['Normalized Residuals vs Magnitude (Full Model), k=' num2str(k)]);
    grid on;
    
    % Plot normalized residuals for sparse model (train)
    figure;
    scatter(y_train_sub, residuals_train_sparse/std(residuals_train_sparse));
    xlabel('Earthquake Magnitude y');
    ylabel('Normalized Residuals (Sparse Model)');
    title(['Normalized Residuals vs Magnitude (Sparse Model), k=' num2str(k)]);
    grid on;
    
    fprintf('k=%d: adjR2 train full=%.4f, sparse=%.4f; adjR2 test full=%.4f, sparse=%.4f\n', ...
        k, adjR2_full_train, adjR2_sparse_train, adjR2_full_test, adjR2_sparse_test);
end