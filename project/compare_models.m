function results_table = compare_models(X, y)
    %% Full model
    lm_full = fitlm(X, y);
    mse_full = lm_full.MSE;
    r2_full_adj = lm_full.Rsquared.Adjusted;

    %% Stepwise model
    lm_stepwise = stepwiselm(X, y, 'Verbose', 0);
    Y_pred_stepwise = predict(lm_stepwise, X);
    mse_stepwise = mean((y - Y_pred_stepwise).^2);
    r2_stepwise_adj = lm_stepwise.Rsquared.Adjusted;

    %% LASSO
    rng(1); % For reproducibility
    [beta, fitinfo] = lasso(X, y, 'CV', 10);
    lambda_optimal = fitinfo.LambdaMinMSE;
    [~, ilmin] = min(abs(fitinfo.Lambda - lambda_optimal));
    bLASSOV = beta(:, ilmin);
    bLASSOV = [mean(y) - mean(X) * bLASSOV; bLASSOV];
    yfitLASSOV = [ones(length(y), 1) X] * bLASSOV;

    % Calculate the MSE and R^2
    res_lasso = y - yfitLASSOV;
    RSS_lasso = sum(res_lasso .^ 2);
    mse_lasso = mean(res_lasso .^ 2);
    TSS = sum((y - mean(y)) .^ 2);
    r2_lasso = 1 - RSS_lasso / TSS;
    r2_lasso_adj = 1 - (1 - r2_lasso) * (length(y) - 1) / (length(y) - size(X, 2) - 1);

    %% PCR
    n = length(X);
    [PCALoadings, PCAScores, PCAVar] = pca(X, 'Economy', false);
    desiredVariance = 0.95;
    cumVar = cumsum(PCAVar) / sum(PCAVar);
    numComponents = find(cumVar >= desiredVariance, 1);

    betaPCR = regress(y - mean(y), PCAScores(:, 1:numComponents));
    betaPCR = PCALoadings(:, 1:numComponents) * betaPCR;
    betaPCR = [mean(y) - mean(X) * betaPCR; betaPCR];
    yfitPCR = [ones(n, 1) X] * betaPCR;

    % Calculate the MSE and R^2
    RSS_PCR = sum((y - yfitPCR) .^ 2);
    mse_PCR = mean((y - yfitPCR) .^ 2);
    r2_PCR = 1 - RSS_PCR / TSS;
    r2_PCR_adj = 1 - (1 - r2_PCR) * (length(y) - 1) / (length(y) - size(X, 2) - 1);

    %% Print the results in a table
    results_table = table({'Full Model'; 'Stepwise Model'; 'LASSO Model'; 'PCR Model'}, ...
                          [mse_full; mse_stepwise; mse_lasso; mse_PCR], ...
                          [r2_full_adj; r2_stepwise_adj; r2_lasso_adj; r2_PCR_adj], ...
                          'VariableNames', {'Model', 'MSE', 'Adjusted_R2'});

    disp('Model Comparison Results:');
    disp(results_table);
end
