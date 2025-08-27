%% DATA ANALYSIS CHEATSHEET — EXPANDED WITH PARAMS/RETURNS + IMPROVEMENTS
% This script demonstrates core data analysis tasks with explicit comments
% explaining function parameters and return values, plus improved usages
% where appropriate and descriptive titles on figures.

rng default  % Seed the random number generator for reproducibility.

%% 1) DISTRIBUTIONS AND SAMPLING
n = 1000; M = 1;                     % n: rows (observations), M: columns (replicates)
tau = 0.8;                           % tau: exponential rate (>0). Note: exprnd expects MEAN.
mu_exp = 1/tau;                      % Exponential mean (mu) = 1/tau to match f(x)=tau*exp(-tau*x)

% exprnd(mu, n, M): Draws exponential random numbers with mean 'mu' (>0).
% Returns an n-by-M matrix of samples.
X_exp = exprnd(mu_exp, n, M);

a = -1; b = 2;                       % Uniform bounds
% unifrnd(a,b,n,M): Samples Uniform(a,b). Returns n-by-M matrix.
X_unif = unifrnd(a, b, n, M);

lambda = 3;                          % Poisson mean (>=0)
% poissrnd(lambda,n,M): Samples Poisson counts with mean 'lambda'.
% Returns an n-by-M integer matrix.
X_pois = poissrnd(lambda, n, M);

% rand(n,M): Samples i.i.d. Uniform(0,1).
% Returns an n-by-M matrix of samples.
X_u01 = rand(n, M);

%% 2) HISTOGRAMS AND ANALYTIC VS EMPIRICAL PDF

figure('Name','Exponential: Empirical vs Analytic PDF'); clf
% histogram(...,'Normalization','pdf'): Plots density estimate (area=1). Returns histogram object.
h = histogram(X_exp, 'Normalization','pdf', 'FaceAlpha',0.25, 'EdgeColor','none');
hold on
% fitdist(data,'exponential'): Fits an Exponential(mean) distribution object ('prob.FittedDist').
% Returns a prob.ExponentialDistribution with properties (mu) and methods (pdf,cdf,icdf,random).

dExp = fitdist(X_exp, 'exponential'); 
xgrid = linspace(max(0,min(X_exp)), max(X_exp), 400)';

% pdf(d,x): Evaluates PDF of fitted distribution 'd' at x. Returns vector of pdf values.
plot(xgrid, pdf(dExp, xgrid), 'r-', 'LineWidth', 2)

% exppdf(x,mu): Analytic exponential PDF with MEAN 'mu'. Returns density values.
plot(xgrid, exppdf(xgrid, mu_exp), 'k--', 'LineWidth', 1.25)
title('Exponential Samples: Empirical PDF vs Fitted and Analytic')
legend('Empirical (hist pdf)','Fitted: fitdist','Analytic: exppdf(\mu=1/\tau)','Location','best')
grid on

%% 3) DESCRIPTIVE STATISTICS
% mean(X): Sample mean (ignores NaN if 'omitnan' used). Returns scalar or vector by dim.
muX   = mean(X_exp);
% var(X): Sample variance (normalizes by n-1). Returns scalar or vector by dim.
sig2X = var(X_exp);
% std(X): Sample standard deviation (sqrt(var)). Returns scalar or vector by dim.
sigX  = std(X_exp);
% median(X): Sample median. Returns scalar or vector by dim.
medX  = median(X_exp);

%% 4) MLE EXAMPLES
% mle(X,'distribution','poisson'): MLE for Poisson mean. Returns parameter estimate(s) in 'phat'.
phat_pois = mle(X_pois, 'distribution','poisson');

% mle with custom pdf/cdf:
% mle(data,'pdf',pdfHandle,'cdf',cdfHandle,'start',theta0) fits a custom distribution.
% 'pdf'/'cdf' are function handles mapping (data,theta) -> densities/CDFs.
% Returns parameter estimates 'theta_hat' (vector if multiple params).
custom_pdf = @(x,theta) (1/theta).*exp(-x./theta).*(x>=0);
custom_cdf = @(x,theta) (1 - exp(-x./theta)).*(x>=0);
theta0 = mean(X_exp); % Initial guess for scale
theta_hat = mle(X_exp, 'pdf',custom_pdf, 'cdf',custom_cdf, 'start',theta0);

%% 5) GOODNESS-OF-FIT (NORMAL AS EXAMPLE)
% chi2gof(X,'CDF',F,'NParams',k): Chi-square GOF vs target CDF handle F.
% Returns [h,p,stats] where stats.O/E are observed/expected bin counts.
F_norm = @(z) normcdf(z, mean(X_unif), std(X_unif));
[h_gof,p_gof,stats_gof] = chi2gof(X_unif, 'CDF',F_norm, 'NParams',2);

%% 6) PARAMETRIC TESTS (MEAN/VARIANCE; ONE- AND TWO-SAMPLE)
alpha = 0.05;

test_value = 1.2;  % Null mean for one-sample test
% ttest(X,mu0,'Alpha',a,'Tail','both'): One-sample t-test of mean==mu0.
% Returns [h,p,ci,stats], with stats.tstat and stats.df.
[H1,P1,CI1,stats1] = ttest(X_exp, test_value, 'Alpha',alpha, 'Tail','both');

% Two-sample: independent samples with potential unequal variances by default (Welch's).
% ttest2(X,Y,'Alpha',a): Two-sample t-test of mean(X)==mean(Y).
% Returns [h,p,ci,stats].
Y2 = exprnd(mu_exp, n, 1);
[H2,P2,CI2,stats2] = ttest2(X_exp, Y2, 'Alpha',alpha);

% vartest(X,sigma2_0,'Alpha',a): Tests H0: variance == sigma2_0 using chi-square.
% Returns [h,p,ci,stats] with ci bounds for variance.
sigma2_0 = 1;
[H3,P3,CI3,stats3] = vartest(X_exp, sigma2_0, 'Alpha',alpha);

%% 7) BOOTSTRAP (MEAN AND CI)
B = 2000;
% bootstrp(B,@mean,X): Returns B-by-1 vector of bootstrap means.
boot_mu = bootstrp(B, @mean, X_exp);
% std(boot_mu): Empirical bootstrap SE for mean. Returns scalar.
se_mu   = std(boot_mu);
% bootci(B,{@mean,X},'alpha',a): Percentile (by default) CI for mean. Returns [low;high].
CI_mu   = bootci(B, {@mean, X_exp}, 'alpha', alpha);

%% 8) PERMUTATION (RANDOMIZATION) TEST FOR DIFFERENCE IN MEANS (BETTER THAN BOOTSTRAP FOR H0)
alpha = 0.05; B = 10000; replacement = false;  % choose true for pooled bootstrap
n = numel(X); m = numel(Y);                    
XY = [X; Y];

boot = NaN(B,1);                               % preallocate
for b = 1:B
    s = randsample(XY, n+m, replacement);      % resample pooled data
    boot(b) = mean(s(1:n)) - mean(s(n+1:end)); % difference in means
end

Tobs = mean(X) - mean(Y);                      % observed statistic
q = quantile(boot, [alpha/2, 1 - alpha/2]);    % percentile cutoffs
H = (Tobs < q(1)) || (Tobs > q(2));            % reject if outside central region

%% 9) CORRELATION, HYPOTHESIS TEST, AND CI FOR r
% corr(x,y,'Type','Pearson') returns [R,P] when two outputs requested.
% R: correlation coefficients matrix; P: p-values matrix.
t = linspace(0,10,n)';
xC = sin(t) + 0.1*randn(n,1);
yC = 0.8*sin(t) + 0.1*randn(n,1);
[Rmat, Pmat] = corr(xC, yC, 'Type','Pearson');  % Returns 1x1 here; general case is kxk
r = Rmat(1); p_r = Pmat(1);

% Fisher z CI for r:
z = 0.5*log((1+r)/(1-r));
z_halfwidth = norminv(1 - alpha/2) * sqrt(1/(n-3));
r_CI = [tanh(z - z_halfwidth), tanh(z + z_halfwidth)];

%% 10) SIMPLE LINEAR REGRESSION (fitlm IMPROVEMENT + CLASSIC FORMULA)
% Improvement: prefer fitlm for rich outputs, diagnostics, and prediction intervals.
tbl = table(xC, yC, 'VariableNames',{'x','y'});
% fitlm(tbl,'y ~ x'): Fits OLS y = beta0 + beta1*x.
% Returns LinearModel with properties (Coefficients, Rsquared, etc.) and methods (predict, plot).
mdl = fitlm(tbl, 'y ~ x');

% Coefficient table: mdl.Coefficients has Estimate, SE, tStat, pValue and CIs via coefCI(mdl).
coef_CI = coefCI(mdl, alpha);

% Predictions with intervals:
xstep = linspace(min(xC), max(xC), 200)';
% predict(mdl,X,'Prediction','observation'|'curve'): Returns yhat and interval bounds if requested.
[yhat_curve, CI_mean] = predict(mdl, xstep, 'Prediction','curve');
[yhat_pred,  PI_obs ] = predict(mdl, xstep, 'Prediction','observation');

figure('Name','Simple Linear Regression with Intervals'); clf
scatter(xC,yC,12,'filled'); hold on
plot(xstep, yhat_curve, 'r-', 'LineWidth',2)
plot(xstep, CI_mean(:,1), 'g--', xstep, CI_mean(:,2), 'g--')
plot(xstep, PI_obs(:,1), 'm-.', xstep, PI_obs(:,2), 'm-.')
title('fitlm: Fit, Mean-Response CI, and Prediction Intervals')
legend('Data','Fit','Mean CI','Mean CI','Pred PI','Pred PI','Location','best')
grid on

% Classic matrix form (for reference):
X1 = [ones(n,1) xC];            % Design matrix with intercept
b = X1 \ yC;                    % \ solves least-squares: returns [b0;b1]
yfit = X1*b;                    % Fitted values: returns n-by-1 vector
e = yC - yfit;                  % Residuals: returns n-by-1 vector
se = std(e);                    % Residual std estimate
sxx = sum((xC-mean(xC)).^2);    % Sum of squares in x
tcrit = tinv(1 - alpha/2, n-2); % t critical
b0_se = se * sqrt(1/n + mean(xC)^2/sxx);
b1_se = se / sqrt(sxx);
b0_CI = [b(1)-tcrit*b0_se, b(1)+tcrit*b0_se];  % 95% CI for intercept
b1_CI = [b(2)-tcrit*b1_se, b(2)+tcrit*b1_se];  % 95% CI for slope

%% 11) BOOTSTRAP CI FOR REGRESSION COEFFICIENTS
Mboot = 2000;
b0_arr = NaN(Mboot,1); b1_arr = NaN(Mboot,1);
for i = 1:Mboot
    idx = randi(n, n, 1);           % randi: bootstrap indices with replacement
    Xi = [ones(n,1) xC(idx)];
    yi = yC(idx);
    bb = Xi \ yi;                   % Returns [b0;b1] for bootstrap sample
    b0_arr(i) = bb(1);
    b1_arr(i) = bb(2);
end
b0_arr = sort(b0_arr); b1_arr = sort(b1_arr);
iL = floor(Mboot*alpha/2)+1; iU = floor(Mboot*(1-alpha/2));
b0_CI_boot = [b0_arr(iL) b0_arr(iU)];
b1_CI_boot = [b1_arr(iL) b1_arr(iU)];

%% 12) POLYNOMIAL REGRESSION
s = 3;  % degree
% polyfit(x,y,n): Least-squares polynomial coefficients (highest power first).
% Returns a row vector of length n+1. polyval(b,x): Evaluates polynomial; returns yhat vector.
bp = polyfit(xC, yC, s);
yfitp = polyval(bp, xC);

% R^2 and adj-R^2
e_poly = yC - yfitp;
SSres  = sum(e_poly.^2);
SStot  = (n-1)*var(yC);
Rsq    = 1 - SSres/SStot;
adjRsq = 1 - (SSres/SStot) * (n-1)/(n - (s+1) - 1);

figure('Name','Polynomial Regression'); clf
scatter(xC, yC, 12, 'filled'); hold on
plot(xC, yfitp, 'Color',[0.1 0.4 0.8], 'LineWidth',2)
title(sprintf('Polynomial Degree %d (R^2=%.3f, adjR^2=%.3f)', s, Rsq, adjRsq))
legend('Data','Polynomial fit','Location','best'); grid on

%% 13) MULTIPLE REGRESSION (fitlm/stepwiselm IMPROVEMENT + regress)
p = 6;
Xmulti = randn(n,p);
betaTrue = [1; 0; -2; 0; 3; 0];               % True coefficients (no intercept)
ymulti = Xmulti*betaTrue + 0.4*randn(n,1);

% Improvement: fitlm with formula interface and diagnostics
Tbl = array2table([ymulti Xmulti], 'VariableNames', [{'y'}, compose('x%d',1:p)]);
mdlFull = fitlm(Tbl, 'y ~ x1 + x2 + x3 + x4 + x5 + x6');  % Returns LinearModel
coef_CI_full = coefCI(mdlFull, alpha);

% Stepwise improvement: stepwiselm for automatic model selection
mdlStep = stepwiselm(Tbl, 'y ~ 1', 'Upper','linear', 'Criterion','aic', 'Verbose',0); % Returns selected LinearModel

% Classic: regress
XX = [ones(n,1) Xmulti];
% regress(y,XX): Multiple regression with intercept. Returns [b,bint,r,rint,stats].
[b_m,bCI_m,r_m,rint_m,stats_m] = regress(ymulti, XX);

figure('Name','Multiple Regression: FitLM vs Stepwise'); clf
subplot(1,2,1); plot(mdlFull); title('Full Linear Model (fitlm)'); grid on
subplot(1,2,2); plot(mdlStep); title('Stepwise Linear Model (stepwiselm)'); grid on

%% 14) PCA (DIMENSION REDUCTION) AND ICA
% pca(X): Principal components with centered/scaled option via zscore.
% Returns [coeff,score,latent,tsquared,explained,mu].
[coeff,score,latent,tsq,explained,mu] = pca(zscore(Xmulti));
cumvar = cumsum(explained);
idx95 = find(cumvar >= 95, 1, 'first');   % Smallest number of PCs explaining >=95%

figure('Name','PCA: Scree and Scores'); clf
subplot(1,2,1)
plot(explained,'-o','LineWidth',1.5); hold on
yline(95,'r--','95%'); title('Scree: Percent Variance Explained')
xlabel('PC index'); ylabel('% Explained'); grid on
subplot(1,2,2)
scatter(score(:,1),score(:,2),10,'filled'); axis equal
title('PC Scores: PC1 vs PC2'); xlabel('PC1'); ylabel('PC2'); grid on

% ICA via RICA (regularized ICA). Returns a rica model with TransformWeights/transform methods.
% Note: Requires Statistics and Machine Learning Toolbox; wrap in try-catch for portability.
try
    Xw = (Xmulti - mean(Xmulti,1))./std(Xmulti,[],1);  % simple whitening/standardization
    mdlICA = rica(Xw, p, 'Lambda', 0.5);               % Returns a rica model object
catch
    mdlICA = [];
end

%% 15) REGULARIZATION AND LATENT-VARIABLE REGRESSION
% Ridge regression:
k = linspace(0, 0.5, 61);   % Ridge penalty vector (nonnegative)
% ridge(y,X,k,0): Returns coefficients for each k in columns; 0 indicates no centering/scaling.
bRR = ridge(ymulti, Xmulti, k, 0);  % Returns (p+1)-by-numel(k) including intercept row

figure('Name','Ridge Trace'); clf
plot(k, bRR(2:end,:)', 'LineWidth',1.5)          % Skip intercept for trace
xlabel('Ridge Penalty k'); ylabel('Coefficient'); title('Ridge Trace'); grid on

% LASSO with 10-fold CV:
% lasso(X,y,'CV',10): Returns [B,FitInfo], where columns of B are solutions per lambda.
% FitInfo includes IndexMinMSE (best lambda), Intercept, MSE, SE, and Lambda vector.
[bLASSO, FitInfo] = lasso(Xmulti, ymulti, 'CV', 10, 'Standardize', true);
idxMin = FitInfo.IndexMinMSE;
bLassoFull = [FitInfo.Intercept(idxMin); bLASSO(:,idxMin)];  % Combine intercept + coefficients
yfit_lasso = [ones(n,1) Xmulti] * bLassoFull;

figure('Name','LASSO CV Curve'); clf
lassoPlot(bLASSO, FitInfo, 'PlotType','CV');
title('LASSO: Cross-Validation MSE vs \lambda')

% Partial Least Squares regression:
ncomp = min(10, p);
% plsregress(X,y,ncomp): Returns [XL,yl,XS,YS,beta,PCTVAR,MSE,STATS]; beta includes intercept.
[XL,yl,XS,YS,bPLS,PCTVAR,MSE,STATS] = plsregress(Xmulti, ymulti, ncomp);
yfit_pls = [ones(n,1) Xmulti] * bPLS;

figure('Name','PLS: Variance Explained'); clf
plot(1:ncomp, cumsum(100*PCTVAR(2,1:ncomp)),'-bo','LineWidth',1.5)
xlabel('# PLS Components'); ylabel('% Var Explained in y'); title('PLS Cumulative Variance'); grid on

% OLS via SVD (stable alternative):
% svd(Xc): Xc = centered predictors. Returns U,S,V such that Xc = U*S*V'.
Xm = Xmulti - mean(Xmulti,1);
[U,S,V] = svd(Xm,'econ');
% V*(S\(U'*(y-mean(y)))): Returns OLS slopes in original columns after centering.
bOLS_slopes = V * (S \ (U' * (ymulti - mean(ymulti))));
% Rebuild intercept for original scale: intercept = mean(y) - mean(X)*slopes
bOLS = [mean(ymulti) - mean(Xmulti,1)*bOLS_slopes; bOLS_slopes];
yfit_ols = [ones(n,1) Xmulti] * bOLS;

% Principal Components Regression (PCR):
% pca -> choose PCs -> regress y on scores -> map back to coefficient in original space.
[PCALoad,PCAScores,~,~,explPCR,muPCR] = pca(Xmulti, 'Economy',false); 
cumPCR = cumsum(explPCR);
idxPCR = find(cumPCR >= 95, 1, 'first');
bPCR_scores = regress(ymulti - mean(ymulti), PCAScores(:,1:idxPCR));  % Returns scores coefficients
bPCR_slopes = PCALoad(:,1:idxPCR) * bPCR_scores;                       % Map back to X-space
bPCR = [mean(ymulti) - mean(Xmulti,1)*bPCR_slopes; bPCR_slopes];       % Add intercept
yfit_pcr = [ones(n,1) Xmulti] * bPCR;

figure('Name','Method Comparison: In-Sample Residuals'); clf
res_pls = ymulti - yfit_pls;
res_las = ymulti - yfit_lasso;
res_ols = ymulti - yfit_ols;
res_pcr = ymulti - yfit_pcr;
hold on
stem(res_pls,'filled','DisplayName','PLS'); 
stem(res_las,'filled','DisplayName','LASSO');
stem(res_ols,'filled','DisplayName','OLS (SVD)');
stem(res_pcr,'filled','DisplayName','PCR');
title('Residuals by Method (In-Sample)')
xlabel('Observation'); ylabel('Residual'); legend('Location','best'); grid on

%% 16) APPENDIX: UTILITIES QUICK DEMOS
A = randn(100,1);
% mink(A,k)/maxk(A,k): Return k smallest/largest elements (and optionally indices).
small3 = mink(A,3); large3 = maxk(A,3);

% meshgrid & surf: 3D visualization
v = linspace(-2,2,101);
[x1,x2] = meshgrid(v,v); z = x1.*x2;
figure('Name','Surface Example'); clf
surf(x1,x2,z); title('surf(x_1,x_2,z=x_1x_2)'); xlabel('x_1'); ylabel('x_2'); zlabel('z')
shading interp; colorbar
