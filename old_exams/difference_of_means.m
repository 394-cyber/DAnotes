function Exe1Func2(mx, sigmax, my, sigmay, lambda)
    rng(1);
    n = 20; % sample size
    n_repeat = 100;
    alpha = 0.05;

    Hparam_before = false(1,n_repeat);
    Hparam_after  = false(1,n_repeat);
    Hboot_before  = false(1,n_repeat);
    Hboot_after   = false(1,n_repeat);

    for i=1:n_repeat
        X = lognrnd(mx, sigmax, n, 1);
        Y = lognrnd(my, sigmay, n, 1);

        % Transform (Box-Cox log if lambda=0)
        Zx = Exe1Func1(X, lambda);
        Zy = Exe1Func1(Y, lambda);

        % --- Parametric t-test ---
        Hparam_before(i) = ttest2(X, Y, 'Vartype','unequal','Alpha',alpha);
        Hparam_after(i)  = ttest2(Zx, Zy, 'Vartype','unequal','Alpha',alpha);

        % --- Bootstrap mean difference ---
        B = 1000;
        bootX  = bootstrp(B, @mean, X);
        bootY  = bootstrp(B, @mean, Y);
        diffXY = bootX - bootY;
        CI = prctile(diffXY,[100*alpha/2, 100*(1-alpha/2)]);
        Hboot_before(i) = (CI(1) > 0 || CI(2) < 0);

        bootZx = bootstrp(B, @mean, Zx);
        bootZy = bootstrp(B, @mean, Zy);
        diffZ  = bootZx - bootZy;
        CI = prctile(diffZ,[100*alpha/2, 100*(1-alpha/2)]);
        Hboot_after(i) = (CI(1) > 0 || CI(2) < 0);
    end

    fprintf("Before Box–Cox: rejected %d/100 (parametric), %d/100 (bootstrap)\n", ...
        sum(Hparam_before), sum(Hboot_before));
    fprintf("After  Box–Cox: rejected %d/100 (parametric), %d/100 (bootstrap)\n", ...
        sum(Hparam_after), sum(Hboot_after));
end
