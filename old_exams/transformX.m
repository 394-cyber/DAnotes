function Z = Exe1Func1(X, lambda)
    if lambda == 0
        Z = log(X);
    else
        Z = (X .^ lambda) ./ lambda;
    end