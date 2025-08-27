function results = randomization(num_random_samples, preTMS, postTMS)
    %randomization Test implementation
    %create 1000 random samples of preTMS data and calculate correlation

    
    results = struct();
    random_r = zeros(num_random_samples, 1);
    random_p = zeros(num_random_samples, 1);

    for j = 1:num_random_samples
        % Shuffle preTMS values
        shuffled_preTMS = preTMS(randperm(length(preTMS)));
        % Calculate correlation for randomized data
        [random_r(j), random_p(j)] = corr(shuffled_preTMS, postTMS, 'Type', 'Pearson');
    end

    results.r = random_r;
    results.p = random_p;

end