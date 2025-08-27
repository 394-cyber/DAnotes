function [joint_p, marginal_p] = joint_pmf(y)
%JOINT_PMF Υπολογισμός δειγματικών πιθανοτήτων από δυαδική χρονοσειρά
%
%   [joint_p, marginal_p] = joint_pmf(y)
%
% Είσοδος:
%   y - διάνυσμα n×1 με τιμές 0 ή 1
%
% Έξοδοι:
%   joint_p    - πίνακας 2x2 με τις σχετικές συχνότητες των ζευγών (yt, yt+1)
%   marginal_p - διάνυσμα 1x2 με τις περιθώριες πιθανότητες του Yt (ή Yt+1)

    assert(all(ismember(y,[0 1])), 'y must be binary 0/1');
    n = length(y);

    % Αριθμός ζευγών
    pairs = [y(1:end-1), y(2:end)];  
    
    % joint counts
    joint_counts = zeros(2,2);
    for i = 1:(n-1)
        joint_counts(pairs(i,1)+1, pairs(i,2)+1) = ...
            joint_counts(pairs(i,1)+1, pairs(i,2)+1) + 1;
    end

    % κανονικοποίηση -> pmf
    joint_p = joint_counts / (n-1);

    % περιθώριες
    marginal_p = sum(joint_p, 2)';  % γραμμές -> Yt
end
