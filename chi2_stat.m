function chi2 = chi2_stat(joint_p, marginal_p)
%CHI2_STAT Υπολογισμός στατιστικού Χ² για έλεγχο ανεξαρτησίας
%
%   chi2 = chi2_stat(joint_p, marginal_p)
%
% Είσοδοι:
%   joint_p    - 2x2 matrix με observed πιθανότητες
%   marginal_p - 1x2 vector με τις περιθώριες πιθανότητες του Yt
%
% Έξοδος:
%   chi2 - τιμή του στατιστικού Χ²

    % expected matrix: εξωτερικό γινόμενο περιθωρίων
    expected = marginal_p' * marginal_p;

    % Χ² υπολογισμός
    chi2 = sum(sum((joint_p - expected).^2 ./ expected));
end
