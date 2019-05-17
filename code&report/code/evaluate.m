function [ce, frac_correct] = evaluate(targets, y)
%    Compute evaluation metrics.
%    Inputs:
%        targets : N x 1 vector of binary targets. Values should be either 0 or 1.
%        y       : N x 1 vector of probabilities.
%    Outputs:
%        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we
%                       want to compute CE(targets, y).
%        frac_correct : (scalar) Fraction of inputs classified correctly.

% TODO: Finish this function
    ce = sum(-log(y(targets == 1))) + sum(-log(1 - y(targets == 0)));
    ce = ce / length(y);
    frac_correct = mean(targets == (y >= 0.5));
end
