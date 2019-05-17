function [f, df, y] = logistic_no_penalize(weights, data, targets)
% Calculate log likelihood and derivatives with respect to weights.
%
% Note: N is the number of examples and 
%       M is the number of features per example.
%
% Inputs:
% 	weights:    (M+1) x 1 vector of weights, where the last element
%               corresponds to bias (intercepts).
% 	data:       N x M data matrix where each row corresponds 
%               to one data point.
%	targets:    N x 1 vector of binary targets. Values should be either 0 or 1.
%
% Outputs:
%	f:             The scalar error value?i.e. negative log likelihood).
%	df:            (M+1) x 1 vector of derivatives of error w.r.t. weights.
%   y:             N x 1 vector of probabilities. This is the output of the classifier.
%

%TODO: finish this function

[N, M] = size(data);

data = [data, ones(N,1)]; % Add 1 to the end of each row of data, now data is N x (M+1)
wx1 = data(targets == 1, :) * weights; % wx+b correspond to yi == 1
wx0 = data(targets == 0, :) * weights; % wx+b correspond to yi == 0

f = sum(log(1 + exp(-wx1))) + sum(log(1 + exp(wx0)));
df = -sum(data(targets == 1, :) ./ repmat(1 + exp(wx1), 1, M+1)) +...
    sum(data(targets == 0, :) ./ repmat(1 + exp(-wx0), 1, M+1));
df = df';
y = 1 ./ (1 + exp(-data * weights));
end
