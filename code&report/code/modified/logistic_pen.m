function [f, df, y] = logistic_pen(weights, data, targets, hyperparameters)
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
%   targets:    N x 1 vector of targets class probabilities.
%   hyperparameters: The hyperparameter structure
%
% Outputs:
%	f:             The scalar error value.
%	df:            (M+1) x 1 vector of derivatives of error w.r.t. weights.
%   y:             N x 1 vector of probabilities. This is the output of the classifier.
%

%TODO: finish this function

[N, M] = size(data);
lambda = hyperparameters.weight_regularization;

data = [data, ones(N,1)]; % Add 1 to the end of each row of data, now data is N x (M+1)
wx1 = data(targets == 1, :) * weights; % wx+b correspond to yi == 1
wx0 = data(targets == 0, :) * weights; % wx+b correspond to yi == 0
w = weights(1:M); % w doesn't include the bias b

f = sum(log(1 + exp(-wx1))) + sum(log(1 + exp(wx0))) + lambda / 2 * (w' * w);
df = -sum(data(targets == 1, :) ./ repmat(1 + exp(wx1), 1, M+1)) +...
    sum(data(targets == 0, :) ./ repmat(1 + exp(-wx0), 1, M+1)) + ...
    lambda * [w', 0];
df = df';
y = 1 ./ (1 + exp(-data * weights));

end
