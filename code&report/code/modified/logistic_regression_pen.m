%% Clear workspace.
clear;
close all;
clc;

%% Load data.
load mnist_train;
load mnist_train_small;
load mnist_valid;
load mnist_test;

%% Initialize hyperparameters.
D = size(train_inputs, 2);
% Learning rate
hyperparameters.learning_rate = 0.1;
% Weight regularization parameter
hyperparameters.weight_regularization = 10^(-4);
% Number of iterations
hyperparameters.num_iterations = 400;
% Logistics regression weights
weights = randn(1, D+1) / sqrt(10);

%% Verify that logistic function produces the right gradient, diff should be very close to 0
% this creates small random data with 20 examples and 10 dimensions and checks the gradient on that data.
nexamples = 20;
ndimensions = 10;

diff = checkgrad('logistic_pen', ...
    randn((ndimensions + 1), 1), ...   % weights
    0.001,...                          % perturbation
    randn(nexamples, ndimensions), ... % data
    rand(nexamples, 1) > randn(1), ... % targets
    hyperparameters)                   % other hyperparameters

%% Experiment with different values of ¦Ë
% Learning rate
hyperparameters.learning_rate = 0.1;
% Number of iterations
hyperparameters.num_iterations = 400;
% Change lambda
lglambda = -3:0;
nl = length(lglambda);

% Initialization
ce_train = zeros(nl, 1); fc_train = zeros(nl, 1);
ce_valid = zeros(nl, 1); fc_valid = zeros(nl, 1);

% When you want to run the code on the minist_train set, comment the two lines below.
% When you want to run the code on the minist_train_small set, uncomment the two lines below.
%train_inputs = train_inputs_small; train_targets = train_targets_small;
%hyperparameters.num_iterations = 20;
N = size(train_inputs,1);

%% Begin learning with gradient descent.
for i = 1:nl
    hyperparameters.weight_regularization = 10^lglambda(i);
    
    % Re-run 50 times for each value of lambda with the weights randomly initialized each time.
    for j = 1:50
        weights = randn(D+1, 1) / 10;
        
        for t = 1:hyperparameters.num_iterations
            % Update parameters.
            [~, df, ~] = logistic(weights, train_inputs, train_targets);
            weights = weights - hyperparameters.learning_rate .* df / N;
        end
        
        % Compute results for each loop
        [f, df, predictions] = logistic(weights, train_inputs, train_targets);
        [tmp1, tmp2] = evaluate(train_targets, predictions);
        ce_train(i) = ce_train(i) + tmp1; fc_train(i) = fc_train(i) + tmp2;
        predictions_valid = logistic_predict(weights, valid_inputs);
        [tmp1, tmp2] = evaluate(valid_targets, predictions_valid);
        ce_valid(i) = ce_valid(i) + tmp1; fc_valid(i) = fc_valid(i) + tmp2;
        
    end
end

% Compute average
ce_train = ce_train / 50; ce_valid = ce_valid / 50;
fc_train = fc_train / 50; fc_valid = fc_valid / 50;

%% Plot
figure(1)
subplot(1, 2, 1)
semilogx(10.^lglambda, ce_train, 'b')
hold on
semilogx(10.^lglambda, ce_valid, 'g')
axis([-inf inf 0 1])
xlabel('\lambda'); ylabel('cross-entropy')
legend('train', 'validation')
subplot(1, 2, 2)
semilogx(10.^lglambda, fc_train, 'b')
hold on
semilogx(10.^lglambda, fc_valid, 'g')
axis([-inf inf 0 1])
xlabel('\lambda'); ylabel('fraction of correct predictions')
legend('train', 'validation')


%% The best value of ¦Ë
hyperparameters.weight_regularization = 0.01;
% Learning rate
hyperparameters.learning_rate = 0.1;
% Number of iterations
hyperparameters.num_iterations = 400;

% Initalization
ce_train = zeros(hyperparameters.num_iterations, 1);
ce_valid = zeros(hyperparameters.num_iterations, 1);
fc_train = zeros(hyperparameters.num_iterations, 1);
fc_valid = zeros(hyperparameters.num_iterations, 1);

% When you want to run the code on the minist_train set, comment the three lines below.
% When you want to run the code on the minist_train_small set, uncomment three two lines below.
%train_inputs = train_inputs_small; train_targets = train_targets_small;
%hyperparameters.num_iterations = 20;
%hyperparameters.weight_regularization = 0.1;
N = size(train_inputs,1);

%% Begin learning with gradient descent.
% Re-run 50 times for each value of lambda with the weights randomly initialized each time.
for j = 1:50
    weights = randn(D+1, 1) / 10;
    for t = 1:hyperparameters.num_iterations
        % Find the negative log likelihood and derivative w.r.t. weights.
        [f, df, predictions] = logistic(weights, train_inputs, train_targets);
        [tmp1, tmp2] = evaluate(train_targets, predictions);
        ce_train(t) = ce_train(t) + tmp1; fc_train(t) = fc_train(t) + tmp2;
        predictions_valid = logistic_predict(weights, valid_inputs);
        [tmp1, tmp2] = evaluate(valid_targets, predictions_valid);
        ce_valid(t) = ce_valid(t) + tmp1; fc_valid(t) = fc_valid(t) + tmp2;
        
        if isnan(f) || isinf(f)
            error('nan/inf error');
        end
        
        % Update parameters.
        weights = weights - hyperparameters.learning_rate .* df / N;
        
    end
    
end

% Compute average
ce_train = ce_train / 50; ce_valid = ce_valid / 50;
fc_train = fc_train / 50; fc_valid = fc_valid / 50;

%% Plot
figure(2)
subplot(1, 2, 1)
plot(ce_train, 'b')
hold on
plot(ce_valid, 'g')
axis([1 hyperparameters.num_iterations 0 1])
legend('train', 'validation')
subplot(1, 2, 2)
plot(1 - fc_train, 'b')
hold on
plot(1 - fc_valid, 'g')
axis([1 hyperparameters.num_iterations 0 1])
legend('train', 'validation')


%% test error
predictions_test = logistic_predict(weights, test_inputs);
[ce_test, fc_test] = evaluate(test_targets, predictions_test);
fprintf(1, 'test error = %.4f, cross-entropy = %.4f\n', 1 - fc_test, ce_test);