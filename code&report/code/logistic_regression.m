%% Clear workspace.
clear;
close all;
clc;

%% Load data.
load mnist_train;
load mnist_train_small;
load mnist_valid;
load mnist_test;

%% Verify that logistic function produces the right gradient, diff should be very close to 0
% this creates small random data with 20 examples and 10 dimensions and checks the gradient on that data.
nexamples = 20;
ndimensions = 10;

diff = checkgrad('logistic', ...
    randn((ndimensions + 1), 1), ...   % weights
    0.001,...                          % perturbation
    randn(nexamples, ndimensions), ... % data
    rand(nexamples, 1) > randn(1))     % targets(they should be 0 or 1, there is a mistake in the template)

%% Experiment with Learning rate
% Weight regularization parameter
hyperparameters.weight_regularization = 0;
% Fixed times of iteration
hyperparameters.num_iterations = 100;
% Initialization
lglr = -4:0.5:0;
cross_entropy_train = zeros(length(lglr), hyperparameters.num_iterations);
cross_entropy_valid = zeros(length(lglr), hyperparameters.num_iterations);
frac_correct_train = zeros(length(lglr), hyperparameters.num_iterations);
frac_correct_valid = zeros(length(lglr), hyperparameters.num_iterations);

% When you want to run the code on the minist_train_small set, uncomment the code below.
% When you want to run the code on the minist_train set, comment the code below.
%train_inputs = train_inputs_small; train_targets = train_targets_small;
N = size(train_inputs,1);

%% Begin learning with gradient descent.
for i = 1:length(lglr)
    hyperparameters.learning_rate = 10^lglr(i); % range from 10^-5 to 1
    % Fixed initial weights.
    weights = zeros(size(train_inputs, 2)+1, 1);
    for t = 1:hyperparameters.num_iterations
        % Find the negative log likelihood and derivative w.r.t. weights.
        [f, df, predictions] = logistic(weights, train_inputs, train_targets);
        [cross_entropy_train(i,t), frac_correct_train(i,t)] = evaluate(train_targets, predictions);
        predictions_valid = logistic_predict(weights, valid_inputs);
        [cross_entropy_valid(i,t), frac_correct_valid(i,t)] = evaluate(valid_targets, predictions_valid);
        
        if isnan(f) || isinf(f)
            error('nan/inf error');
        end
        
        % Update parameters.
        weights = weights - hyperparameters.learning_rate .* df / N;

        % Print some stats.
        if mod(t, 10) == 0
        fprintf(1, 'ITERATION:%4i   NLOGL:%4.2f TRAIN CE %.6f TRAIN FRAC:%2.2f VALIC_CE %.6f VALID FRAC:%2.2f\n',...
            t, f/N, cross_entropy_train(i,t), frac_correct_train(i,t) * 100, ...
            cross_entropy_valid(i,t), frac_correct_valid(i,t) * 100);
        end
    end
    % plot
    figure(1);
    subplot(3, 3, i); hold on
    plot(cross_entropy_train(i, :), 'b')
    plot(cross_entropy_valid(i, :), 'g')
    axis([0 100 0 1])
    title(['lr = ',num2str(10),'\^','(',num2str(lglr(i)),')'])
    figure(2);
    subplot(3, 3, i); hold on
    plot(frac_correct_train(i, :), 'b')
    plot(frac_correct_valid(i, :), 'g')
    axis([0 100 0 1])
    title(['lr = ',num2str(10),'\^','(',num2str(lglr(i)),')'])
end

% Add legends
figure(1)
legend('train', 'validation')
figure(2)
legend('train', 'validation')


%% Experiment with initial weights
% Choose the best learning rate
hyperparameters.learning_rate = 0.1;
% Choose a large enough number to obsserve its convergency
hyperparameters.num_iterations = 100;
% Change initial weights
weights_init = [rand(size(train_inputs, 2)+1, 1), ...
    randn(size(train_inputs, 2)+1, 1), ...
    zeros(size(train_inputs, 2)+1, 1), ...
    ones(size(train_inputs, 2)+1, 1) / 2, ...
    ones(size(train_inputs, 2)+1, 1)];
nw = size(weights_init, 2);
title1 = 'U(0,1)'; title2 = 'N(0,1)'; title3 = '0'; title4 = '0.5'; title5 = '1';
% Initalization
cross_entropy_train = zeros(nw, hyperparameters.num_iterations);
cross_entropy_valid = zeros(nw, hyperparameters.num_iterations);
frac_correct_train = zeros(nw, hyperparameters.num_iterations);
frac_correct_valid = zeros(nw, hyperparameters.num_iterations);

%% Begin learning with gradient descent.
for i = 1:nw
    weights = weights_init(:, i);
    for t = 1:hyperparameters.num_iterations
        % Find the negative log likelihood and derivative w.r.t. weights.
        [f, df, predictions] = logistic(weights, train_inputs, train_targets);
        [cross_entropy_train(i, t), frac_correct_train(i, t)] = evaluate(train_targets, predictions);
        predictions_valid = logistic_predict(weights, valid_inputs);
        [cross_entropy_valid(i, t), frac_correct_valid(i, t)] = evaluate(valid_targets, predictions_valid);
        
        if isnan(f) || isinf(f)
            error('nan/inf error');
        end
        
        % Update parameters.
        weights = weights - hyperparameters.learning_rate .* df / N;
        
        % Print some stats.
        if mod(t, 10) == 0
            fprintf(1, 'ITERATION:%4i   NLOGL:%4.2f TRAIN CE %.6f TRAIN FRAC:%2.2f VALIC_CE %.6f VALID FRAC:%2.2f\n',...
                t, f/N, cross_entropy_train(i, t), frac_correct_train(i, t) * 100, ...
                cross_entropy_valid(i, t), frac_correct_valid(i, t) * 100);
        end
    end
    % plot
    figure(3)
    subplot(2, 3, i); hold on
    plot(cross_entropy_train(i, :), 'b')
    plot(cross_entropy_valid(i, :), 'g')
    title(eval(['title', num2str(i)]))
    axis([0 100 0 5])
    figure(4)
    subplot(2, 3, i); hold on
    plot(frac_correct_train(i, :), 'b')
    plot(frac_correct_valid(i, :), 'g')
    title(eval(['title', num2str(i)]))
    axis([0 100 0 1])
end

%Add legends
figure(3)
legend('train', 'validation');
figure(4)
legend('train', 'validation');

%% Experiment with times of iteration
% Choose the best learning rate
hyperparameters.learning_rate = 0.1;
% Choose the best initial weights.
weights = zeros(size(train_inputs, 2)+1, 1);
% Choose a large enough number to obsserve its convergency
hyperparameters.num_iterations = 1000;
% Initalization
cross_entropy_train = zeros(1, hyperparameters.num_iterations);
cross_entropy_valid = zeros(1, hyperparameters.num_iterations);
frac_correct_train = zeros(1, hyperparameters.num_iterations);
frac_correct_valid = zeros(1, hyperparameters.num_iterations);

%% Begin learning with gradient descent.
for t = 1:hyperparameters.num_iterations
    % Find the negative log likelihood and derivative w.r.t. weights.
    [f, df, predictions] = logistic(weights, train_inputs, train_targets);
    [cross_entropy_train(t), frac_correct_train(t)] = evaluate(train_targets, predictions);
    predictions_valid = logistic_predict(weights, valid_inputs);
    [cross_entropy_valid(t), frac_correct_valid(t)] = evaluate(valid_targets, predictions_valid);
    
    if isnan(f) || isinf(f)
        error('nan/inf error');
    end
    
    % Update parameters.
    weights = weights - hyperparameters.learning_rate .* df / N;
    
    % Print some stats.
    if mod(t, 100) == 0
        fprintf(1, 'ITERATION:%4i   NLOGL:%4.2f TRAIN CE %.6f TRAIN FRAC:%2.2f VALIC_CE %.6f VALID FRAC:%2.2f\n',...
            t, f/N, cross_entropy_train(t), frac_correct_train(t) * 100, ...
            cross_entropy_valid(t), frac_correct_valid(t) * 100);
    end
end
% plot
figure(5);
subplot(1, 2, 1); hold on
plot(cross_entropy_train, 'b')
plot(cross_entropy_valid, 'g')
axis([0 hyperparameters.num_iterations 0 1])
legend('train', 'validation');
subplot(1, 2, 2); hold on
plot(frac_correct_train, 'b')
plot(frac_correct_valid, 'g')
axis([0 hyperparameters.num_iterations 0 1])
legend('train', 'validation');



%% Hyperparameter settings which work the best
% Choose the best settings
hyperparameters.learning_rate = 0.1;
weights = zeros(size(train_inputs, 2)+1, 1);
hyperparameters.num_iterations = 400;

% Initalization
cross_entropy_train = zeros(1, hyperparameters.num_iterations);
cross_entropy_valid = zeros(1, hyperparameters.num_iterations);
frac_correct_train = zeros(1, hyperparameters.num_iterations);
frac_correct_valid = zeros(1, hyperparameters.num_iterations);

% When you want to run the code on the minist_train set, comment the two lines below.
% When you want to run the code on the minist_train_small set, uncomment the two lines below.
%train_inputs = train_inputs_small; train_targets = train_targets_small;
%hyperparameters.num_iterations = 20;
N = size(train_inputs,1);

%% Begin learning with gradient descent.
for t = 1:hyperparameters.num_iterations
    % Find the negative log likelihood and derivative w.r.t. weights.
    [f, df, predictions] = logistic(weights, train_inputs, train_targets);
    [cross_entropy_train(t), frac_correct_train(t)] = evaluate(train_targets, predictions);
    predictions_valid = logistic_predict(weights, valid_inputs);
    [cross_entropy_valid(t), frac_correct_valid(t)] = evaluate(valid_targets, predictions_valid);
    
    if isnan(f) || isinf(f)
        error('nan/inf error');
    end
    
    % Update parameters.
    weights = weights - hyperparameters.learning_rate .* df / N;
    
end

%% Print results.
predictions_train = logistic_predict(weights, valid_inputs);
[fce_train, ffc_train] = evaluate(train_targets, predictions);
predictions_valid = logistic_predict(weights, valid_inputs);
[fce_valid, ffc_valid] = evaluate(valid_targets, predictions_valid);
predictions_test = logistic_predict(weights, test_inputs);
[fce_test, ffc_test] = evaluate(test_targets, predictions_test);
fprintf(1, ['Train: cross-entropy %.6f, error:%2.2f\n', ...
    'Validation: cross-entropy %.6f, error:%2.2f\n', ...
    'Test: cross-entropy %.6f, error:%2.2f\n'],...
    fce_train, 1 - ffc_train, fce_valid, 1 - ffc_valid, fce_test, 1 - ffc_test)

%% Plot
figure(6);
subplot(1, 2, 1); hold on
plot(cross_entropy_train, 'b')
plot(cross_entropy_valid, 'g')
axis([0 hyperparameters.num_iterations 0 1])
legend('train', 'validation');
subplot(1, 2, 2); hold on
plot(frac_correct_train, 'b')
plot(frac_correct_valid, 'g')
axis([0 hyperparameters.num_iterations 0 1])
legend('train', 'validation');
