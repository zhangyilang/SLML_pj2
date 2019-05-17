% A script that runs kNN for different values of k
clear;
close all;
clc;

% Import data
load mnist_train
load mnist_test
load mnist_valid

% kNN for different values of k
k = [1,3,5,7,9];
y_val = zeros(length(valid_targets), length(k));
for i = 1:length(k)
    y_val(:,i) = run_knn(k(i), train_inputs, train_targets, valid_inputs);
end

% Compute correct rate and plot
val_correct_rate = mean(y_val == valid_targets);

figure(1)
plot(k, val_correct_rate, 'b') % We choose k=5
axis([1 9 0 1]); xlabel('k'); ylabel('correct rate')
title('correct rate for different values of k')
hold on

% Perfomance of k={3,5,7} on test set
k_star = [3,5,7];
y_test = zeros(length(test_targets), length(k_star));
for i = 1:length(k_star)
    y_test(:,i) = run_knn(k_star(i), train_inputs, train_targets, test_inputs);
end

test_correct_rate = mean(y_test == test_targets);
plot(k_star, test_correct_rate, 'g')
legend('validation', 'test', 'Location', 'southeast')
