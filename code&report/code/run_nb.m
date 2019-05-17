% Learn a Naive Bayes classifier on the digit dataset, evaluate its
% performance on training and test sets, then visualize the mean and variance
% for each class.

load mnist_train;
load mnist_test;

% Add your code here (it should be less than 10 lines)
[log_prior, class_mean, class_var] = train_nb(train_inputs, train_targets);
[prediction_train, accuracy_train] = test_nb(train_inputs, train_targets, log_prior, class_mean, class_var);
[prediction_test, accuracy_test] = test_nb(test_inputs, test_targets, log_prior, class_mean, class_var);
fprintf(1, 'training accuracy %.4f\ntest accuracy %.4f\n', accuracy_train, accuracy_test)
plot_digits(class_mean)
plot_digits(class_var)
