clear
clc
close all

load quantum.mat
[n,d] = size(X);

% Split into training and validation set
perm = randperm(n);
Xvalid = X(n/2+1:end,:);
yvalid = y(n/2+1:end);
X = X(1:n/2,:);
y = y(1:n/2);

n = n/2;
lambda = 1/n * 1000;
model = svmAvg(X,y,lambda,25*n);
yhat_train = model.predict(model, X);
yhat_train(yhat_train >= 0) = 1;
yhat_train(yhat_train < 0) = -1;
train_accu = mean(yhat_train == y);
fprintf(1, 'training accuracy %.4f\n', train_accu)
yhat_valid = model.predict(model, Xvalid);
yhat_valid(yhat_valid >= 0) = 1;
yhat_valid(yhat_valid < 0) = -1;
valid_accu = mean(yhat_valid == yvalid);
fprintf(1, 'validation accuracy %.4f\n', valid_accu)