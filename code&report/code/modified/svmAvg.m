function [model] = svmAvg(X,y,lambda,maxIter)

% Add bias variable
[n,d] = size(X);
X = [ones(n,1) X];
Xt = X';

% Initial values of regression parameters
w = zeros(d+1,1);

% Initialize the gradient table for all the input
index = (1 - y .* (X * w)) > 0;
gt = -y .* index * ones(1,d+1) .* X; % gradient table
gt = gt'; % (d+1) by n
gt_sum = sum(gt, 2);

% Apply stochastic gradient method
for t = 1:maxIter
    if mod(t-1,n) == 0
        % Plot our progress
        % (turn this off for speed)
        
        objValues(1+(t-1)/n) = (1/n)*sum(max(0,1-y.*(X*w))) + (lambda/2)*(w'*w);
        semilogy([0:t/n],objValues);
        %pause(.1);
    end
    
    % Pick a random training example
    i = ceil(rand*n);
    
    % Compute sub-gradient
    sg = hingeLossSubGrad(w,Xt,y,i);
    
    % Updata the gradient table and its mean
    gt_sum = gt_sum - gt(:,i) + sg;
    gt(:,i) = sg;
    
    % Set step size
    alpha = 1/(lambda*t);
    
    % Take stochastic subgradient step
    if t < maxIter/2
        w = w - alpha * (sg + lambda*w);
    else
        w = w - alpha * (gt_sum / n + lambda*w);
    end
    
end

model.w = w;
model.predict = @predict;

end

function [yhat] = predict(model,Xhat)
[t,d] = size(Xhat);
Xhat = [ones(t,1) Xhat];
w = model.w;
yhat = sign(Xhat*w);
end

function sg = hingeLossSubGrad(w,Xt,y,i)

[d,n] = size(Xt);

% Function value
wtx = w'*Xt(:,i);
loss = max(0,1-y(i)*wtx);

% Subgradient
if loss > 0
    sg = -y(i)*Xt(:,i);
else
    sg = sparse(d,1);
end
end

