function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
for iter = 1:num_iters
    disc1 = 0;
    disc2 = 0;
    move1 = 0;
    move2 = 0;
    for i = 1:m
        disc1 = disc1 + (X(i,:) * theta - y(i)) * X(i,1);
        disc2 = disc2 + (X(i,:) * theta - y(i)) * X(i,2);
    end
    move1 = disc1 * alpha / m;
    move2 = disc2 * alpha / m;
    theta = theta - [move1;move2];



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
