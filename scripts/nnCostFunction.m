function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ---------------------------- Cost Function ------------------------------

% Recoding the y labels to Y matrix
%Y = recodeY(y,m,num_labels);
eye_mat = eye(num_labels);
Y = eye_mat(y, :);

% Adding bias to input layer
a1 = [ones(m,1) X];

% Computing hidden layer
z2 = a1 * Theta1';
a2 = sigmoid(z2);

% Adding bias to hidden layer
a2 = [ones(m,1) a2];

% Computing the output layer
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% Storing our hypothesis
H = a3;

% Sum up the cost or error in our hypothesis
% Double sum for each K number of output units
J = sum(sum(-Y .* log(H) - (1 - Y) .* log(1 - H))) / m;

% Adding regularization parameters to cost function
Theta1_reg = Theta1(:, 2:end) .^ 2;
Theta2_reg = Theta2(:, 2:end) .^ 2;
regParams = (lambda * (sum(sum(Theta1_reg)) + sum(sum(Theta2_reg))) )...
			/ (2 * m);
J = J + regParams;

% ----------------------- Gradient Cost Function --------------------------

% Initialising the Accumulators
D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));

% Each column of X is a full example with all features
% Hence we convert each example with all features
% into a rows for better visualisation consistent with NN diagrams
X = X';
Y = Y';

% For each example from training set
for i = 1:m,

	%% --- Feed Forward Pass begins ---

	% Feed the example into the first input layer
	a1 = X(:, i);

	% Add bias to first layer
	a1 = [1; a1];

	% Compute hidden layer
	z2 = Theta1 * a1;
	a2 = sigmoid(z2);

	% Add bias to the hidden layer
	z2 = [1; z2];
	a2 = [1; a2];

	% Compute output layer
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);

	%% --- Back Propagation begins ---

	% Error of the output layer
	d3 = a3 - Y(:, i);

	% Error of the hidden layer
	d2 = Theta2' * d3 .* sigmoidGradient(z2);
	d2 = d2(2:end);

	% Accumulating the Gradient for hidden layer
	D2 = D2 + d3 * a2'; 
	% Accumulating the Gradient for input layer
	D1 = D1 + d2 * a1'; 

end

% Regularization of the Gradient
D1 = (1/m) .* D1;
D1(:, 2:end) = D1(:, 2:end) + (Theta1(:, 2:end) .* (lambda/m));

D2 = (1/m) .* D2;
D2(:, 2:end) =  D2(:, 2:end) + (Theta2(:, 2:end) .* (lambda/m)); 


% Skipping the bias column
% D1 = D1(:, 2:end);
% D2 = D2(:, 2:end);

Theta1_grad = D1;
Theta2_grad = D2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
