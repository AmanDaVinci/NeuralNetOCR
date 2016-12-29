%% Initialization
clear ; close all; clc

%% Setup the parameters for this Neural Net
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   

%% ============== Loading and Visualizing Data ==================

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('../data/data.mat');

% Number of examples in data
m = size(X, 1); %% ---------------> X came from the data.mat?

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============= Initializing Pameters ================

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% ================ Checking Gradients ================

% Setting up lambda for checking
lambda = 1;

% Check gradients by running checkNNGradients
checkNNGradients(lambda);

%% ================== Training NN =====================

fprintf('\nTraining Neural Network... \n')

% Experimenting with various iterations
options = optimset('MaxIter', 200);

% Experimenting with learning rates
lambda = 0.1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ===================== Visualize Weights ========================
% Visualizing what the neural network is learning by displaying 
% the hidden units to see what features they are capturing in the data.

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================= Computing the Accuracy =================

% Making predictions on the entire traning set
pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

% Saving the trained weights
save('../data/learntWeights.mat', 'Theta1', 'Theta2');

%% ================= Predicting the Values =================

%  Randomly permute examples
rp = randperm(m);

for i = 1:m
    % Display 
    fprintf('\n>>Displaying Example Image\n');
    displayData(X(rp(i), :));

    pred = predict(Theta1, Theta2, X(rp(i),:));
    fprintf('\n>>Neural Network Prediction: %d (DIGIT %d)\n\n', pred, mod(pred, 10));
    
    % Pause
    % fprintf('Program paused. Press enter to continue.\n');
    pause;
end


%% ================= Classifying Raw Images =================

% Vectorizing the raw jpeg image
X = vectorImage('../data/digit2.jpg');

% Classifying using the trained neural network
pred = predict(Theta1, Theta2, X);
fprintf('\n>>Neural Network Prediction: %d (DIGIT %d)\n\n', pred, mod(pred, 10));
