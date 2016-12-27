function Y = recodeY(y, m, n)

%%% Function Documentation : recodeY
%%%	Written By Aman DaVinci
% Recoded Multi-Class Vector Y = recodeY(Target_Labels,
%										Number_of_Examples,
%										Number_of_Features)
% 	
% Recodes the Target Labels with labels greater than 3
% into Target Vector representation for Neural Networks
% Fully Vectorized and uses no for loops
%

% Multi-class representation of target vector
Y = zeros(m,n);

% Unroll and Row vecotrize Y for better indexing
Y = Y(:)';

% Computes the indexes of each column
indexes = ones(1,m) * n .* linspace(0,m-1,m);

% y now indexes into the elements of the
% target vector which needs to be 1
y = y' + indexes;

% Index into Y to change required elements to 1
Y(y) = 1;

% Retrieve back the matrix form
Y = reshape(Y, [n,m]);

% Multi-class representation ready
Y = Y';