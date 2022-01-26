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

Theta1_delta = zeros(size(Theta1));
Theta2_delta = zeros(size(Theta2));
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1
y_l=eye(num_labels);
X=[ones(m,1) X];

cumulator_y=0;
%%%%%%%%%%% var for back pro %%%%%%%%
delta_y=zeros(length(y),1);
[nr nc]=size(Theta2);
[nr1 nc1]=size(Theta1);
delta_hiden_layer1=zeros(nc,1);
%%%%%%%%%%% var for back pro %%%%%%%%
for i=1:m

%%%%%%%%%forword propagation for the object X(i)%%%%%%%%%%%
%  h=zeros(25,1);
  z2=Theta1*X(i,:)';
  a2=sigmoid(z2);

%  
  z3=Theta2*[1;a2];
  a3=sigmoid(z3);
%%%%%%%%%forword propagation for the object X(i)%%%%%%%%%%%

if(y(i)  == 10)
   cumulator_y =cumulator_y + y_l(:, num_labels)'*log(a3)+(1-y_l(:, num_labels))'*log(1-a3);
else 
   cumulator_y =cumulator_y +  y_l(:, y(i))'*log(a3)+(1-y_l(:, y(i)))'*log(1-a3);  
endif

%%%%%%%%%***************************************%%%%%%%%%%%
%%%%%%%%%***************************************%%%%%%%%%%%
%%%%%%%%%***************************************%%%%%%%%%%%
%%%%%%%%%backword propagation for the object X(i)%%%%%%%%%%
delta_y=a3-y_l(:,y(i));
delta_hiden_layer1= (Theta2'*delta_y).*[1;a2].*(1-[1;a2]);
Theta2_delta = Theta2_delta + delta_y*[1;a2]';
Theta1_delta = Theta1_delta + delta_hiden_layer1(2:end)*X(i,:);


%%%%%%%%%backword propagation for the object X(i)%%%%%%%%%%

endfor

%%%%%%%%%%%%%%%%regularised gradient%%%%%%%%%%%%%%%%%%%%%

for j=1:nr
  for k=1:nc
  if (k==1)
    Theta2_grad(j,k) = 1/m * Theta2_delta(j,k);  
  else
    Theta2_grad(j,k) = (1/m) * Theta2_delta(j,k) + (lambda/m)*Theta2(j,k);
  endif       
  endfor
endfor

for j=1:nr1
  for k=1:nc1
  if (k==1)
    Theta1_grad(j,k) = 1/m * Theta1_delta(j,k);  
  else
    Theta1_grad(j,k) =1/m * Theta1_delta(j,k) + (lambda/m)*Theta1(j,k);
  endif       
  endfor
endfor

%%%%%%%%%%%%%%%%regularised gradient%%%%%%%%%%%%%%%%%%%%%

J=(-1/m)*cumulator_y;

%%%%%%%%%%%%%regularized cost function%%%%%%%%%%%%%%
sum_theta1=0;
[nr nc]=size(Theta1);
for i=1:nr
   for j=2:nc
     sum_theta1= sum_theta1 + Theta1(i,j)*Theta1(i,j);     
     
   endfor	
endfor

sum_theta2=0;
[nr2 nc2]=size(Theta2);
for i=1:nr2
   for j=2:nc2
     sum_theta2= sum_theta2 + Theta2(i,j)*Theta2(i,j);     
     
   endfor	
endfor

J=J + (lambda/(2*m))*(sum_theta1+sum_theta2);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
