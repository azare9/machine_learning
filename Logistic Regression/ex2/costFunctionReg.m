function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h=zeros(m);
h=X*theta;
h=sigmoid(h);

J=1/m*(-((y')*log(h))-((1-y)')*log(1-h));

summ=0;
for i=2:length(theta)
   summ=summ+theta(i)*theta(i);
endfor   

J=J+(summ*lambda)/(2*m);

%------------------gradient------------------------------------
for i=1:length(theta)
  vec=0;
  for j=1:m
    vec=vec+((h(j)-y(j))*X(j,i));
  endfor  
  grad(i)=1/m*vec;
  if(i>1)
    grad(i)=grad(i)+(lambda*theta(i))/m;
  endif  	
endfor 

% =============================================================

end
