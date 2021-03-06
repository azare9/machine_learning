function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).

[nr nc] = size(g);

if (nr ==1 && nc == 1)
  g=sigmoid(z)*(1-sigmoid(z));
elseif(nc == 1)
 for i=1:nr
   g(i)=sigmoid(z(i))*(1-sigmoid(z(i)));
   
 endfor
elseif(nc > 1)
 
 for i=1:nr
   for j=1:nc
     g(i,j)=sigmoid(z(i,j))*(1-sigmoid(z(i,j)));
   endfor
 endfor
endif













% =============================================================




end
