function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
[nr, nc] = size(z)
              
if (nr == 1 && nc == 1)
   g = 1/(1+exp(-z));
elseif (nr > 1 && nc == 1)      	
   for i=1:nr            	
     g(i) = 1/(1 + exp(-z(i)));         
   endfor
else
   for i=1:nr
     for j=1:nc
       g(i,j)= 1/(1 + exp(-z(i,j)));
     endfor
   endfor      	    	             
endif




% =============================================================

end
