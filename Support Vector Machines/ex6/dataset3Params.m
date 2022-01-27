function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
val_err = zeros(64,3);
C1=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma1=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
k=0;
for i=1:length(C1)
  for j=1:length(sigma1)
    k++;
    % Train the SVM
    model= svmTrain(X, y, C1(i), @(x1, x2) gaussianKernel(x1, x2, sigma1(j)));
    
    predictions = svmPredict(model, Xval);

    val_err(k,1) = mean(double(predictions ~= yval));
    val_err(k,2)=C1(i);
    val_err(k,3)=sigma1(j);	    
  endfor 	
endfor

minerror= min(val_err(:,1));

for i=1:64
  
    if(minerror == val_err(i,1))
      C = val_err(i,2);
      sigma=val_err(i,3);
    endif
endfor








% =========================================================================

end
