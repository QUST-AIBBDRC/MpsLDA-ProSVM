function omega = RBF_kernel(Xtrain, kernel_pars,Xt)
% Construct the positive (semi-) definite and symmetric kernel matrix
% 
% >> Omega = kernel_matrix(X, kernel_fct, sig2)
% 
% This matrix should be positive definite if the kernel function
% satisfies the Mercer condition. Construct the kernel values for
% all test data points in the rows of Xt, relative to the points of X.
% 
% >> Omega_Xt = kernel_matrix(X, kernel_fct, sig2, Xt)
% 
%
% Full syntax
% 
% >> Omega = kernel_matrix(X, kernel_fct, sig2)
% >> Omega = kernel_matrix(X, kernel_fct, sig2, Xt)
% 
% Outputs    
%   Omega  : N x N (N x Nt) kernel matrix
% Inputs    
%   X      : N x d matrix with the inputs of the training data
%   kernel : Kernel type (by default 'RBF_kernel')
%   sig2   : Kernel parameter (bandwidth in the case of the 'RBF_kernel')
%   Xt(*)  : Nt x d matrix with the inputs of the test data
% 
% See also:
%  RBF_kernel, lin_kernel, kpca, trainlssvm, kentropy


% Copyright (c) 2002,  KULeuven-ESAT-SCD, License & help @ http://www.esat.kuleuven.ac.be/sista/lssvmlab



nb_data = size(Xtrain,1);


%if size(Xtrain,1)<size(Xtrain,2),
%  warning('dimension of datapoints larger than number of datapoints?');
%end
if(kernel_pars(1)==0)
    kernel_pars(1)=size(Xtrain,2);
end

  if nargin<3,    
    XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
    omega = XXh+XXh'-2*Xtrain*Xtrain';
    omega = exp(-omega./kernel_pars(1));
  else
    XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
    XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
    omega = XXh1+XXh2' - 2*Xtrain*Xt';
    omega = exp(-omega./kernel_pars(1));
  end
    