function [weights] = weight_Chen2007_Entropy(X, Y)
% This weight form is used in 
%      Chen W, Yan J, Zhang B, Chen Z, Yang Q. Document transformation for multi-label feature selection text
%      categorization. The 7th IEEE International Conference on Data Mining (ICDM2007), Oct. 28-31, 2007,
%      Omaha, Nebraska, USA, pp.451-456.
% X: N*d matrix for training vectors, where each row indicates a training isntance
% Y: N*q matrix for label vectors with +1/0;
% weights: N*q weight matrix.

    [N,q] = size(Y);

    for i=1:N
        num_label=sum(Y(i,:));
        weights(i,:)=Y(i,:)/num_label;
    end
        
end