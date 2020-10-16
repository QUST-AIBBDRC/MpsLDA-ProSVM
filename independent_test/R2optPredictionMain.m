function output=R2optPredictionMain(test_data,W,W0)
%R2optPredictionMain returns the output of the test instances given the
%weight of the linear function of each training subset
%
%    Syntax
%
%       output=R2optPredictionMain(test_data,W,W0)
%
%    Description
%
%       ProSVM takes,
%           test_data        - An n2xd array, the ith instance of test set is stored in train_data(i,:)
%           W                - A Zx1 cell. Each entry is a d*(m+1) vector representing the weight of the Z-th training subset
%           W0                - A d*(m+1) array, see [1].
%      and returns,
%           output           - An n2x(m+1) array. The output of the ith testing instance on the jth class is stored in output(i,j). output(i,m+1) stores the output for the threshold label.
%
%[1] Miao Xu, Yu-Feng Li and Zhi-Hua Zhou. Multi-Label Learning with PRO Loss. In: AAAI'13.


R=size(W,1);

[n,d]=size(test_data);
m=size(W0,1)/(d+1)-1;

output=zeros(n,m+1);
for r=1:R
    output=output+R2optPrediction(test_data,W{r,1});
end
output=output+R2optPrediction(test_data,W0);
output=output/(R+1);

    