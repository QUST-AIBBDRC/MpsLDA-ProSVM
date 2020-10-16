function retValue=PerformanceMeasureMain(test_length,test_rank,output)
%PerformanceMeasureMain returns the measurements for the output
%
%    Syntax
%
%       retValue=PerformanceMeasureMain(test_length,test_rank,output)
%
%    Description
%
%       Average Precision takes,
%           test_rank        - An n2xm array, for the i-th test instance, the larger train_rank(i,j) is ,the higher the jth label ranked
%           test_length      - An n2x1 array, the i-th test instance has train_length(i,1) relevant labels.  
%           output           - An n2x(m+1) array. The output of the ith testing instance on the jth class is stored in output(i,j). output(i,m+1) stores the output for the threshold label.
%      and returns,
%			retValue         - A 8x1 array, presenting Pro Loss, Hamming Loss, Ranking Loss, One error, Average Precision, Coverage, Subset Accuracy and F1 in order.

retValue=zeros(1,1);
[retValue(1,1),retValue1]=ProLoss(test_length,test_rank,output);
% retValue(2,1)=HammingLoss(test_length,test_rank,output);
% retValue(4,1)=OneError(test_length,test_rank,output);
% retValue(6,1)=Coverage(test_length,test_rank,output);
% retValue(5,1)=AveragePrecision(test_length,test_rank,output);
% retValue(7,1)=SubsetAccuracy(test_length,test_rank,output);
% retValue(8,1)=F1(test_length,test_rank,output);



