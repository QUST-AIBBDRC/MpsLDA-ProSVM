function output=R2optPrediction(test_data,W)
%R2optPrediction returns the output of the test instances given the
%weight of the linear function of one training subset
%
%    Syntax
%
%       output=R2optPredictionMain(test_data,W,W0)
%
%    Description
%
%       ProSVM takes,
%           test_data        - An n2xd array, the ith instance of test set is stored in train_data(i,:)
%           W0               - A d*(m+1) array, the weight of one linear function
%      and returns,
%           output           - An n2x(m+1) array. The output of the ith testing instance on the jth class is stored in output(i,j). output(i,m+1) stores the output for the threshold label.



[n,d]=size(test_data);
m=size(W,1)/(d+1)-1;

test_data=[test_data ones(n,1)];

retValue=zeros(n,m+1);
output=zeros(n,m+1);
W=W';
for i=1:n
    for j=1:m+1
        output(i,j)=sum(test_data(i,:).*W(1,(j-1)*(d+1)+1:j*(d+1)));
    end
end
[SortResult SortIndex]=sort(output,2,'descend');
for i=1:n
    for j=1:m+1
        retValue(i,SortIndex(i,j))=m+2-j;
    end
end

end

