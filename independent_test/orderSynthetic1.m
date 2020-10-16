function [train_data,train_length,train_rank,test_data,test_length,test_rank]...
    =orderSynthetic1(train_data, train_target, test_data,test_target)
%orderSynthetic synthetic the relevant labels' rank by adding the result of
%three benchmark multi-label methods [1][2][3]. These codes are downloaded
%from http://cse.seu.edu.cn/people/zhangml/.
%
%    Syntax
%
%       [train_data,train_length,train_rank,test_data,test_length,test_rank]=orderSynthetic(train_data, train_target, test_data,test_target)
%
%    Description
%
%       ProSVM takes,
%           train_data       - An n1xd array, the ith instance of training set is stored in train_data(i,:)
%           train_target     - An n1xm array, train_target(i,j)=1 shows the j-th label is relevant to the i-th training instance; train_target(i,j)=0 shows the j-th label is irrelevant to the i-th training instance
%           test_data        - An n2xd array, the ith instance of test set is stored in train_data(i,:)
%           test_target      - An n2xm array, test_target(i,j)=1 shows the j-th label is relevant to the i-th test instance; test_target(i,j)=0 shows the j-th label is irrelevant to the i-th test instance
%      and returns,
%           train_data       - An n1xd array, the ith instance of training set is stored in train_data(i,:)
%           train_rank       - An n1xm array, for the ith training instance, the larger train_rank(i,j) is ,the higher the jth label ranked
%           train_length     - An n1x1 array, the ith training instance has train_length(i,1) relevant labels.  
%           test_data        - An n2xd array, the ith instance of test set is stored in train_data(i,:)
%           test_rank        - An n2xm array, for the ith test instance, the larger train_rank(i,j) is ,the higher the jth label ranked
%           test_length      - An n2x1 array, the ith test instance has train_length(i,1) relevant labels.  
%
%[1]Min-Ling Zhang, Jos?Peña, Victor Robles. Feature selection for
%multi-label naive bayes classification. Information Science 179(19),
%3218-3229, 2009.
%[2]Min-Ling Zhang, Zhi-Hua Zhou. Multi-label neural networks with
%applications to functional genomics and text categorization. IEEE TKDE,
%18(10), 1338-1351, 2006.
%[3]Min-Ling Zhang, Zhi-Hua Zhou. Multi-Label Learning by Instance 
%Differentiation. In: AAAI, 2007.


addpath orderSynthetic;

%disp('Rank generation using Multi-label Neural Network: ');
%Predict_BPMLL=BPMLL([test_data;train_data],[test_target;train_target]);
Predict_MLKNN=MLKNN_main([test_data;train_data],[test_target;train_target]);

%disp('Rank generation using Multi-label Instance Differentiation: ');
%Predict_INSDIF = InsDif_main([test_data;train_data],[test_target;train_target]);

%disp('Rank generation using Multi-label Naive Bayes: ');
%Predict_MLNB=MLNB_main([test_data;train_data],[test_target;train_target]);
Predict_LIFT = LIFT_mian([test_data;train_data],[test_target;train_target]);   

Y=[test_target;train_target];
Y=(Y+1)/2;
Predict=(Predict_MLKNN +Predict_LIFT)/2;
n=size([test_target;train_target],1);
test_size=size(test_data,1);
LabelLength=zeros(n,1);

for i=1:n
    LabelLength(i,1)=nnz(Y(i,:));
end
[~,LabelIndex]=sort(Predict,2,'descend');

test_length=LabelLength(1:test_size,:);
train_length=LabelLength(test_size+1:n,:);
test_rank=LabelIndex(1:test_size,:);
train_rank=LabelIndex(test_size+1:n,:);