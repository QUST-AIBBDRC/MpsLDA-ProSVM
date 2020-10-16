function [loss,output]=ProSVM(train_data,train_length,train_rank,...
    test_data,test_length,test_rank,lambda,str_tag,part_size)
%ProSVM implements the algorithms ProSVM, ProSVM-A, ProSVM' and ProSVM-A' proposed in [1]
%
%    Syntax
%
%       [loss,output,train_time]=ProSVM(train_data,train_length,train_rank,test_data,test_length,test_rank,lambda,str_tag)
%
%    Description
%
%       ProSVM takes,
%           train_data       - An n1xd array, the ith instance of training set is stored in train_data(i,:)
%           train_rank       - An n1xm array, for the ith training
%           instance, the train_rank(i,j)th label is ranked at the jth
%           place
%           train_length     - An n1x1 array, the ith training instance has train_length(i,1) relevant labels.  
%           test_data        - An n2xd array, the ith instance of test set is stored in train_data(i,:)
%           test_rank        - An n2xm array, for the ith test instance, the train_rank(i,j)th label is ranked at the jth
%           test_length      - An n2x1 array, the ith test instance has train_length(i,1) relevant labels.  
%           lambda           - The regularization parameter, default=1
%           str_tag          - A string represent which method to take. It can be one of the following values 'ProSVM', 'ProSVM-A','ProSVM_Prime', 'ProSVM-A_Prime'
%           part_size        - A value determines the number Z of training subset. The larger part_size is, the smaller Z is, default=10^7.
%      and returns,
%			loss             - A 8x1 array, presenting Pro Loss, Hamming Loss, Ranking Loss, One error, Average Precision, Coverage, Subset Accuracy and F1 in order.
%           output           - An n2x(m+1) array. The output of the ith testing instance on the jth class is stored in output(i,j). output(i,m+1) stores the output for the threshold label.
%           train_time       - The training time measured in seconds.
%
%[1] Miao Xu, Yu-Feng Li and Zhi-Hua Zhou. Multi-Label Learning with PRO Loss. In: AAAI'13.

%rng('shuffle');
rand('seed',sum(100*clock))

%Check Parameter
if nargin<7
    lambda=1;
    str_tag='ProSVM';
    part_size=10^7;
elseif nargin<8
    str_tag='ProSVM';
    part_size=10^7;
elseif nargin<9
    part_size=10^7;
end

%disp all the necessary information
disp(['method name: ' str_tag]);
disp(['lambda=' num2str(lambda)]);
disp(['part_size=' num2str(part_size)]);

%Initialization
t1=tic;
eta=0.1;
d=size(train_data,2);
d=d+1;
[n, m]=size(train_rank);
g_num=d*(m+1);
if strcmp(str_tag,'ProSVM')
    Z=ceil(sum((2*m-1-train_length).*train_length/2+m)*d/part_size);
elseif strcmp(str_tag,'ProSVM-A')
    Z=ceil(sum(train_length+m-1)*d/part_size);
elseif strcmp(str_tag,'ProSVM_Prime')
    Z=ceil(sum((m-train_length).*train_length+m)*d/part_size);
elseif strcmp(str_tag,'ProSVM-A_Prime')
    Z=ceil(sum(n*m)*d/part_size);
end

%divide the original training data into Z parts
instanceindex=randperm(n)';
everysamplesize=round(n/Z);
newX=cell(Z,1);
newY=cell(Z,1);
MLC=cell(Z,1);
for r=1:Z
        if r*everysamplesize<n
            [newX{r,1} newY{r,1} MLC{r,1}]=SparseTheMatrix(train_data(instanceindex((r-1)*everysamplesize+1:r*everysamplesize,:),:),train_length(instanceindex((r-1)*everysamplesize+1:r*everysamplesize,:),:),train_rank(instanceindex((r-1)*everysamplesize+1:r*everysamplesize,:),:),str_tag); 
        else
            [newX{r,1} newY{r,1} MLC{r,1}]=SparseTheMatrix(train_data(instanceindex((r-1)*everysamplesize+1:n,:),:),train_length(instanceindex((r-1)*everysamplesize+1:n,:),:),train_rank(instanceindex((r-1)*everysamplesize+1:n,:),:),str_tag);
            break;
        end  
end

Z=r;

%initialization for optimization
W0=zeros(d*(m+1),1);
alpha=cell(Z,1);
W=cell(Z,1);
for r=1:Z
    alpha{r,1}=zeros(d*(m+1),1);
end

tag=0;i=1;
lagrangedual=zeros(1,50);
while(tag==0)
    disp(['Iteration ' num2str(i)]);
    
    for r=1:Z
        g=eta*W0-alpha{r,1};

      
        W{r,1}=UpdataW_r(newX{r,1},newY{r,1},MLC{r,1},lambda,g,g_num,eta);
        
    end
    
    %Update W0
    W0=zeros(d*(m+1),1);
    for r=1:Z
        W0=W0+alpha{r,1}+W{r,1}*eta;
    end
    W0=W0/eta/Z;
    
    %Updata alpha
    for r=1:Z
       alpha{r,1}=alpha{r,1}+eta*(W{r,1}-W0);
    end 
    
    %Calculate the objective
    xumtemp=0;
    for rxum=1:Z
        if rxum<Z
            xumtemp=xumtemp+lambda*R2Loss2(train_length(instanceindex((rxum-1)*everysamplesize+1:rxum*everysamplesize,:),:),train_rank(instanceindex((rxum-1)*everysamplesize+1:rxum*everysamplesize,:),:),R2optPrediction(train_data(instanceindex((rxum-1)*everysamplesize+1:rxum*everysamplesize,:),:),W{rxum,1}),str_tag);
        else
            xumtemp=xumtemp+lambda*R2Loss2(train_length(instanceindex((rxum-1)*everysamplesize+1:n,:),:),train_rank(instanceindex((rxum-1)*everysamplesize+1:n,:),:),R2optPrediction(train_data(instanceindex((rxum-1)*everysamplesize+1:n,:),:),W{rxum,1}),str_tag); 
        end
        xumtemp=xumtemp+norm(W{rxum,1})^2/2;
    end
    for r=1:Z
       xumtemp=xumtemp+sum(alpha{r,1}.*(W{r,1}-W0));
    end
    for r=1:Z
        xumtemp=xumtemp+norm(W{r,1}-W0)^2/2/eta;
    end
    lagrangedual(1,i)=xumtemp;
    
    if i>1 && abs(lagrangedual(1,i-1)-lagrangedual(1,i))<0.01
        tag=1;
    elseif i>99
        tag=2;
    elseif Z==1
        tag=3;
    end
    i=i+1;
end

train_time=toc(t1);

output=R2optPredictionMain(test_data,W,W0);
loss=PerformanceMeasureMain(test_length,test_rank,output);