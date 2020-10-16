
load('EXAMPLE.mat')
shuju=gozhengli;
L=vilabel;
[YANGBEN,WEISHU]=size(L);
%% ½Ü¿Ëµ¶²âÊÔ+LIFT
lambda=4;
str_tag= 'ProSVM';
part_size=10^7;
P=[];Q=[];
for i=1:YANGBEN
    A=shuju;
    B=L';
    test_data=A(i,:);test_target=B(:,i)';
    A(i,:)=[];B(:,i)=[];
    train_data=A;train_target=B';
 [train_data,train_length,train_rank,test_data,test_length,test_rank]...
    =orderSynthetic1(train_data, train_target, test_data,test_target);
[loss,output]=ProSVM(train_data,train_length,train_rank,...
    test_data,test_length,test_rank,lambda);
Q=[Q,loss];
 [num_class,num_testing]=size(test_target');
    %Threshold=get_threshold(train_data,train_target,test_data,net);
    Threshold=output(:,end)';
    Output123=output(:,1:end-1)';
    Pre_Labels=zeros(num_class,num_testing);
    for t=1:num_testing
        for k=1:num_class
            if(Output123(k,t)>=Threshold(1,t))
                Pre_Labels(k,t)=1;
            else
                Pre_Labels(k,t)=-1;
            end
        end
    end
P=[P;Pre_Labels'];
clear A B      
end
Q=mean(Q,2);
HL=Hamming_loss(P,L);
AP=Average_precision(P,L);
CV=coverage_new(P,L);
RL=Ranking_loss(P,L);
F1= F1_example(P,L);
ACC=Subset_accuracy(P,L);
Q(2,1)=HL;Q(3,1)=RL;Q(4,1)=AP;Q(5,1)=ACC;Q(6,1)=CV;Q(7,1)=F1;
OAA=0;
for i=1:YANGBEN
if P(i,:)==L(i,:);
OAA=OAA+1;
end
end
zuiOAA=OAA;
clear OAA
zhengque(:,:)=gelei(L,P,WEISHU,YANGBEN);