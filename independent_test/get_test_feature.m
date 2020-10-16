function test_feature=get_test_feature(test_data,train_data,train_feature)

n_fea=size(train_feature,2);
n_test=size(test_data,1);
test_feature=zeros(n_test,n_fea);
para_set=[0.1,1,10;0.01,0.1,1];
best_paras=zeros(n_fea,2);
for i=1:n_fea
    [best_para1,best_para2]=cv_para(train_data,train_feature(:,i),para_set);
    best_paras(i,:)=[best_para1,best_para2];
    model=svmtrain(train_feature(:,i),train_data,['-s 3 -t 2 -g ',num2str(best_para1),' -p ',num2str(best_para2)]);
    [fea,tmp1,tmp2]=svmpredict(test_feature(:,i),test_data,model);
    test_feature(:,i)=fea;
end
    
end

function [best_para1,best_para2]=cv_para(data,labels,para_set)
num_folds=5;
n=size(data,1);
n_paras=length(para_set);
idx=randperm(n);
n_test=floor(n/num_folds);
test_idx=zeros(num_folds,n_test);
train_idx=zeros(num_folds,n-n_test);
for i=1:num_folds
    test_idx(i,:)=idx((i-1)*n_test+1:i*n_test);
    tmp=1:n;
    tmp(test_idx(i,:))=[];
    train_idx(i,:)=tmp;
end
best_accs=inf;
best_para1=1;best_para2=1;
for i=1:n_paras
    for k=1:n_paras
        one_accs=0;
        for j=1:num_folds
            model=svmtrain(labels(train_idx(j,:)),data(train_idx(j,:),:),['-s 3 -t 2 -g ',num2str(para_set(1,i)),' -p ',num2str(para_set(2,k))]);
            [pres,acc,tmp2]=svmpredict(labels(test_idx(j,:)),data(test_idx(j,:),:),model);
            one_accs=one_accs+acc(2);
        end
        if(best_accs>one_accs)
            best_para1=para_set(1,i);
            best_para2=para_set(2,k);
            best_accs=one_accs;
        end
    end
end
end