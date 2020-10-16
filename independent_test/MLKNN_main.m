 function  RankY= MLKNN_main(X,Y)
 
[n,D]=size(X);
m=size(Y,2);

[Prior,PriorN,Cond,CondN]=MLKNN_train(X,Y',9,1)
PredictValue2=MLKNN_test(X,Y',X,Y',9,Prior,PriorN,Cond,CondN)

PredictValue=PredictValue2'.*((Y+1)/2);

[SortResult,SortIndex]=sort(PredictValue,2,'descend');

RankY=zeros(n,m);

Y=(Y+1)/2;
for i=1:n
    Rankxum=1;
    for j=1:m
        if Y(i,SortIndex(i,j))~=0
            RankY(i,SortIndex(i,j))=m+1-Rankxum;
            Rankxum=Rankxum+1;
        end
    end
end

end

