function RankY = LIFT_mian(X,Y)   
%现在的X是test+train数据，标签也是同样

[n,~]=size(X);
m=size(Y,2);
ratio=0.1
svm.type='RBF'
svm.para=0.1;

[PredictValue1]=LIFT(X,Y',X,Y',ratio,svm);
PredictValue=PredictValue1';
PredictValue=PredictValue.*((Y+1)/2);

[~,SortIndex]=sort(PredictValue,2,'descend');

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