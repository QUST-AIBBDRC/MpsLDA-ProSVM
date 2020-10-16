function [newX,newY,MLC] = SparseTheMatrix(train_data,train_length,train_rank,str_tag)
%SparseTheMatrix make the training data into a matrix fitting the QP model

if strcmp(str_tag,'ProSVM')
    
    d=size(train_data,2);
    d=d+1;
    [n,m]=size(train_rank);

    train_data=[train_data ones(n,1)];
    nzmax=sum((2*m-1-train_length).*train_length/2+m);
    index=zeros(1,nzmax);
    rows=zeros(1,nzmax*2*d);
    columns=zeros(1,nzmax*2*d);
    value=zeros(1,nzmax*2*d);
    count=1;
    MLCrows=zeros(1,nzmax);
    MLCvalue=zeros(1,nzmax);
    for i=n:-1:1
        for j=1:train_length(i)
            for k=j+1:train_length(i)
                rows(1,(count-1)*2*d+1:count*2*d)=ones(1,2*d)*((i-1)*(m+1)^2+(train_rank(i,j)-1)*(m+1)+train_rank(i,k));
                columns(1,(count-1)*2*d+1:count*2*d)=[(train_rank(i,j)-1)*d+1:train_rank(i,j)*d (train_rank(i,k)-1)*d+1:train_rank(i,k)*d];
                value(1,(count-1)*2*d+1:count*2*d)=[train_data(i,:) -train_data(i,:)];
                index(1,count)=(i-1)*(m+1)^2+(train_rank(i,j)-1)*(m+1)+train_rank(i,k);
                MLCrows(1,count)=(i-1)*(m+1)^2+(train_rank(i,j)-1)*(m+1)+train_rank(i,k);
                MLCvalue(1,count)=2/train_length(i)/(train_length(i)-1);
                count=count+1;
            end
            for k=train_length(i)+1:m
                rows(1,(count-1)*2*d+1:count*2*d)=ones(1,2*d)*((i-1)*(m+1)^2+(train_rank(i,j)-1)*(m+1)+train_rank(i,k));
                columns(1,(count-1)*2*d+1:count*2*d)=[(train_rank(i,j)-1)*d+1:train_rank(i,j)*d (train_rank(i,k)-1)*d+1:train_rank(i,k)*d];
                value(1,(count-1)*2*d+1:count*2*d)=[train_data(i,:) -train_data(i,:)];
                index(1,count)=(i-1)*(m+1)^2+(train_rank(i,j)-1)*(m+1)+train_rank(i,k);
                MLCrows(1,count)=(i-1)*(m+1)^2+(train_rank(i,j)-1)*(m+1)+train_rank(i,k);
                MLCvalue(1,count)=1/train_length(i)/(m-train_length(i));
                count=count+1;
            end
        end
        
        for j=1:train_length(i)
            rows(1,(count-1)*2*d+1:count*2*d)=ones(1,2*d)*((i-1)*(m+1)^2+(train_rank(i,j)-1)*(m+1)+m+1);
            columns(1,(count-1)*2*d+1:count*2*d)=[(train_rank(i,j)-1)*d+1:train_rank(i,j)*d,m*d+1:(m+1)*d];
            value(1,(count-1)*2*d+1:count*2*d)=[train_data(i,:) -train_data(i,:)];
            index(1,count)=(i-1)*(m+1)^2+(train_rank(i,j)-1)*(m+1)+m+1;
            MLCrows(1,count)=(i-1)*(m+1)^2+(train_rank(i,j)-1)*(m+1)+m+1;
            MLCvalue(1,count)=1/train_length(i);
            count=count+1;
        end
        
        for k=train_length(i)+1:m
            rows(1,(count-1)*2*d+1:count*2*d)=ones(1,2*d)*((i-1)*(m+1)^2+m*(m+1)+train_rank(i,k));
            columns(1,(count-1)*2*d+1:count*2*d)=[m*d+1:(m+1)*d (train_rank(i,k)-1)*d+1:train_rank(i,k)*d];
            value(1,(count-1)*2*d+1:count*2*d)=[train_data(i,:) -train_data(i,:)];
            index(1,count)=(i-1)*(m+1)^2+m*(m+1)+train_rank(i,k);
            MLCrows(1,count)=(i-1)*(m+1)^2+m*(m+1)+train_rank(i,k);
            MLCvalue(1,count)=1/(m-train_length(i));
            count=count+1;
        end

    end

    newX=sparse(rows,columns,value);
    MLC=sparse(MLCrows,ones(1,count-1),MLCvalue);
    newX=newX(index,:);
    MLC=MLC(index,:);
    newY=ones(n*(m+1)^2,1);
    newY=newY(index);
    
elseif strcmp(str_tag,'ProSVM-A')
    d=size(train_data,2);
    d=d+1;
    [n,m]=size(train_rank);
    train_data=[train_data ones(n,1)];

    nzmax=sum(train_length-1+m);
    index=zeros(1,nzmax);
    rows=zeros(1,nzmax*2*d);
    columns=zeros(1,nzmax*2*d);
    value=zeros(1,nzmax*2*d);
    count=1;
    MLCrows=zeros(1,nzmax);
    MLCvalue=zeros(1,nzmax);
    for i=n:-1:1
        for j=1:train_length(i)-1
            k=j+1;
            rows(1,(count-1)*2*d+1:count*2*d)=ones(1,2*d)*((i-1)*(m+1)^2+(train_rank(i,j)-1)*(m+1)+train_rank(i,k));
            columns(1,(count-1)*2*d+1:count*2*d)=[(train_rank(i,j)-1)*d+1:train_rank(i,j)*d (train_rank(i,k)-1)*d+1:train_rank(i,k)*d];
            value(1,(count-1)*2*d+1:count*2*d)=[train_data(i,:) -train_data(i,:)];
            index(1,count)=(i-1)*(m+1)^2+(train_rank(i,j)-1)*(m+1)+train_rank(i,k);
            MLCrows(1,count)=(i-1)*(m+1)^2+(train_rank(i,j)-1)*(m+1)+train_rank(i,k);%貌似MLCrows和index是一样的
            MLCvalue(1,count)=2/train_length(i)/(train_length(i)-1)*j*(train_length(i)-j);
            count=count+1;
        end
        
        for j=1:train_length(i)
            rows(1,(count-1)*2*d+1:count*2*d)=ones(1,2*d)*((i-1)*(m+1)^2+(train_rank(i,j)-1)*(m+1)+m+1);
            columns(1,(count-1)*2*d+1:count*2*d)=[(train_rank(i,j)-1)*d+1:train_rank(i,j)*d,m*d+1:(m+1)*d];
            value(1,(count-1)*2*d+1:count*2*d)=[train_data(i,:) -train_data(i,:)];
            index(1,count)=(i-1)*(m+1)^2+(train_rank(i,j)-1)*(m+1)+m+1;
            MLCrows(1,count)=(i-1)*(m+1)^2+(train_rank(i,j)-1)*(m+1)+m+1;
            MLCvalue(1,count)=3/train_length(i);
            count=count+1;
        end
        
        for k=train_length(i)+1:m
            rows(1,(count-1)*2*d+1:count*2*d)=ones(1,2*d)*((i-1)*(m+1)^2+m*(m+1)+train_rank(i,k));
            columns(1,(count-1)*2*d+1:count*2*d)=[m*d+1:(m+1)*d (train_rank(i,k)-1)*d+1:train_rank(i,k)*d];
            value(1,(count-1)*2*d+1:count*2*d)=[train_data(i,:) -train_data(i,:)];
            index(1,count)=(i-1)*(m+1)^2+m*(m+1)+train_rank(i,k);
            MLCrows(1,count)=(i-1)*(m+1)^2+m*(m+1)+train_rank(i,k);
            MLCvalue(1,count)=3/(m-train_length(i));
            count=count+1;
        end

    end

    newX=sparse(rows,columns,value);
    MLC=sparse(MLCrows,ones(1,count-1),MLCvalue);

    newX=newX(index,:);
    MLC=MLC(index,:);
    newY=ones(n*(m+1)^2,1);
    newY=newY(index);
    
    elseif strcmp(str_tag,'ProSVM_Prime')
        d=size(train_data,2);
		d=d+1;
		[n,m]=size(train_rank);
		train_data=[train_data ones(n,1)];
		nzmax=sum((m-train_length).*train_length+m);
		index=zeros(1,nzmax);
		rows=zeros(1,nzmax*2*d);
		columns=zeros(1,nzmax*2*d);
		value=zeros(1,nzmax*2*d);
		count=1;
		MLCrows=zeros(1,nzmax);
		MLCvalue=zeros(1,nzmax);
		for i=n:-1:1
			for j=1:train_length(i)
				for k=train_length(i)+1:m
					rows(1,(count-1)*2*d+1:count*2*d)=ones(1,2*d)*((i-1)*(m+1)^2+(train_rank(i,j)-1)*(m+1)+train_rank(i,k));
					columns(1,(count-1)*2*d+1:count*2*d)=[(train_rank(i,j)-1)*d+1:train_rank(i,j)*d (train_rank(i,k)-1)*d+1:train_rank(i,k)*d];
					value(1,(count-1)*2*d+1:count*2*d)=[train_data(i,:) -train_data(i,:)];
					index(1,count)=(i-1)*(m+1)^2+(train_rank(i,j)-1)*(m+1)+train_rank(i,k);
					MLCrows(1,count)=(i-1)*(m+1)^2+(train_rank(i,j)-1)*(m+1)+train_rank(i,k);
					MLCvalue(1,count)=1/train_length(i)/(m-train_length(i));
					count=count+1;
				end
			end
			
			for j=1:train_length(i)
				rows(1,(count-1)*2*d+1:count*2*d)=ones(1,2*d)*((i-1)*(m+1)^2+(train_rank(i,j)-1)*(m+1)+m+1);
				columns(1,(count-1)*2*d+1:count*2*d)=[(train_rank(i,j)-1)*d+1:train_rank(i,j)*d,m*d+1:(m+1)*d];
				value(1,(count-1)*2*d+1:count*2*d)=[train_data(i,:) -train_data(i,:)];
				index(1,count)=(i-1)*(m+1)^2+(train_rank(i,j)-1)*(m+1)+m+1;
				MLCrows(1,count)=(i-1)*(m+1)^2+(train_rank(i,j)-1)*(m+1)+m+1;
				MLCvalue(1,count)=1/train_length(i);
				count=count+1;
			end
			
			for k=train_length(i)+1:m
				rows(1,(count-1)*2*d+1:count*2*d)=ones(1,2*d)*((i-1)*(m+1)^2+m*(m+1)+train_rank(i,k));
				columns(1,(count-1)*2*d+1:count*2*d)=[m*d+1:(m+1)*d (train_rank(i,k)-1)*d+1:train_rank(i,k)*d];
				value(1,(count-1)*2*d+1:count*2*d)=[train_data(i,:) -train_data(i,:)];
				index(1,count)=(i-1)*(m+1)^2+m*(m+1)+train_rank(i,k);
				MLCrows(1,count)=(i-1)*(m+1)^2+m*(m+1)+train_rank(i,k);
				MLCvalue(1,count)=1/(m-train_length(i));
				count=count+1;
			end

		end

		newX=sparse(rows,columns,value);
		MLC=sparse(MLCrows,ones(1,count-1),MLCvalue);

		newX=newX(index,:);
		MLC=MLC(index,:);
		newY=ones(n*(m+1)^2,1);
		newY=newY(index);

        
    elseif strcmp(str_tag,'ProSVM-A_Prime')
        		d=size(train_data,2);
		d=d+1;
		[n,m]=size(train_rank);
		train_data=[train_data ones(n,1)];
		nzmax=m*n;
		index=zeros(1,nzmax);
		rows=zeros(1,nzmax*2*d);
		columns=zeros(1,nzmax*2*d);
		value=zeros(1,nzmax*2*d);
		count=1;
		MLCrows=zeros(1,nzmax);
		MLCvalue=zeros(1,nzmax);
		for i=n:-1:1
			for j=1:train_length(i)
				rows(1,(count-1)*2*d+1:count*2*d)=ones(1,2*d)*((i-1)*(m+1)^2+(train_rank(i,j)-1)*(m+1)+m+1);
				columns(1,(count-1)*2*d+1:count*2*d)=[(train_rank(i,j)-1)*d+1:train_rank(i,j)*d,m*d+1:(m+1)*d];
				value(1,(count-1)*2*d+1:count*2*d)=[train_data(i,:) -train_data(i,:)];
				index(1,count)=(i-1)*(m+1)^2+(train_rank(i,j)-1)*(m+1)+m+1;
				MLCrows(1,count)=(i-1)*(m+1)^2+(train_rank(i,j)-1)*(m+1)+m+1;
				MLCvalue(1,count)=1/train_length(i);
				count=count+1;
			end
			
			for k=train_length(i)+1:m
				rows(1,(count-1)*2*d+1:count*2*d)=ones(1,2*d)*((i-1)*(m+1)^2+m*(m+1)+train_rank(i,k));
				columns(1,(count-1)*2*d+1:count*2*d)=[m*d+1:(m+1)*d (train_rank(i,k)-1)*d+1:train_rank(i,k)*d];
				value(1,(count-1)*2*d+1:count*2*d)=[train_data(i,:) -train_data(i,:)];
				index(1,count)=(i-1)*(m+1)^2+m*(m+1)+train_rank(i,k);
				MLCrows(1,count)=(i-1)*(m+1)^2+m*(m+1)+train_rank(i,k);
				MLCvalue(1,count)=1/(m-train_length(i));
				count=count+1;
			end

		end

		newX=sparse(rows,columns,value);
		MLC=sparse(MLCrows,ones(1,count-1),MLCvalue);

		newX=newX(index,:);
		MLC=MLC(index,:);
		newY=ones(n*(m+1)^2,1);
		newY=newY(index);
end


end

