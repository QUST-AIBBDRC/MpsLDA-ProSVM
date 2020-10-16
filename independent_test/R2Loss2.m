function sumLoss=R2Loss2(test_length,test_rank,output,str_tag)
%R2LOSS2 Calculate the loss part in the optimization objective
%  
%    Syntax
%
%       sumLoss=R2Loss2(test_length,test_rank,output,str_tag)
%
%    Description
%
%       Average Precision takes,
%           test_rank        - An n2xm array, for the i-th test instance, the larger train_rank(i,j) is ,the higher the jth label ranked
%           test_length      - An n2x1 array, the i-th test instance has train_length(i,1) relevant labels.  
%           output           - An n2x(m+1) array. The output of the ith testing instance on the jth class is stored in output(i,j). output(i,m+1) stores the output for the threshold label.
%           str_tag          - A string represent which optimization problem is. It can be one of the following values 'ProSVM', 'ProSVM-A','ProSVM_Prime', 'ProSVM-A_Prime'
%      and returns,
%			sumLoss          - A Value represent the returned loss in the optimization objective

if strcmp(str_tag,'ProSVM')
    [n,m]=size(test_rank);

	Loss1=0;
	for i=1:n
		temp=0;
		for j=1:test_length(i,1)
			if output(i,test_rank(i,j))<output(i,m+1)
				temp=temp+1;
			elseif output(i,test_rank(i,j))==output(i,m+1)
				temp=temp+1/2;
			end
		end
		if test_length(i,1)~=0
			temp=temp/test_length(i,1);
		end
		Loss1=Loss1+temp;
	end
	Loss1=Loss1/n;

	Loss2=0;
	for i=1:n
		temp=0;
		for j=test_length(i,1)+1:m
			if output(i,test_rank(i,j))>output(i,m+1)
				temp=temp+1;
			elseif output(i,test_rank(i,j))==output(i,m+1)
				temp=temp+1/2;
			end
		end
		if (m-test_length(i,1))~=0
			temp=temp/(m-test_length(i,1));
		end
		Loss2=Loss2+temp;
	end
	Loss2=Loss2/n;

	Loss3=0;
	for i=1:n
		temp=0;
		for j=1:test_length(i,1)
			for k=test_length(i,1)+1:m
				if output(i,test_rank(i,j))<output(i,test_rank(i,k))
					temp=temp+1;
				elseif output(i,test_rank(i,j))==output(i,test_rank(i,k))
					temp=temp+1/2;
				end
			end
		end
		if test_length(i,1)~=0 && m-test_length(i,1)~=0
			temp=temp/test_length(i,1)/(m-test_length(i,1));
		end
		Loss3=Loss3+temp;
	end
	Loss3=Loss3/n;

	Loss4=0;
	for i=1:n
		temp=0;
		for j=1:test_length(i,1)-1
			for k=j+1:test_length(i,1)
				if output(i,test_rank(i,j))<output(i,test_rank(i,k))
					temp=temp+1;
				elseif output(i,test_rank(i,j))==output(i,test_rank(i,k))
					temp=temp+1/2;
				end
			end
		end
		if test_length(i,1)~=0 && test_length(i,1)~=1
			temp=temp*2/(test_length(i,1)-1)/test_length(i,1);
		end
		Loss4=Loss4+temp;
	end
	Loss4=Loss4/n;

	sumLoss=Loss1+Loss2+Loss3+Loss4;
elseif strcmp(str_tag,'ProSVM-A')
    [n,m]=size(test_rank);

	Loss1=0;
	for i=1:n
		temp=0;
		for j=1:test_length(i,1)
			if output(i,test_rank(i,j))<output(i,m+1)
				temp=temp+1;
			elseif output(i,test_rank(i,j))==output(i,m+1)
				temp=temp+1/2;
			end
		end
		if test_length(i,1)~=0
			temp=temp/test_length(i,1);
		end
		Loss1=Loss1+temp;
	end
	Loss1=Loss1/n;

	Loss2=0;
	for i=1:n
		temp=0;
		for j=test_length(i,1)+1:m
			if output(i,test_rank(i,j))>output(i,m+1)
				temp=temp+1;
			elseif output(i,test_rank(i,j))==output(i,m+1)
				temp=temp+1/2;
			end
		end
		if (m-test_length(i,1))~=0
			temp=temp/(m-test_length(i,1));
		end
		Loss2=Loss2+temp;
	end
	Loss2=Loss2/n;

	Loss4=0;
	for i=1:n
		temp=0;
		for j=1:test_length(i,1)-1
			for k=j+1:j+1
				if output(i,test_rank(i,j))<output(i,test_rank(i,k))
					temp=temp+j*(test_length(i,1)-j);
				elseif output(i,test_rank(i,j))==output(i,test_rank(i,k))
					temp=temp+1/2*j*(test_length(i,1)-j);
				end
			end
		end
		if test_length(i,1)~=0 && test_length(i,1)~=1
			temp=temp*2/(test_length(i,1)-1)/test_length(i,1);
		end
		Loss4=Loss4+temp;
	end
	Loss4=Loss4/n;

	sumLoss=Loss1+Loss2+Loss4;
	sumLoss=sumLoss*2;

elseif strcmp(str_tag,'ProSVM_Prime')
    	[n,m]=size(test_rank);

	Loss1=0;
	for i=1:n
		temp=0;
		for j=1:test_length(i,1)
			if output(i,test_rank(i,j))<output(i,m+1)
				temp=temp+1;
			elseif output(i,test_rank(i,j))==output(i,m+1)
				temp=temp+1/2;
			end
		end
		if test_length(i,1)~=0
			temp=temp/test_length(i,1);
		end
		Loss1=Loss1+temp;
	end
	Loss1=Loss1/n;

	Loss2=0;
	for i=1:n
		temp=0;
		for j=test_length(i,1)+1:m
			if output(i,test_rank(i,j))>output(i,m+1)
				temp=temp+1;
			elseif output(i,test_rank(i,j))==output(i,m+1)
				temp=temp+1/2;
			end
		end
		if (m-test_length(i,1))~=0
			temp=temp/(m-test_length(i,1));
		end
		Loss2=Loss2+temp;
	end
	Loss2=Loss2/n;

	Loss3=0;
	for i=1:n
		temp=0;
		for j=1:test_length(i,1)
			for k=test_length(i,1)+1:m
				if output(i,test_rank(i,j))<output(i,test_rank(i,k))
					temp=temp+1;
				elseif output(i,test_rank(i,j))==output(i,test_rank(i,k))
					temp=temp+1/2;
				end
			end
		end
		if test_length(i,1)~=0 && m-test_length(i,1)~=0
			temp=temp/test_length(i,1)/(m-test_length(i,1));
		end
		Loss3=Loss3+temp;
	end
	Loss3=Loss3/n;

	sumLoss=Loss1+Loss2+Loss3;
elseif strcmp(str_tag,'ProSVM-A_Prime')
    [n,m]=size(test_rank);

	Loss1=0;
	for i=1:n
		temp=0;
		for j=1:test_length(i,1)
			if output(i,test_rank(i,j))<output(i,m+1)
				temp=temp+1;
			elseif output(i,test_rank(i,j))==output(i,m+1)
				temp=temp+1/2;
			end
		end
		if test_length(i,1)~=0
			temp=temp/test_length(i,1);
		end
		Loss1=Loss1+temp;
	end
	Loss1=Loss1/n;

	Loss2=0;
	for i=1:n
		temp=0;
		for j=test_length(i,1)+1:m
			if output(i,test_rank(i,j))>output(i,m+1)
				temp=temp+1;
			elseif output(i,test_rank(i,j))==output(i,m+1)
				temp=temp+1/2;
			end
		end
		if (m-test_length(i,1))~=0
			temp=temp/(m-test_length(i,1));
		end
		Loss2=Loss2+temp;
	end
	Loss2=Loss2/n;

	sumLoss=Loss1+Loss2;
	sumLoss=sumLoss*2;
end


end

