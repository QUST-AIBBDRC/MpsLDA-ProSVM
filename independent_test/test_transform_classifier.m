function test_transform_classifier(trainfile, testfile, resultfile, transform_type, transform_parameter, classifier_type, classifier_parameter)
% This function evaluates a pair of training and testing data sets with a
% specific transform and a fixed classifier.
% Editor: Jianhua Xu (xujianhua@njnu.edu.cn)
% Date: May, 2015
%INPUT:
%    trainfile --> the training file name to save feature and label matrices
%    testfile --> the testing file name to save feature and label matrices
%    resultfile --> the result file name to save the results
%    transform_type --> the transform type (0--12)
%    classifier_type --> the classifier type (1 or 2)
%    transform_parameter --> the parameter settings for used transform
%    classifier_parameter --> the classifier settings for used classifier
%
%-------------------------------------------------------------------------
% read training data
load(trainfile);
train_data = data;
train_label = label;
[N, D] = size(train_data);
mean_train = mean(train_data);
train_CX = train_data - repmat(mean_train,N,1); %centralizing training features
mean_label = mean(train_label);
train_CY = train_label - repmat(mean_label, N, 1);

% read testing data
load(testfile);
test_data = data;
test_label = label;
[N, D] = size(test_data);
test_CX = test_data - repmat(mean_train, N, 1); % centralizing testing features 

result_all = [];
[N,Q]=size(train_label);

disp('Current setting');
transform_parameter
classifier_parameter

t0 = clock;
currentTrainData = train_CX;
currentTestData = test_CX;
    
if (transform_type >0)
    % execute a FE method
    if(transform_type >=1 & transform_type <=6) %PCA, CCA, MLSI, MDDMp/f, MVMD which need centered training labels
            [P] = execute_transform(currentTrainData, train_CY, transform_type, transform_parameter); 
    else % LDA type transforms which do not need centered training labels
            [P] = execute_transform(currentTrainData, train_label, transform_type, transform_parameter);
    end
    
    % convert training and testing features
    currentTrainData = currentTrainData * P;        
    currentTestData  = currentTestData * P;

end
time = etime(clock,t0);

% train and test a fixed classifier
[RK_6M, IB_6M, MM_8M] = execute_classifier(currentTrainData,train_label,currentTestData,......
                            test_label,classifier_type, classifier_parameter);

if (transform_type >0) 
    dim = size(P, 2);
else
    dim = D;
end

disp(strcat('Reduced Dimensions         =',num2str(dim)));
[result] = ShowResult(RK_6M,IB_6M,MM_8M); 
disp(strcat('Time         =',num2str(time)));
[result_all] = [result_all, [dim;result;time]];
    
clear currentTrainData;
clear currentTestData;

save(resultfile,'result_all','-ASCII');

end
