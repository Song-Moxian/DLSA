function Accuracy = DLSA(data_path,K,dim_para,T,mu,gama,beta,alpha,Maxiter)
% Load the file containing the necessary inputs for calling the EUPAL function

load(data_path);

labeled_data = partialData;
labeled_target = partialTarget;
unlabeled_data = unlabeledData;
test_data = testData;
test_target = testTarget;
class_num = size(labeled_target, 2);

[pl_target] = label_set_assignment(labeled_data, labeled_target, unlabeled_data, K);

%all training examples
train_data_all = [labeled_data;unlabeled_data];
train_taget_all = [labeled_target;pl_target];

% Reliable label confidence recovery
[~, ~,label_generation] = label_confidence_recovery(train_data_all, train_taget_all', T, mu, dim_para, K);

label_generation(label_generation>(1/class_num))=1;
label_generation(label_generation~=1)=0;

par = 1*mean(pdist(train_data_all)); %Parameters of kernel function
%training and testing
ker  = 'rbf';
test_outputs = premodel(train_data_all,label_generation,test_data,K,ker,par,Maxiter,gama,beta,alpha);
accuracy = CalAccuracy(test_outputs, test_target);
fprintf('The accuracy of DLSA is: %f \n',accuracy);

Accuracy = accuracy;