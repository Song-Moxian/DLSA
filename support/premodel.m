function test_outputs = premodel(train_data,train_p_target,test_data,k,ker,par,Maxiter,lambda,mu,gama)

if nargin < 10
	gama = 0.05;
end
if nargin < 9
	mu = 1;
end
if nargin < 8
	lambda = 1;
end
if nargin < 7
	Maxiter = 10;
end
if nargin < 6
	par = 1*mean(pdist(train_data));
end
if nargin < 5
	ker = 'rbf';
end
if nargin < 4
	k = 10;
end
if nargin < 3
	error('Not enough input parameters!');
end
y=build_label_manifold(train_data,train_p_target,k);
fprintf('Update parameters...\n')
[train_outputs, test_outputs] = MulRegression(train_data, y, test_data, gama, par, ker);
for i = 1:Maxiter
	fprintf('The %d-th iteration\n',i);
	W = obtain_W(train_data,y,k,lambda,mu);
	fprintf('Generate the labeling confidence...\n');
	y = UpdateY(W,train_p_target,train_outputs,mu);
	fprintf('Update parameters...\n')
	[train_outputs, test_outputs] = MulRegression(train_data, y, test_data, gama, par, ker);
end

end