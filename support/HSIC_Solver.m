function [P, lambda] = HSIC_Solver(X, L, mu, dim_para)
%identifies the projected matrix by solving a tailored generalized eigenvalue problem

[D N] = size(X);
tmpL = L - repmat(mean(L, 1), N, 1);
HLH = tmpL - repmat(mean(tmpL, 2), 1, N);

S = X * HLH * X';

B = mu * X * X' + (1 - mu) * eye(D);

clear X L;

[tmp_P, tmp_lambda] = eig(S, B);
tmp_P = real(tmp_P);
tmp_lambda = real(diag(tmp_lambda));
[lambda, order] = sort(tmp_lambda, 'descend');
P = tmp_P(:,order);

proper_dim = getProperDim(lambda, dim_para);
P = P(:,1:proper_dim);
lambda = lambda(1:proper_dim);

end

