function Outputs = UpdateY(W, train_p_target,train_outputs,mu)
%Update label confidence Y

[p,q]=size(train_p_target);

options = optimoptions('quadprog',...
'Display', 'iter','Algorithm','interior-point-convex' );
WT = W';
sum(WT);
fprintf('Obtain Hessian matrix...\n');
tic

T = 2*(eye(p)-W)'*(eye(p)-W)+2/mu*eye(p);

toc
T1 = repmat({T},1,q);
M = spblkdiag(T1{:});

lb=sparse(p*q,1);
ub=reshape(train_p_target,p*q,1);
II = sparse(eye(p));
A = repmat(II,1,q);
b=ones(p,1);

fprintf('quadprog...\n')
f = reshape(train_outputs, p*q, 1);
Outputs= quadprog(M, -2*(1/mu)*f, [],[], A, b, lb, ub,[], options);
Outputs=reshape(Outputs,p,q);
end