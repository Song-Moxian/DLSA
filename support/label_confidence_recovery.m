function [projection_matrix, lower_train_data,Y] = label_confidence_recovery(train_data, train_p_target, T, mu, dim_para, k)

D = size(train_data, 2);          
M = size(train_data, 1);               
Q = size(train_p_target, 1);        

X = zscore(train_data);
X = X';

LabelSet = zeros(Q, M + 6);    
InsSet = zeros(M, Q + 6);       

train_p_target = train_p_target';
for i = 1:M
    for j = 1:Q
        if train_p_target(i, j) == 1
            LabelSet(j, 1) = LabelSet(j, 1) + 1;
            t = LabelSet(j, 1);
            LabelSet(j, t + 1) = i;
            
            InsSet(i, 1) = InsSet(i, 1) + 1;
            t = InsSet(i, 1);
            InsSet(i, t + 1) = j;
        end
    end
end


Y = zeros(M, Q);
for i = 1:M
    tot = InsSet(i, 1);
    if tot ~= 0
        for j = 1:tot
            la = InsSet(i, j + 1);
            Y(i, la) = 1 / tot;
        end
    end
    
    Y(i, :) = Y(i, :) / norm(Y(i, :), 1);
    
end


for t = 1:T
   
    L = Y * Y';
    
    [P, lambda] = HSIC_Solver(X, L, mu, dim_para);
    lower_train_data = P' * X;
    
    [Y_new] = YUpdate_New(lower_train_data, Y, k, InsSet);
    Y = Y_new;
end

projection_matrix = P;

end
