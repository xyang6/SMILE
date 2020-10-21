function [train_acc,test_acc,train_err,test_err,obj_fun] = fun_SMILE(name1,name2,pars)

load(name1);
S        = pars.S;
lambda   = pars.lambda;
N_a      = pars.N_a;
N_b      = pars.N_b;
alpha    = pars.alpha;
max_iter = pars.N_iter;

%L2-normalization and label conversion 
X = S*data_train_norm';
Y = (label_train-0.5)*2;
X_test = S*data_test_norm';
Y_test = (label_test-0.5)*2;

%initialize representative instances
D   = size(X,1);
X_a = X(:,(Y==1));
X_b = X(:,(Y==-1));
L   = eye(D);
rng(1)
[~,R_a] = kmeans(X_a',N_a); R_a = R_a';
[~,R_b] = kmeans(X_b',N_b); R_b = R_b';

obj_fun   = zeros(max_iter,1);
train_err = zeros(max_iter,1);
test_err  = zeros(max_iter,1);

for iter = 1:max_iter
    
    %update prototypes
    [G_ra,G_rb] = gradient_r(X,Y,L,R_a,R_b);
    R_a = (1-alpha*lambda)*R_a-alpha*G_ra;
    R_b = (1-alpha*lambda)*R_b-alpha*G_rb;
    
    %update projection matrix
    G_L = gradient_l(X,Y,L,R_a,R_b);
    L   = (1-alpha*lambda)*L-alpha*G_L;
    
    %calculate objective function, training and test error
    H = classifier_ie(X,L,R_a,R_b);
    obj_fun(iter)    = mean(loss_ie(H,Y)) + lambda/2*...
                      (sum(vecnorm(R_a,2,1)+vecnorm(R_b,2,1))+norm(L,'fro'));
    train_err(iter) = mean(H.*Y<=0);
    H_test = classifier_ie(X_test,L,R_a,R_b);
    test_err(iter)  = mean(H_test.*Y_test<=0);

end


[train_acc,idx] = min(train_err);
train_acc = 1-train_acc;
test_acc  = 1-test_err(idx,1);

save(name2,'train_acc','test_acc','train_err','test_err','obj_fun');
end

