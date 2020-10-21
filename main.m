addpath('functions')
clear;clc

%% experimental setting
path1   = 'data/';
path2   = 'result/';
list_t  = 'australian';
N_round = 10;

pars.S      = 1; %L2-norm of x
pars.lambda = 1; %trade-off parameter
pars.N_a    = 2; %number of positive prototypes
pars.N_b    = 2; %number of negative prototypes
pars.alpha  = 0.001; %learning rate
pars.N_iter = 5000;  %maximum number of iterations



%% SMILE
train_acc = zeros(N_round,1);
test_acc  = zeros(N_round,1);
for i_round = 1:N_round
    fprintf('round = %d\n',i_round)
    name_1 = [path1, list_t,'_', int2str(i_round),'.mat'];
    name_2 = [path2, list_t,'_', int2str(i_round),'_result'];
    [train_acc(i_round),test_acc(i_round)]=fun_SMILE(name_1,name_2,pars);
end;clear i_round name_t1 name_t2

disp([mean(test_acc),std(test_acc)])

