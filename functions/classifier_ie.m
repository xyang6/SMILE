function H = classifier_ie(X,L,R_a,R_b)

%Classifier of SMILE
%h = sum(exp(-||Lx-r_pos||^2) - sum(exp(-||Lx-r_neg||^2)
%INPUT
%   X: D*N, where D is the feature dimension, N is the sample size
%   L: U*D, where U is the dimension after linear mapping;
%                   ie.the intrinsic dimension of the metric
%   R_a: U*N_a, prototypes of positive class
%   R_b: U*N_b, prototypes of negative class
%OUTPUT
%   H: N*1, the classification value
%      Larger H indicates higher confidence in belonging to the positive class

X_u  = L*X;
D_xa = pdist2(X_u',R_a'); 
D_xb = pdist2(X_u',R_b'); 
A    = sum(exp(-D_xa.^2),2); 
B    = sum(exp(-D_xb.^2),2);
H    = A - B;
end

