function [G_ra,G_rb] = gradient_r(X,Y,L,R_a,R_b)

% INPUT
%   X: D*N, where D is the feature dimension, N is the sample size
%   Y: N*1, Y_i= {-1,1};
%   L: U*D, where U is the dimension after linear mapping;
%                   ie.the intrinsic dimension of the metric
%   R_a: U*N_a, prototypes of positive class
%   R_b: U*N_b, prototypes of negative class
% OUTPUT
%	G_ra: U*N_a, gradient of R_a
%	G_rb: U*N_b, gradient of R_b

[U,N_a] = size(R_a);
N_b     = size(R_b,2);

H  = classifier_ie(X,L,R_a,R_b);%output of the classifier
Hy = Y.*H; %input to the loss function

%derivative of loss
Idx_1 = (Hy<=0);
Idx_3 = (Hy>2);
Idx_2 = ~(Idx_1 | Idx_3);
G_l   = zeros(size(Hy));
G_l(Idx_1) = -1;
G_l(Idx_2) = (Hy(Idx_2)-2)/2;
G_l(Idx_3) = 0;
G_ly = 2*Y.*G_l;

%calculate gradient of positive prototypes
G_ra = zeros(U,N_a);
for i_a = 1:N_a
    rL  = L*X-R_a(:,i_a);
    G_ra(:,i_a) = mean(G_ly'.*exp(-vecnorm(rL,2,1).^2).*rL,2);
end

%calculate gradient of negative prototypes
G_rb = zeros(U,N_b);
for i_b = 1:N_b
    rL  = R_b(:,i_b)-L*X;
    G_rb(:,i_b) = mean(G_ly'.*exp(-vecnorm(rL,2,1).^2).*rL,2);
end

end
