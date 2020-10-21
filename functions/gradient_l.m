function G_L = gradient_l(X,Y,L,R_a,R_b)

% INPUT
%   X: D*N, where D is the feature dimension, N is the sample size
%   Y: N*1, Y_i= {-1,1};
%   L: U*D, where U is the dimension after linear mapping;
%                   ie.the intrinsic dimension of the metric
%   R_a: U*N_a, prototypes of positive class
%   R_b: U*N_b, prototypes of negative class
%OUTPUT
%   G_L: U*D, gradient of L

[D,N]   = size(X);
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
G_ly  = 2*Y.*G_l;


%gradient wrt positive prototypes
G_la_all = zeros(U,D,N_a);
for i_a = 1:N_a
    rL  = R_a(:,i_a)-L*X;
    G_la_all(:,:,i_a) = (G_ly'.*exp(-vecnorm(rL,2,1).^2).*rL)*X';
end
G_la = sum(G_la_all,3)/N;


%gradient wrt negative prototypes
G_lb_all = zeros(U,D,N_b);
for i_b = 1:N_b
    rL  = L*X - R_b(:,i_b);
    G_lb_all(:,:,i_b) = (G_ly'.*exp(-vecnorm(rL,2,1).^2).*rL)*X';
end
G_lb = sum(G_lb_all,3)/N;


G_L  = G_la+G_lb; 

end
