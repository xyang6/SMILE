function F = loss_ie(H,Y)
%INPUT
%   H: N*1, output from the classifier
%   Y: N*1, ground truth label -1,1
%OUTPUT
%   L: N*1, Lipschitz loss

Hy    =  H.*Y;
Idx_1 = (Hy<=0);
Idx_3 = (Hy>2);
Idx_2 = ~(Idx_1 | Idx_3);
F     = zeros(size(Hy));
F(Idx_1) = 1-Hy(Idx_1);
F(Idx_2) = (Hy(Idx_2)-2).^2 /4;
F(Idx_3) = 0;


end

