function [eps_out, region] = saddlepoint_approximation(n, R, a, b, c, d)
% Function that computes the saddlepoint approximation according to the
% expressions given in the paper "URLLC with Massive MIMO: Analysis and Design at Finite Blocklength"
% by J. Östman, A. Lancho, G. Durisi and L. Sanguinetti. 
% INPUTS:
% n = blocklength
% R = rate in bits
% a = betaA;
% b = betaB;
% c = nu;
% d = preterm;
%
% OUTPUS:
% eps_out = error probability of the given link
% region = region in which the rate lies wrt the mutual information and the
% critical rate. Only for debug purposes. 

R=R*log(2); %rate in nats

% Useful definitions for compactness:
xi      = 2*a*b*(1-c);
lambda  = sqrt((a-b)^2+4*a*b*(1-c));
gamma   = a - b;

%Region of convergence:
zeta_low = -(gamma + lambda)/xi; % % equ.10 of paper
zeta_high = (lambda - gamma)/xi; % equ.11 of paper

% CGF and derivatives as defined in the paper:
kappa0 = @(t) -t*d - log(1 + (b-a)*t - a*b*(1-c)*t.^2); % equ.16 of paper，CGF of info density
kappa1 = @(t) -(-2*a*b*(1-c)*t +b-a)./(-a*b*(1-c).*t.^2+t*(b-a)+1) - d; % equ.17 of paper，first derivative of CGF of info density
kappa2 = @(t) (b-a-2*a*b*(1-c)*t).^2./(1+(b-a)*t-a*b*(1-c)*t.^2).^2 + (2*a*b*(1-c))./(1+(b-a)*t-a*b*(1-c).*t.^2); % equ.18 of paper,second derivative of CGF of info density

% Some useful functions:
Rcr = -kappa1(1); %critical rate % equ.19 of paper
Is = -kappa1(0); %generalized MI

% Further definitions of functions according to the paper:
Psi  = @(u,k2) exp(n*u^2*k2/2 + log(qfunc(u*sqrt(n*k2)))); % equ.21 of paper
PsiTilde = @(x,y)  exp(n*x*(Rcr-R+kappa2(1)/2) + log(qfunc(x*sqrt(n*kappa2(1)) + y*n*(Rcr - R)/sqrt(n*kappa2(1))))); % equ.23 of paper

%-------------------------------------------- 求鞍点：使 -kappa'(zeta)=R的zeta
% The following piece of code is used to find the value of zeta satisfiying
% the stationary equation -kappa'(zeta)=R. It is numerically delicated, so we used the
% following lines to minimize potential numerical issues. 
%find start value for stationary equation

small_tmp = eps;%通过 eps 定义一个很小的值作为步长。
zeta_candidate_low = zeta_low+small_tmp;% 从 zeta_low 加上步长开始搜索合适的起始点 zeta_candidate_low。
% 将步长扩大两倍，重新计算 zeta_candidate_low，
% 直到 -kappa'(zeta_candidate_low)大于等于Is 并且不等于无穷大。
while kappa1(zeta_candidate_low) > kappa1(0) || abs(kappa1(zeta_candidate_low))== inf%because of numerical issues.
    small_tmp= 2*small_tmp;
    zeta_candidate_low = zeta_low+small_tmp;
end
% 继续增加 zeta_candidate_low(+=1) 的值，
% 直到 -kappa'(zeta_candidate_low) 小于等于 R 或者 超过 zeta_high。
while -kappa1(zeta_candidate_low) > R && zeta_candidate_low < zeta_high%search while we are still above R
    zeta_candidate_low = zeta_candidate_low + 1;
end
% 将 zeta_candidate_low 减去 2，或者与 zeta_low 取最大值
%以确保 zeta_candidate_low 在 zeta_low 和 zeta_high 之间。
zeta_candidate_low = max(zeta_candidate_low-2, zeta_low); 

%find end and start value for stationary equation -kappa'(zeta)=R
small_tmp = eps;
zeta_candidate_high = zeta_high-small_tmp; % 从 zeta_high 减去步长开始搜索合适的终止点 zeta_candidate_high。
% 将步长扩大两倍，重新计算 zeta_candidate_high
%直到 -kappa'(zeta_candidate_high)小于等于 Is 并且不等于无穷大。
while kappa1(zeta_candidate_high) == inf ||  kappa1(zeta_candidate_high) < kappa1(0) %because of numerical issues.
    small_tmp= 2*small_tmp;
    zeta_candidate_high = zeta_high-small_tmp ;
end
% 继续减少 zeta_candidate_high（-=1） 的值
% 直到 -kappa'(zeta_candidate_high) 大于等于 R 或者 zeta_candidate_high 的值小于 zeta_low。
while -kappa1(zeta_candidate_high) < R && zeta_candidate_high > zeta_low%search while we are still below R
    zeta_candidate_high = zeta_candidate_high - 1;
end
% 将 zeta_candidate_high 加上 2，或者与 zeta_high 取最小值
% 以确保 zeta_candidate_high 在 zeta_low 和 zeta_high 之间。
zeta_candidate_high = min(zeta_candidate_high+2, zeta_high);

% 根据上述起始点和终止点之间的间隔，生成一个可能的 zeta 值的向量 interval。
if abs(zeta_candidate_high -zeta_high) < 1 ||  abs(zeta_candidate_low - zeta_low) < 1 % we need to look very close to the ROC border
    interval = linspace(zeta_candidate_low, zeta_candidate_high, 1e4) ;
else
    interval =linspace(zeta_candidate_low, zeta_candidate_high, 1e3) ;
end
zeta_candidates = interval;
zeta_candidates(1)=[];
zeta_candidates(end)=[];
%--------------------------------------------

k1 = kappa1(zeta_candidates); % Compute kappa1 for possible values of zeta

%make sure we have no infs because we got too close to RoC endpoints:
[~,iduniq] = unique(k1);
k1 = k1(iduniq);
zeta_candidates= zeta_candidates(iduniq);
idinf = find(isinf(k1));
k1(idinf) = [];
zeta_candidates(idinf)= [];

zeta = interp1(k1, zeta_candidates, -R); % Finding zeta s.t. kappa'(zeta) = threshold (-rate).

% Computation of the saddlepoint approximation equations depending on the
% value of zeta:
if zeta > 1 % R < Rcr
    k0 = kappa0(1);
    eps_out = exp(n*(k0+R)) *(PsiTilde(1,1) + PsiTilde(0,-1)); % equ.22 of paper
    region = 3; % Only used for debugging purposes
elseif zeta < 0 %R > Is
    k0 = kappa0(zeta);
    k2 = kappa2(zeta);
    error_exp = exp(n*(k0+zeta*R)); % Error exponent.
    eps_out = 1 - error_exp * (Psi(-zeta,k2) - Psi(1-zeta, k2)); % equ.24 of paper
    region = 1; % Only used for debugging purposes
else %Rcr < R < Is
    k0 = kappa0(zeta);
    k2 = kappa2(zeta); 
    error_exp = exp(n*(k0 + zeta*R)); % Error exponent.
    eps_out = error_exp * (Psi(zeta,k2) + Psi(1-zeta, k2)); % equ.20 of paper
    region = 2; % Only used for debugging purposes
end
if isnan(eps_out)
    keyboard; % Stopping here would imply numerical issues when finding the zeta values. 
end
if ~isnan(eps_out)
    eps_out = min(1, eps_out); % avoid values larger than 1. 
end

end
