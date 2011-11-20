function [alpha beta] = m_step_linear_regression(X, Y, m, s)
% M-step of EM algorithm
%
% @param X      : design matrix for regression (n x d, includes intercept)
% @param Y      : target vector
% @param m      : mean of weight vector
% @param s      : covariance matrix of weight vector
%
% @return alpha : weight precision = 1/(weight variance)
% @return beta  : noise precision = 1 / (noise variance)

M = size(m,1);

alpha_inverse = m'*m + trace(s);

alpha_inverse = alpha_inverse/M;

alpha = inv(alpha_inverse);

[n,d] = size(X);

first = (Y - X*m);
first3 = first'*first;
first4 = trace(X'*X*s) + first3;

beta = n/first4;

