function [m s] = e_step_linear_regression(X, Y, alpha, beta)
% E-step of EM algorithm
%
% @param X      : design matrix for regression (n x d, includes intercept)
% @param Y      : target vector
% @param alpha  : weight precision = 1/(weight variance)
% @param beta   : noise precision = 1 / (noise variance)
%
% @return m     : posterior mean of weight vector
% @return s     : posterior covariance matrix of weight vector


% size of design matrix
[n,d] = size(X);


s_inverse = zeros(d,d);

s_inverse = eye(d,d).*alpha + (X'*X).*beta;

m = (s_inverse\X'*Y).*beta;

s = inv(s_inverse);


