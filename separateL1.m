function [ w, gamm, obj, misclass ] = separateL1( dataMat, features, tau)
%     dataMat is the matrix of training data returned from the code
%     features is an integer vector containing some subset of 
%     {2, 3, 4, . . . , 31} indicating which features are to be used in the 
%     classification
%     mu is the value of µ from (1)
%     `m` number of malignant examples
%     `k` number of benign examples, 
%     `n` length of one feature vector
%     `w` `gamm` separating plane is given by w * x + gamm
%     `obj` optimal objective value
%     `misclass` number of misclassifications in dataMat

X = dataMat(:, features + 1);
labels = dataMat(:, 1);

M = X(labels == 1, :);  % malignant features
B = X(labels == 0, :);  % benign features
n = length(features);  % size of a feature vector
m = size(M, 1);  % no. of malignant examples
k = size(B, 1);  % no. of benign examples
quiet = true;

if ~exist('quiet', 'var'); quiet = true; end;
cvx_quiet(quiet);

cvx_begin
    variable w(n)
    variable gamm 
    variable y(m)
    variable z(k);
    minimize (1/m * ones(1, m) * y + 1/k * ones(1, k) * z + tau*(norm(w,1)))
    subject to
        M * w - gamm * ones(m, 1) + y >= ones(m, 1);
        -B * w + gamm * ones(k, 1) + z >= ones(k, 1);
        y >= 0;
        z >= 0;    
cvx_end

obj = cvx_optval;

predict = X*w - gamm > 0;
misclass = sum(predict~=labels);

end