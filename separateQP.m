function [ w, gamm, obj, misclass ] = separateQP( dataMat, features, mu)
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

M = X(labels == 1, :);  % malignant 
B = X(labels == 0, :);  % benign 
n = length(features);  % featureSize
m = size(M, 1);  % malignant numbers
k = size(B, 1);  % beinign numbers
quiet = true;

if ~exist('quiet', 'var'); quiet = true; end;
cvx_quiet(quiet);

cvx_begin
    variable w(n)
    variable gamm 
    variable y(m)
    variable z(k);
    minimize (1/m * ones(1, m) * y + 1/k * ones(1, k) * z + mu / 2 * w' * w)
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