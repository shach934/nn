% the matlab version of neural network
clear, clc, close,

train_ims = loadMNISTImages('train-images.idx3-ubyte');
train_lbs = loadMNISTLabels('train-labels.idx1-ubyte');

test_ims = loadMNISTImages('t10k-images.idx3-ubyte');
test_lbs = loadMNISTLabels('t10k-labels.idx1-ubyte');

[L, m_train] = size(train_ims);
m_test  = size(test_ims,  2);

train_labels = zeros(10, m_train);

row_indx = train_lbs' + 1;
col_indx = 1:m_train;
indx = sub2ind(size(train_labels), row_indx, col_indx);
train_labels(indx) = 1;

net_struc = [20, 30, 20, 10];
alpha = 0.001;
iter = 100;
[W, b] = neural_network(train_ims, train_labels, alpha, iter, net_struc);
% accu = predict(test_ims, test_lbs, W, b);


function [W, b] = neural_network(input, label, alpha, iter, net_struc)

[W, b] = random_init(input, net_struc);
epoc = zeros(iter, 1);
for it = 1:iter
   [J, y, cache] = forward_prog(input, label, net_struc, W, b); 
   [dW, db] = backward_prog(input, y, label, cache);
   for i = 1:length(net_struc)
      W{i} = W{i} - alpha * dW{i};
      b{i} = b{i} - alpha * db{i};
   end
   epoc(it) = J;
   plot(1:it, epoc(1:it), '-o');
   drawnow;
   hold on;
end

end


function [J, y, cache] = forward_prog(input, label, net_struc, W, b) 
% no broadcast, so it need to be initialized to be the same size of the W
% cache W_i, a_i
m = size(input,2);
z = input;
L = length(net_struc);
cache = cell(L, 2);
for i = 1:L - 1
    z = W{i} * z + repmat(b{i}, 1, m);
    cache{i, 1} = W{i}; % cache the W and z, to be used in the back propagation.
    cache{i, 2} = z;
    z = ReLU(z);
end
z = W{L} * z + repmat(b{L}, 1, m);
y = softmax(z);
cache{L, 1} = W{L};
cache{L, 2} = y;

% softmax cross entropy -y_i * ln(a_i) 
% the derivative is also the label - y;
J = - sum(sum(log(label .* y))) / m ;
end

function [dW, db] = backward_prog(input, y, label, cache)

L = length(cache);
dZ = y - label;
dW = cell(L, 1);
db = cell(L, 1);
for i = L : -1 : 2
    dW{i} = dZ * cache{i - 1, 2}';
    db{i} = sum(dZ, 2);
    dZ = cache{i, 1}' * dZ .* (cache{i-1, 2} > 0);
end
dW{1} = dZ * input';
db{1} = sum(dZ, 2);
end

function [W, b] = random_init(input, net_struc)

rng(0);  % seed the random number
m = size(input, 1);
W = cell(length(net_struc), 1);
b = cell(length(net_struc), 1);
net_struc = [m, net_struc];
for i = 2:length(net_struc)
   W{i-1} = randn(net_struc(i), net_struc(i-1));
   b{i-1} = zeros(net_struc(i), 1);    % b is initialized a 0
end

end

function Z = ReLU(Z)
Z(Z<0) = 0;
end

function a = softmax(Z)

% softmax is very easily go over float64 and become NaNs, 
% according to https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
% shift it close to zero to avoid this situation.

m = size(Z,1);
% Z = Z - repmat(max(Z), m, 1);
Z = exp(Z);
summ = sum(Z);
a = Z ./ repmat(summ, m, 1);
end