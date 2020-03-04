clear all; clc;
import brml.*

N = 10;
Phixixj = [exp(1) exp(0); exp(0) exp(1)];

fl = 1; tr = 2;

temppot = array([1, 2]);
for i = 0:1:2^N-1
    for j = 0:1:2^N-1
        % convert to binary states
        X_i = zeros(1, 10);
        temp = de2bi(i);
        X_i(1:length(temp)) = temp;
        X_j = zeros(1, 10);
        temp = de2bi(j);
        X_j(1:length(temp)) = temp;
        % count number of edges with their nodes being in the same state
        temppot.table(i+1, j+1) = exp(sum(X_i == X_j) + sum(X_i(1:end-1) == X_i(2:end)) + sum(X_j(1:end-1) == X_j(2:end)));
    end
end
pot{1} = temppot;

temptable = zeros(2^N);
for i = 0:1:2^N-1
    for j = 0:1:2^N-1
        % convert to binary states
        X_i = zeros(1, 10);
        temp = de2bi(i);
        X_i(1:length(temp)) = temp;
        X_j = zeros(1, 10);
        temp = de2bi(j);
        X_j(1:length(temp)) = temp;
        temptable(i+1, j+1) = exp(sum(X_i == X_j) + sum(X_j(1:end-1) == X_j(2:end)));
    end
end

for m = 2:1:9
    temppot = array([m, m+1]);
    temppot.table = temptable;
    pot{m} = temppot;
end

separator = array();

% Absorption Procedure
potstar{1} = pot{1};
for m = 2:1:9
    [temp, potstar{m}] = absorb(potstar{m-1}, separator, pot{m}, m);
end

logZ = log(sum(sum(potstar{9}.table)));
fprintf('The logarithm of Z is: %f\n', logZ);