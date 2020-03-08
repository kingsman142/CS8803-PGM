import brml.*

edgePotentials = [exp(1) exp(0); exp(0) exp(1)];
dim = 10;

temppot = array([1, 2]);
for i = 0:(2^dim-1)
    for j = 0:(2^dim-1)
        nodeI = zeros(1, 10);
        temp = de2bi(i);
        nodeI(1:length(temp)) = temp;
        nodeJ = zeros(1, 10);
        temp = de2bi(j);
        nodeJ(1:length(temp)) = temp;
        temppot.table(i+1, j+1) = exp(sum(nodeI == nodeJ) + sum(nodeI(1:end-1) == nodeI(2:end)) + sum(nodeJ(1:end-1) == nodeJ(2:end)));
    end
end
pot{1} = temppot;

temptable = zeros(2^dim);
for i = 0:(2^dim-1)
    for j = 0:(2^dim-1)
        nodeI = zeros(1, 10);
        temp = de2bi(i);
        nodeI(1:length(temp)) = temp;
        nodeJ = zeros(1, 10);
        temp = de2bi(j);
        nodeJ(1:length(temp)) = temp;
        temptable(i+1, j+1) = exp(sum(nodeI == nodeJ) + sum(nodeJ(1:end-1) == nodeJ(2:end)));
    end
end

for m = 2:9
    temppot = array([m, m+1]);
    temppot.table = temptable;
    pot{m} = temppot;
end

separator = array();

potstar{1} = pot{1};
for m = 2:1:9
    [temp, potstar{m}] = absorb(potstar{m-1}, separator, pot{m}, m);
end

logZ = log(sum(sum(potstar{9}.table)));
fprintf('%f\n', logZ);