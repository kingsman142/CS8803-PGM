import brml.*
load('p.mat');

p = setpotclass(p, 'array');
qXY = array([1 2], condp(rand(3, 3), [1 2])); % 3x3 distribution
qZ = array(3, condp(rand(3, 1))); % 3x1 distribution
calcKLDivergence = @(p, q) sum(p .* log(p ./ q) - p + q, 'all'); % anonymous function to calculate KL-divergence

numIterations = 1000;
for i = 0:numIterations
    tmp = exppot(sumpot(multpots({logpot(p), qXY}), [1 2]));
    qZ = condpot(tmp);
    
    tmp = exppot(sumpot(multpots({logpot(p), qZ}), 3));
    qXY = condpot(tmp);
end

q = condpot(multpots([qXY qZ])); % multiply two distributions to get final Q

disp(calcKLDivergence(q.table, p.table)); % compute the element-wise kl divergence