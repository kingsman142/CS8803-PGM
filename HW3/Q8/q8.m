import brml.*
load('pMRF.mat')

phi = setpotclass(phi, 'array');

% part A -- loopy belief propagation
numIterations = 25;
mess = rand(10, 2); % 10 messages: m21 m31 m41 m12 m32 m23 m43 m14 m34
for i = 1:numIterations
    mess(1, :) = normalize([sum(phi(1).table(1, :) .* mess(5, :)) sum(phi(1).table(2, :) .* mess(5, :))], 'norm', 1); % m21
    mess(2, :) = normalize([sum(phi(5).table(1, :) .* mess(7, :) .* mess(8, :)) sum(phi(5).table(2, :) .* mess(7, :) .* mess(8, :))], 'norm', 1); % m31
    mess(3, :) = normalize([sum(phi(4).table(:, 1).' .* mess(10, :)) sum(phi(4).table(:, 2).' .* mess(10, :))], 'norm', 1); % m41
    mess(4, :) = normalize([sum(phi(1).table(:, 1).' .* mess(2, :) .* mess(3, :)) sum(phi(1).table(:, 2).' .* mess(2, :) .* mess(3, :))], 'norm', 1); % m12
    mess(5, :) = normalize([sum(phi(2).table(1, :) .* mess(6, :) .* mess(8, :)) sum(phi(2).table(2, :) .* mess(6, :) .* mess(8, :))], 'norm', 1); % m32
    mess(6, :) = normalize([sum(phi(5).table(:, 1).' .* mess(1, :) .* mess(3, :)) sum(phi(5).table(:, 2).' .* mess(1, :) .* mess(3, :))], 'norm', 1); % m13
    mess(7, :) = normalize([sum(phi(2).table(:, 1).' .* mess(4, :)) sum(phi(2).table(:, 2).' .* mess(4, :))], 'norm', 1); % m23
    mess(8, :) = normalize([sum(phi(3).table(1, :) .* mess(9, :)) sum(phi(3).table(2, :) .* mess(9, :))], 'norm', 1); % m43
    mess(9, :) = normalize([sum(phi(4).table(1, :) .* mess(1, :) .* mess(2, :)) sum(phi(4).table(2, :) .* mess(1, :) .* mess(2, :))], 'norm', 1); % m14
    mess(10, :) = normalize([sum(phi(3).table(:, 1).' .* mess(6, :) .* mess(7, :)) sum(phi(3).table(:, 2).' .* mess(6, :) .* mess(7, :))], 'norm', 1); % m34
end
loopyMarginals = zeros(4, 2);
loopyMarginals(1, :) = normalize(mess(1, :) .* mess(2, :) .* mess(3, :), 'norm', 1); % marginal of node 1
loopyMarginals(2, :) = normalize(mess(4, :) .* mess(5, :), 'norm', 1); % marginal of node 2
loopyMarginals(3, :) = normalize(mess(6, :) .* mess(7, :) .* mess(8, :), 'norm', 1); % marginal of node 3 
loopyMarginals(4, :) = normalize(mess(9, :) .* mess(10, :), 'norm', 1); % marginal of node 4
disp(loopyMarginals.');

% part B -- variational mean-field equations
numIterations = 50;
oneMarginal = rand(2, 1); % 2x1 distribution
twoMarginal = rand(2, 1); % 2x1 distribution
threeMarginal = rand(2, 1); % 2x1 distribution
fourMarginal = rand(2, 1); % 2x1 distribution
tmp = zeros(2, 2, 2, 2);
for i = 1:2 % x_1
    for j = 1:2 % x_2
        for k = 1:2 % x_3
            for m = 1:2 % x_4
                tmp(i, j, k, m) = phi(1).table(i, j) * phi(2).table(j, k) * phi(3).table(k, m) * phi(4).table(m, i) * phi(5).table(i, k);
            end
        end
    end
end
for i = 0:numIterations    
    oneMarginalTmp = zeros(2, 2, 2, 2);
    twoMarginalTmp = zeros(2, 2, 2, 2);
    threeMarginalTmp = zeros(2, 2, 2, 2);
    fourMarginalTmp = zeros(2, 2, 2, 2);
    % multiply potentials
    for j = 1:2 % x_1
        for k = 1:2 % x_2
            for m = 1:2 % x_3
                for n = 1:2 % x_4
                    oneMarginalTmp(j, k, m, n) = log(tmp(j, k, m, n)) * twoMarginal(k) * threeMarginal(m) * fourMarginal(n);
                    twoMarginalTmp(j, k, m, n) = log(tmp(j, k, m, n)) * oneMarginal(j) * threeMarginal(m) * fourMarginal(n);
                    threeMarginalTmp(j, k, m, n) = log(tmp(j, k, m, n)) * oneMarginal(j) * twoMarginal(k) * fourMarginal(n);
                    fourMarginalTmp(j, k, m, n) = log(tmp(j, k, m, n)) * oneMarginal(j) * twoMarginal(k) * threeMarginal(m);
                end
            end
        end
    end
    
    % sum out variables that aren't associated with this node's marginal
    oneMarginalTmp = squeeze(sum(oneMarginalTmp, [2 3 4]));
    twoMarginalTmp = squeeze(sum(twoMarginalTmp, [1 3 4]));
    threeMarginalTmp = squeeze(sum(threeMarginalTmp, [1 2 4]));
    fourMarginalTmp = squeeze(sum(fourMarginalTmp, [1 2 3]));
    
    
    % exponentiate potentials to return them to decimals
    oneMarginalTmp = exp(oneMarginalTmp);
    twoMarginalTmp = exp(twoMarginalTmp);
    threeMarginalTmp = exp(threeMarginalTmp);
    fourMarginalTmp = exp(fourMarginalTmp);
    
    % normalize distributions
    oneMarginal = oneMarginalTmp ./ sum(oneMarginalTmp, 'all');
    twoMarginal = twoMarginalTmp ./ sum(twoMarginalTmp, 'all');
    threeMarginal = threeMarginalTmp ./ sum(threeMarginalTmp, 'all');
    fourMarginal = fourMarginalTmp ./ sum(fourMarginalTmp, 'all');
end
mfMarginals = [oneMarginal twoMarginal.' threeMarginal fourMarginal];
disp(mfMarginals);

% part C
joint = tmp ./ sum(tmp, 'all');
exactMarginals = [squeeze(sum(joint, [2 3 4])) squeeze(sum(joint, [1 3 4])).' squeeze(sum(joint, [1 2 4])) squeeze(sum(joint, [1 2 3]))];
disp(exactMarginals);

% part D
calculateMED = @(predMarginals, exactMarginals) sum(abs(predMarginals.' - exactMarginals), 'all') / 8.0; % compute the MED (Mean Expected Deviation)
loopyMED = calculateMED(loopyMarginals, exactMarginals); % compute MED for loopy marginals
mfMED = calculateMED(mfMarginals.', exactMarginals); % compute MED for mean-field marginals
fprintf("Loopy MED: %f\n", loopyMED);
fprintf("Mean field MED: %f\n", mfMED);