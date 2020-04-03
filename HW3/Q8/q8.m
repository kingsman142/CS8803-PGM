import brml.*
load('pMRF.mat')

phi = setpotclass(phi, 'array');
%phi = setpotclass(phi, 'potential');

% part A
numIterations = 1000;
fg = FactorGraph(phi);
nmess = full(sum(fg(:) ~= 0));
messlidx = find(fg);
for i = 1:numIterations
    r = 1:nmess;
    for j = 1:nmess
        fg(messlidx(j)) = r(j);
        k(r(j)) = j;
    end
    if i > 1
        [marg mess(k)] = sumprodFG(phi, fg, mess(k));
        mess = condpot(mess);
        marg = condpot(marg);
        if isa(marg, 'brml.array') || isa(marg, 'brml.logarray')
            margtable = horzcat(cellfun(@table, marg, 'UniformOutput', false));
            oldmargtable = horzcat(cellfun(@table, oldmarg, 'UniformOutput', false));
            if mean(abs(margtable - oldmargtable)) < 1e-5; break; end
        end
    else
        [marg mess] = sumprodFG(phi, fg); mess(k) = mess;
        mess = condpot(mess);
        marg = condpot(marg);
    end
    oldmarg = marg;
end
mess = mess(k);
[marg, mess, A] = LoopyBP(jtpot);

% part B


% part C
joint = condpot(multpots([phi(1), phi(2), phi(3), phi(4)]));
exactMarginals = [sumpot(joint, [2 3 4]).table sumpot(joint, [1 3 4]).table sumpot(joint, [1 2 4]).table sumpot(joint, [1 2 3]).table];
disp(exactMarginals);

% part D
calculateMED = @(predMarginals, exactMarginals) sum(abs(loopyMarginals - exactMarginals), 'all') / 8.0; % compute the MED (Mean Expected Deviation)
loopyMED = calculateMED(loopyMarginals, exactMarginals); % compute MED for loopy marginals
mfMED = calculateMED(mfMarginals, exactMarginals); % compute MED for mean-field marginals
fprintf("Loopy MED: %f\n", loopyMED);
fprintf("Mean field MED: %f\n", mfMED);