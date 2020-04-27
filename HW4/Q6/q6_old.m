import brml.*;
load("EMprinter.mat");

EPS = 0.01;
[D, N] = size(x); % get number of samples and dimensionality

fuse = array(1, condp(rand(1, 2)));
drum = array(2, condp(rand(1, 2)));
toner = array(3, condp(rand(1, 2))); 
paper = array(4, condp(rand(1, 2)));
roller = array(5, condp(rand(1, 2))); 
burning = array([6 1], condp(rand(2, 2)));
quality = array([7 2 3 4], condp(rand(2, 2, 2, 2)));
wrinkled = array([8 1 4], condp(rand(2, 2, 2)));
multpages = array([9 4 5], condp(rand(2, 2, 2))); 
paperjam = array([10 1 5], condp(rand(2, 2, 2)));
diff = 0.02;
while diff > EPS % if true, we have not converged yet
    oldJoint = multpots({fuse drum toner paper roller burning quality wrinkled multpages paperjam});
    %for n = 1:N
    %    % TODO: compute q_t^n % E step
    %end
    %for i = 1:D
    %    
    %end
    for n = 1:N
        sample = x(:, n);
        keepVariables = find(isnan(sample)); % find the variables in this data point that are NaN
        newDist = oldJoint.table;
        newDist = (~any(keepVariables(:) == 1)) : newDist(sample(1), :, :, :, :, :, :, :, :, :) : newDist;
        newDist = (~any(keepVariables(:) == 2)) : newDist(:, sample(2), :, :, :, :, :, :, :, :) : newDist;
        newDist = (~any(keepVariables(:) == 3)) : newDist(:, :, sample(3), :, :, :, :, :, :, :) : newDist;
        newDist = (~any(keepVariables(:) == 4)) : newDist(:, :, :, sample(4), :, :, :, :, :, :) : newDist;
        newDist = (~any(keepVariables(:) == 5)) : newDist(:, :, :, :, sample(5), :, :, :, :, :) : newDist;
        newDist = (~any(keepVariables(:) == 6)) : newDist(:, :, :, :, :, sample(6), :, :, :, :) : newDist;
        newDist = (~any(keepVariables(:) == 7)) : newDist(:, :, :, :, :, :, sample(7), :, :, :) : newDist;
        newDist = (~any(keepVariables(:) == 8)) : newDist(:, :, :, :, :, :, :, sample(8), :, :) : newDist;
        newDist = (~any(keepVariables(:) == 9)) : newDist(:, :, :, :, :, :, :, :, sample(9), :) : newDist;
        newDist = (~any(keepVariables(:) == 10)) : newDist(:, :, :, :, :, :, :, :, :, sample(10)) : newDist;
        newDist = array(keepVariables, condp(squeeze(newDist), [1 2 3 4 5 6 7 8 9 10]));
        [argvalue, argmax] = max(newDist);
        for i = 1:size(keepVariables)
            currVariable = keepVariables(i);
            currValue = argmax(i);
            x(currVariable, n) = currValue;
        end
    end
    
    newJoint = multpots({fuse drum toner paper roller burning quality wrinkled multpages paperjam});
    
    diff = sumpots(abs(newJoint - oldJoint), [1 2 3 4 5 6 7 8 9 10]); % L1 norm
end