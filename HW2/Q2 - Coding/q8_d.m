import brml.*
load("dataset.dat")
load("joint.dat")

jointPred = zeros(4096, 2);

for i = 1:4096
    decimalTag = joint(i, 1);
    assignments = de2bi(decimalTag, 12);
    
    summerProb = summer_probs(assignments(1) + 1);
    prob = summerProb;
    
    for j = 2:5
        prob = prob * diseases(j-1, assignments(1)+1, assignments(j)+1);
    end
    
    disease_num = bi2de(assignments(2:5))+1;
    for j = 6:12
        prob = prob * symptoms(j-5, disease_num, assignments(j)+1);
    end
    
    jointPred(i, 1) = decimalTag;
    jointPred(i, 2) = prob;
end

predTrueDiff = abs(joint(:, 2) - jointPred(:, 2));
sumDiff = sum(predTrueDiff);
[maxDiff, maxIndex] = max(predTrueDiff);

fprintf("Error: %f\n", sumDiff);
fprintf("Max diff: %f, true: %f, pred: %f\n", maxDiff, joint(maxIndex, 2), jointPred(maxIndex, 2));