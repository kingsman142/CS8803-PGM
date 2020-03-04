clear all; clc;
import brml.*
load('diseaseNet.mat')
pot = setpotclass(pot, 'array');
[jtpot, jtsep, infostruct] = jtree(pot);
[jtpotfullabsorb, jtsepfullabsorb, Z] = absorption(jtpot, jtsep, infostruct);

CalculateMarginals = false(1, 40);
while(1)
    for i = 1:1:length(jtpotfullabsorb)
        for j = jtpotfullabsorb{i}.variables
            if(j > 20)
                if(~CalculateMarginals(j-20))
                    SymptomMarginalsJT{j-20} = sumpot(jtpotfullabsorb{i}, j, 0);
                    CalculatedMarginals(j-20) = true;
                    if(sum(CalculatedMarginals) == 20)
                        break;
                    end
                end
            end
        end
        
        if(sum(CalculatedMarginals) == 40)
            break;
        end
    end
    
    if(sum(CalculatedMarginals) == 40)
        break;
    end
end

for i = 1:1:40
    % changed {i+20} to (i+20)
    SymptomMarginalsBN{i} = sumpot(multpots(pot(pot(i+20).variables)), i+20, 0);
end

for i = 1:1:40
    MarginalErrors(:, i) = abs(SymptomMarginalsBN{i}.table - SymptomMarginalsJT{i}.table);
end

fprintf('Maximum error: %e\n', max(max(MarginalErrors)));

for i = 1:1:60
    % changed {i} to (i)
    potcond{i} = setpot(pot(i), [21:1:30], [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]);
end
[newpot, newvars, uniquevariables, uniquenstates] = squeezepots(potcond);

[jtpotcond, jtsepcond, infostructcond] = jtree(newpot);
[jtpotfullabsorbcond, jtsepfullabsorbcond, Zcond] = absorption(jtpotcond, jtsepcond, infostructcond);
for i = 1:1:length(jtpotfullabsorbcond)
    jtpotfullabsorbcond{i}.table = jtpotfullabsorbcond{i}.table/Zcond;
end

CalculatedMarginals = false(1, 20);
while(1)
    for i = 1:1:length(jtpotfullabsorbcond)
        for j = jtpotfullabsorbcond{i}.variables
            if(j < 21)
                if(~CalculatedMarginals(j))
                    DiseaseMarginalsJT{j} = sumpot(jtpotfullabsorbcond{i}, j, 0);
                    CalculatedMarginals(j) = true;
                    if(sum(CalculatedMarginals) == 20)
                        break;
                    end
                end
            end
        end
        
        if(sum(CalculatedMarginals) == 20)
            break;
        end
    end
    
    if(sum(CalculatedMarginals) == 20)
        break;
    end
end

CalculatedMarginals = false(1, 20);
while(1)
    for i = 1:1:length(jtpotfullabsorb)
        for j = jtpotfullabsorb{i}.variables
            if(j < 21)
                if(~CalculatedMarginals(j))
                    DiseaseMarginalsJTuncond{j} = sumpot(jtpotfullabsorb{i}, j, 0);
                    CalculatedMarginals(j) = true;
                    if(sum(CalculatedMarginals) == 20)
                        break;
                    end
                end
            end
        end
        
        if(sum(CalculatedMarginals) == 20)
            break;
        end
    end
    
    if(sum(CalculatedMarginals) == 20)
        break;
    end
end

for i = 1:1:20
    ProbabilityChange(:, i) = [DiseaseMarginalsJTuncond{i}.table(1);
        DiseaseMarginalsJT{j}.table(1)];
end