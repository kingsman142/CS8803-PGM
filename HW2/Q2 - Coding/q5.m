import brml.*
load('diseaseNet.mat')
pot = setpotclass(pot, 'array');
[jtpot, jtsep, infostruct] = jtree(pot);
[jtpotfullabsorb, jtsepfullabsorb, Z] = absorption(jtpot, jtsep, infostruct);

numSymptoms = 40;

% part a
marginalDone = false(1, numSymptoms);
while(sum(marginalDone) < numSymptoms)
    for i = 1:1:length(jtpotfullabsorb)
        for j = jtpotfullabsorb{i}.variables
            if(j > 20)
                if(~marginalDone(j-20))
                    naiveMarginals{j-20} = sumpot(jtpotfullabsorb{i}, j, 0);
                    marginalDone(j-20) = true;
                    if(sum(marginalDone) == 20)
                        break;
                    end
                end
            end
        end
        
        if(sum(marginalDone) == numSymptoms)
            break;
        end
    end
end

% part b
for i = 1:numSymptoms
    newSymptomMarginals{i} = sumpot(multpots(pot(pot(i+20).variables)), i+20, 0);
end

for i = 1:numSymptoms
    algorithmError(:, i) = abs(newSymptomMarginals{i}.table - naiveMarginals{i}.table);
end

fprintf('Maximum error: %e\n', max(max(algorithmError)));

% part c
for i = 1:60
    potcond{i} = setpot(pot(i), [21:30], [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]); 
end
[newpot, newvars, uniquevariables, uniquenstates] = squeezepots(potcond);

[jtpotcond, jtsepcond, infostructcond] = jtree(newpot);
[jtpotfullabsorbcond, jtsepfullabsorbcond, Zcond] = absorption(jtpotcond, jtsepcond, infostructcond);
for i = 1:length(jtpotfullabsorbcond)
    jtpotfullabsorbcond{i}.table = jtpotfullabsorbcond{i}.table/Zcond;
end

% calculate marginals with our set states
marginalDone = false(1, 20);
while(1)
    for i = 1:1:length(jtpotfullabsorbcond)
        for j = jtpotfullabsorbcond{i}.variables
            if(j < 21)
                if(~marginalDone(j))
                    diseaseMarginals{j} = sumpot(jtpotfullabsorbcond{i}, j, 0);
                    marginalDone(j) = true;
                    if(sum(marginalDone) == 20)
                        break;
                    end
                end
            end
        end
        
        if(sum(marginalDone) == 20)
            break;
        end
    end
    
    if(sum(marginalDone) == 20)
        break;
    end
end

% calculate marginals without our set states so we can compare
marginalDone = false(1, 20);
while(1)
    for i = 1:1:length(jtpotfullabsorb)
        for j = jtpotfullabsorb{i}.variables
            if(j < 21)
                if(~marginalDone(j))
                    diseaseMarginalsNaive{j} = sumpot(jtpotfullabsorb{i}, j, 0);
                    marginalDone(j) = true;
                    if(sum(marginalDone) == 20)
                        break;
                    end
                end
            end
        end
        
        if(sum(marginalDone) == 20)
            break;
        end
    end
    
    if(sum(marginalDone) == 20)
        break;
    end
end