clear all; clc;
import brml.*
load("dataset.dat")
load("joint.dat")

summer_probs = zeros(2, 1);
diseases = zeros(4, 2, 2);
symptoms = zeros(7, 16, 2);

for i = 1:4000000
    if mod(i, 100000) == 0
        fprintf("Iteration %d done...\n", i);
    end
    
    assignments = de2bi(dataset(i), 12);
    
    if assignments(1) == 0
        summer_probs(1) = summer_probs(1) + 1;
    else
        summer_probs(2) = summer_probs(2) + 1;
    end
    
    for j = 1:4
        true_or_false = 1;
        if assignments(j+1) == 1 % if disease in state 0, then put it in the false bin, otherwise put it in the true bin
            true_or_false = 2;
        end
        diseases(j, assignments(1)+1, true_or_false) = diseases(j, assignments(1)+1, true_or_false) + 1;
    end
    
    disease_states = assignments(2:5);
    disease_num = bi2de(disease_states) + 1;
    for j = 1:7
        true_or_false = 1;
        if assignments(j+5) == 1 % if disease in state 0, then put it in the false bin, otherwise put it in the true bin
            true_or_false = 2;
        end
        symptoms(j, disease_num, true_or_false) = symptoms(j, disease_num, true_or_false) + 1;
    end
end

summer_probs_total_counts = summer_probs(1) + summer_probs(2);
summer_probs(1) = summer_probs(1) / summer_probs_total_counts;
summer_probs(2) = summer_probs(2) / summer_probs_total_counts;

for i = 1:4
    diseases(i, :, :) = diseases(i, :, :) ./ (diseases(i, :, 1) + diseases(i, :, 2));
end

for i = 1:7
    symptoms(i, :, :) = symptoms(i, :, :) ./ (symptoms(i, :, 1) + symptoms(i, :, 2));
end