import brml.*
load("dataset.dat")
load("joint.dat")

assignments = zeros(12, 1);
givens = [8 11]; % input givens
queries = [1]; % input query variable, and add 1 to it because we're working with 1-based indexing
for g = givens % preprocessing, because we're using 1-based indexing
    assignments(g + 1) = 1;
end
for i = 1:length(queries) % preprocessing, because we're using 1-based indexing
    queries(i) = queries(i) + 1;
end

for i = 1:12
    
end