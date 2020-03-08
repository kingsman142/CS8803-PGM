import brml.*
load("dataset.dat")
load("joint.dat")

assignments = zeros(12, 1);
givens = [8 11]; % input givens
givenAssignments = [0 0 0 0 0 0 0 0 1 0 0 1]; % hasSummer is the left-most node
queries = [1]; % input query variable, and add 1 to it because we're working with 1-based indexing
for g = givens % preprocessing, because we're using 1-based indexing
    assignments(g + 1) = 1;
end
assignments = assignments + 1;
for i = 1:length(queries) % preprocessing, because we're using 1-based indexing
    %queries(i, :) = queries(i, :) + 1;
end

newSymptoms = zeros(7, 16);
newDiseases = zeros(4, 2);

for i = 5:11
    if any(givens == i) == true % when we marginalize over a symptom, its distribution becomes 1
        newSymptoms(i-4, :) = symptoms(i-4, :, assignments(i+1)); % since the variable is given, we take the column indexed for 'True', which is column 2
    elseif any(queries == i) == false
        newSymptoms(i-4, :) = ones(16, 1);
    end
end

messageTable = ones(16, 1); % parameterized by 4 binary variables 2^4 = 16 states
firstMarginalizedDisease = 1;
for i = 4:-1:1 % find the first disease to marginalize over to initialize our message table
    if any(givens == i) == false && any(queries == i) == false % marginalize in order from 1, 2, 3, and 4, and find the first one we marginalize over
        firstMarginalizedDisease = i;
        break
    end
end
numDiseasesNotMarginalized = 4; % there are 4 diseases, and this dictates the size of messageTable at each step
variablesLeft = [1 2 3 4];
hasSummerFalseVarFalse = diseases(firstMarginalizedDisease, 1, 1);
hasSummerFalseVarTrue = diseases(firstMarginalizedDisease, 1, 2);
hasSummerTrueVarFalse = diseases(firstMarginalizedDisease, 2, 1);
hasSummerTrueVarTrue = diseases(firstMarginalizedDisease, 2, 2);
% marginalize out the probability distributions of this disease
for j = 0:(cast(length(messageTable)/2, 'int64')-1) % TODO: switch j to go from 1 to length(messageTable)-1    
    %fprintf("here, iteration %d\n", j);
    %fprintf("marginalizing node: %d\n", firstMarginalizedDisease);
    assign = de2bi(j, numDiseasesNotMarginalized);
    newAssign = zeros(4, 1); %assign(setdiff(1:numDiseasesNotMarginalized, firstMarginalizedDisease));
    currAssignmentIndex = 1;
    for k = 1:4
        if any(givens == k) == false
            newAssign(k) = assign(currAssignmentIndex);
            currAssignmentIndex = currAssignmentIndex + 1;
        end
    end
    
    falseAssign = newAssign.';
    falseAssign(firstMarginalizedDisease) = 0;
    falseAssignDecimal = bi2de(falseAssign);
    trueAssign = newAssign.';
    trueAssign(firstMarginalizedDisease) = 1;
    trueAssignDecimal = bi2de(trueAssign);
    %disp(falseAssign);
    %disp(trueAssign);
    fprintf("here2 %d %d\n", falseAssignDecimal, trueAssignDecimal);
    
    probFalse = 1.0;
    probTrue = 1.0;
    for givenIndex = givens 
        if givenIndex >= 5 % if the given is a leaf node
            fprintf("given: %d, val: %f\n", givenIndex, newSymptoms(givenIndex-4, trueAssignDecimal+1));
            probFalse = probFalse * newSymptoms(givenIndex-4, falseAssignDecimal+1); % the marginalized variable is false
            probTrue = probTrue * newSymptoms(givenIndex-4, trueAssignDecimal+1); % else, the marginalized variable is true
        end
    end

    %fprintf("%f %f %f %f\n", hasSummerFalseVarFalse, hasSummerFalseVarTrue, hasSummerTrueVarFalse, hasSummerTrueVarTrue);
    %fprintf("%f %f\n", probFalse, probTrue);
    varFalseNew = hasSummerFalseVarFalse * probFalse + hasSummerFalseVarTrue * probTrue;
    varTrueNew = hasSummerTrueVarFalse * probFalse + hasSummerTrueVarTrue * probTrue;
    %fprintf("%f %f\n", varFalseNew, varTrueNew);

    varFalseNewIndex = bi2de([0 assign]);
    messageTable(varFalseNewIndex+1, :) = varFalseNew;

    varTrueNewIndex = bi2de([1 assign]);
    messageTable(varTrueNewIndex+1, :) = varTrueNew;
    %disp([0 assign]);
    %disp([1 assign]);
    %fprintf("%d %d\n", varFalseNewIndex, varTrueNewIndex);
    fprintf("messageTable(%d, :) = %f -- %f %f %f %f\n", varFalseNewIndex+1, varFalseNew, hasSummerFalseVarFalse, probFalse, hasSummerFalseVarTrue, probTrue);
    fprintf("messageTable(%d, :) = %f -- %f %f %f %f\n", varTrueNewIndex+1, varTrueNew, hasSummerTrueVarFalse, probFalse, hasSummerTrueVarTrue, probTrue);
end
numDiseasesNotMarginalized = numDiseasesNotMarginalized - 1;
variablesLeft(find(variablesLeft == firstMarginalizedDisease)) = [];
variablesLeft = [0 variablesLeft];
%error("END OF CODE BLOCK")
%END THE ABOVE CODE

for i = (firstMarginalizedDisease-1):-1:1 %(firstMarginalizedDisease+1)
    if any(givens == i) == true % given variable
        newDiseases(i, :) = diseases(i, :, 2);
    elseif any(queries == i) == false % not a query or given variable, so marginalize over it
        fprintf("marginalizing node %d\n", i);
        newMessageTable = zeros(2^numDiseasesNotMarginalized, 1);
        
        hasSummerFalseVarFalse = diseases(i, 1, 1);
        hasSummerFalseVarTrue = diseases(i, 1, 2);
        hasSummerTrueVarFalse = diseases(i, 2, 1);
        hasSummerTrueVarTrue = diseases(i, 2, 2);
        % marginalize out the probability distributions of this disease
        %{
        for j = 0:(cast(length(messageTable)/2, 'int64')-1)
            assign = de2bi(j, 4);
            newAssign = assign(setdiff(1:4, i));

            falseAssign = assign;
            falseAssign(numDiseasesNotMarginalized) = 0;
            falseAssignDecimal = bi2de(falseAssign);
            trueAssign = assign;
            trueAssign(numDiseasesNotMarginalized) = 1;
            trueAssignDecimal = bi2de(trueAssign);

            probFalse = messageTable(falseAssignDecimal+1); % the marginalized variable is false
            probTrue = messageTable(trueAssignDecimal+1); % else, the marginalized variable is true

            varFalseNew = hasSummerFalseVarFalse * probFalse + hasSummerFalseVarTrue * probTrue;
            varTrueNew = hasSummerTrueVarFalse * probFalse + hasSummerTrueVarTrue * probTrue;

            varFalseNewIndex = bi2de([0 newAssign]);
            newMessageTable(varFalseNewIndex+1, :) = varFalseNew;

            varTruenewIndex = bi2de([1 newAssign]);
            newMessageTable(varTrueNewIndex+1, :) = varTrueNew;
        end
        %}
        
        for j = 0:(cast(length(newMessageTable), 'int64')-1) % TODO: switch j to go from 1 to length(messageTable)-1
            %{
            %fprintf("here, iteration %d\n", j);
            %fprintf("marginalizing node: %d\n", firstMarginalizedDisease);
            assign = de2bi(j, 4);
            %newAssign = zeros(4, 1); %assign(setdiff(1:numDiseasesNotMarginalized, firstMarginalizedDisease));
            %currAssignmentIndex = 1;
            %for k = 1:4
            %    if any(givens == k) == false
            %        newAssign(k) = assign(currAssignmentIndex);
            %        currAssignmentIndex = currAssignmentIndex + 1;
            %    end
            %end
            newAssign = [0 assign.'];

            falseAssign = newAssign.';
            falseAssign(i) = 0;
            falseAssignDecimal = bi2de(falseAssign);
            trueAssign = newAssign.';
            trueAssign(i) = 1;
            trueAssignDecimal = bi2de(trueAssign);
            %disp(falseAssign);
            %disp(trueAssign);
            %fprintf("here2 %d %d\n", falseAssignDecimal, trueAssignDecimal);
            
            falseRetrieveMessage = bi2de([newAssign 0]);
            trueRetrieveMessage = bi2de([newAssign 1]);
            probFalse = messageTable(falseRetrieveMessage+1);
            probTrue = messageTable(trueRetrieveMessage+1);

            %fprintf("%f %f %f %f\n", hasSummerFalseVarFalse, hasSummerFalseVarTrue, hasSummerTrueVarFalse, hasSummerTrueVarTrue);
            %fprintf("%f %f\n", probFalse, probTrue);
            varFalseNew = hasSummerFalseVarFalse * probFalse + hasSummerFalseVarTrue * probTrue;
            varTrueNew = hasSummerTrueVarFalse * probFalse + hasSummerTrueVarTrue * probTrue;
            %fprintf("%f %f\n", varFalseNew, varTrueNew);

            varFalseNewIndex = bi2de([0 assign]);
            messageTable(varFalseNewIndex+1, :) = varFalseNew;

            varTrueNewIndex = bi2de([1 assign]);
            messageTable(varTrueNewIndex+1, :) = varTrueNew;
            disp([0 assign]);
            disp([1 assign]);
            %fprintf("%d %d\n", varFalseNewIndex, varTrueNewIndex);
            %}
            
            assign = de2bi(j, numDiseasesNotMarginalized);
            
            falseAssign = [assign(1:end) 0];
            falseAssignDecimal = bi2de(falseAssign);
            trueAssign = [assign(1:end) 1];
            disp("here");
            disp(assign);
            disp(falseAssign);
            disp(trueAssign);
            trueAssignDecimal = bi2de(trueAssign);
            probFalse = messageTable(falseAssignDecimal + 1);
            probTrue = messageTable(trueAssignDecimal + 1);
            
            varFalseNew = diseases(i, assign(1)+1, 1) * probFalse;
            varTrueNew = diseases(i, assign(1)+1, 2) * probTrue;
            
            newMessageTable(j+1, :) = varFalseNew + varTrueNew;
        end
        
        messageTable = newMessageTable;
        numDiseasesNotMarginalized = numDiseasesNotMarginalized - 1;
    
        if i == 2
            error("MARGINALIZING OVER NODE 2")
        elsif
            disp("MARGINALIZING OVER ANOTHER NODE")
        end
    end
end

% sum over hasSummer
% TODO: hardcoded this, change it later
finalTable = zeros(2, 1);
finalTable(1) = summer_probs(1)*diseases(1, 1, 1)*messageTable(bi2de([0 0])+1) + summer_probs(2)*diseases(1, 2, 1)*messageTable(bi2de([1 0])+1);
finalTable(2) = summer_probs(1)*diseases(1, 1, 2)*messageTable(bi2de([0 1])+1) + summer_probs(2)*diseases(1, 2, 2)*messageTable(bi2de([1 1])+1);

disp(finalTable(1))
disp(finalTable(2))
fprintf("Prob x_1 == FALSE: %f\n", finalTable(1) / (finalTable(1) + finalTable(2)));
fprintf("Prob x_1 == TRUE: %f\n", finalTable(2) / (finalTable(1) + finalTable(2)));