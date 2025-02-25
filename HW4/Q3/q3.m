import brml.*;
load("ChowLiuData.mat")

[D, N] = size(X);

weights = zeros(D, D);
for i = 1:D
    for j = 1:D
        % Compute the mutual information for the pair of variables x_i, x_j
        % : w_ij = MI(x_i; x_j)
        mutualInfo = 0.0;
        for x = 1:3
            cX = X(i, :) == x; % samples where X == x
            pX = sum(cX) / N; % p(X == x)
            for y = 1:3
                cY = X(j, :) == y; % samples where Y == y
                cXY = cX & cY; % samples where X == x and Y == y
                pY = sum(cY) / N;% p(Y == y)
                pXY = sum(cXY) / N; % p(X == x, Y == y)
                logMutual = log(pXY / (pX * pY));
                if isnan(logMutual) || isinf(logMutual)
                    logMutual = -100000;
                end
                %fprintf("%f %f %f %f\n", pX, pY, pXY, logMutual);
                
                % calculate mutual information for this iteration and add
                % it to the weight of this node
                if logMutual ~= -100
                    mutualInfo = mutualInfo + pXY * logMutual; % MI += p(x,y) * log( p(x, y) / p(x)*p(y) )
                end
            end
        end
        weights(i, j) = mutualInfo;
    end
end

[Atree, elimseq, weight] = spantree(weights);