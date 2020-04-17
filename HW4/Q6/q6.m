import brml.*;
load("EMprinter.mat");

EPS = 0.01;
[N, K] = size(x, 2); % get number of samples

t = 1;
tab = rand(2, 2);
diff = 0.02;
while diff > EPS % if true, we have not converged yet
    t = t+1;
    %for n = 1:N
    %    % TODO: compute q_t^n % E step
    %end
    %for i = 1:K
    %    
    %end
    
    diff = norm(newTab - tab, 1); % L1 norm
end