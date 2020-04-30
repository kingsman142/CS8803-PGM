import brml.*;
load("EMprinter.mat");

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

% use EM algorithm to learn all CPTs of the network
pot = {fuse, drum, toner, paper, roller, burning, quality, wrinkled, multpages, paperjam};
pars.maxiterations = 10000;
[newpot, loglik] = EMbeliefnet(pot, x, pars);

% inference part of Q6
joint = condpot(multpots(newpot));
summedOut = sumpot(joint, [1 3 4 5 9 10]);
drumUnitProbs = condp(squeeze(summedOut.table(:, 1, 2, 1)));
fprintf("Prob. of drum unit problem: %.4f\n", drumUnitProbs(2));