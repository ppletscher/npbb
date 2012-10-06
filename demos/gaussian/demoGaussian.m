addpath('../../NPBB');
addpath('../../tools');
addpath('../');

% generate a Biclustering with Gaussian/Gaussian model
alphaO = 1;
alphaF = 10;
distribution = [];
distribution.type = 'gaussian';
distribution.mu0 = 5;
distribution.S0 = 2;
distribution.S1 = 0.5;
[D, cO, cF, Theta] = generateBiclusteringDoubleMixtureGaussian(200, 200, alphaO, alphaF, ...
					distribution);

% infer a Biclustering
results = NPBBGibbs(D, alphaO, alphaF, distribution, 'debug', 1, 'maxIter', 20);

% evaluation
[W, c] = correspondence(cO(:), results.cO);
fprintf('error objects: %f\n', c);
[W, c] = correspondence(cF(:), results.cF);
fprintf('error features: %f\n', c);
