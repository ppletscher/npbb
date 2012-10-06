addpath('../../NPBB');
addpath('../../tools');
addpath('../');

% generate a Biclustering with Beta/Bernoulli model
alphaO = 1;
alphaF = 10;
distribution = [];
distribution.type = 'bernoulli';
distribution.beta0 = 1;
distribution.beta1 = 1;
[D, cO, cF, Theta] = generateBiclusteringDoubleMixtureBeta(200, 200, alphaO, alphaF, ...
					distribution);

% infer a Biclustering
results = NPBBGibbs(D, alphaO, alphaF, distribution, 'debug', 1, 'maxIter', 10);

% evaluation
[W, c] = correspondence(cO(:), results.cO);
fprintf('error objects: %f\n', c);
[W, c] = correspondence(cF(:), results.cF);
fprintf('error features: %f\n', c);
