function [D, cO, cF, Theta] = generateBiclusteringDoubleMixtureBeta(nObjects, nFeatures, alphaO, alphaF, distribution)
% GENERATEBICLUSTERINGDOUBLEMIXTUREBETA sample a biclustering
%    [D, cO, cF, Theta] = GENERATEBICLUSTERINGDOUBLEMIXTUREBETA(nObjects, ...
%                          nFeatures, alphaO, alphaF, distribution)
%    sample a biclustering from a double Bernoulli mixture model, i.e.
%    a mixture model for the objects and another one for the features.
%
%    Inputs:
%    nObjects	number of objects
%    nFeatures	number of features
%    alphaO	alpha of the CRP for the objects
%    alphaF	alpha of the CRP for the features
%    distribution
%		Beta hyperparameters of the prior
%
%    Outputs:
%    D		the sampled biclustering data
%    cO		assignments of the objects
%    cF		assignments of the features
%    Theta	component for each bicluster
%
%    Example:
%    distribution.beta0 = 1;
%    distribution.beta1 = 1;
%    [D, cO, cF, Theta] = generateBiclusteringDoubleMixtureBeta(10, 10, 1, 1, distribution)


cO = crp(alphaO, nObjects);
cF = crp(alphaF, nFeatures);

NcO = length(unique(cO));
NcF = length(unique(cF));

Theta = cell(NcO, NcF);
D = zeros(nObjects, nFeatures);

for mu=1:NcO
	for nu=1:NcF
		Theta{mu,nu} = betarnd(distribution.beta0, distribution.beta1);
		
		indsO = find(cO == mu);
		indsF = find(cF == nu);

		a = repmat(1, [length(indsO)*length(indsF) 1]);
		b = repmat(1-Theta{mu,nu}, [length(indsO)*length(indsF) 1]);

		A = binornd(a, b);
		D(indsO, indsF) = reshape(A, length(indsO), length(indsF));
	end
end
