function [D, cO, cF, Theta] = generateBiclusteringDoubleMixtureGaussian(nObjects, nFeatures, alphaO, alphaF, distribution)
% GENERATEBICLUSTERINGDOUBLEMIXTUREGAUSSIAN sample a biclustering
%    [D, cO, cF, Theta] = GENERATEBICLUSTERINGDOUBLEMIXTUREGAUSSIAN(nObjects, ...
%                          nFeatures, alphaO, alphaF, distribution)
%    sample a biclustering from a double Gaussian mixture model, i.e.
%    a mixture model for the objects and another one for the features.
%
%    Inputs:
%    nObjects	number of objects
%    nFeatures	number of features
%    alphaO	alpha of the CRP for the objects
%    alphaF	alpha of the CRP for the features
%    distribution
%		prior and likelihood hyperparameters
%
%    Outputs:
%    D		the sampled biclustering data
%    cO		assignments of the objects
%    cF		assignments of the features
%    Theta	component for each bicluster
%
%    Example:
%    distribution.mu0 = 5;
%    distribution.S0 = 2;
%    distribution.S1 = 0.5;
%    [D, cO, cF, Theta] = generateBiclusteringDoubleMixtureGaussian(10, 10, 1, 1, distribution)


cO = crp(alphaO, nObjects);
cF = crp(alphaF, nFeatures);

NcO = length(unique(cO));
NcF = length(unique(cF));

Theta = cell(NcO, NcF);
D = zeros(nObjects, nFeatures);

for mu=1:NcO
	for nu=1:NcF
		Theta{mu, nu} = randnorm(1, distribution.mu0, [], distribution.S0);
		indsO = find(cO == mu);
		indsF = find(cF == nu);
		A = randnorm(length(indsO)*length(indsF), Theta{mu,nu}, [], distribution.S1);
		D(indsO, indsF) = reshape(A, length(indsO), length(indsF));
	end
end
