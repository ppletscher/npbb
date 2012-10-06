function results = NPBBGibbs(X, alphaO, alphaF, distribution, varargin)
% NPBBGibbs  Nonparametric Bayesian Biclustering with Gibbs sampler
%   results = NPBBGibbs(X, alphaO, alphaF, distribution, varargin)
%   Infer a Biclustering for the data given in X assuming a Nonparameteric
%   Bayesian Biclustering model with a collapsed Gibbs sampler.
%
%   Inputs:
%   X		data (columns: features, rows: objects).
%   alphaO	mixing constant of the CRP (for the object clustering).
%   alphaF	mixing constant of the CRP (for the feature clustering).
%   distribution
%      	object consisting of the prior and the likelihood.
%
%   Outputs (in results struct):
%   cO		data clustering assignments
%   cF		feature clustering assignments
%   hist_cO	"history" of the object clustering
%   hist_cF	"history" of the feature clustering
%
%   The function accepts several option-value pairs, specifying
%   initializations and parameters of the algorithm
%
%   'threshold' 1e-5
%     we stop if the weighted average of (percentual)
%     assignment changes < threshold.
%
%   'maxIter' 30
%     maximum number of iterations
%
%   'debug' 0
%     whether we should show debug information
%
%   'clusteringObjects' []
%     initial object clustering
%
%   'clusteringFeatures' []
%     initial feature clustering

%   Copyright (C) 2007-2008  Patrick Pletscher  [pletscher at inf dot ethz dot ch]
%   ETH Zurich, Switzerland
%
%   Revision: $Rev: 1357 $, $LastChangedDate: 2008-12-26 17:18:30 +0100 (Fri, 26 Dec 2008) $
%
%   This file is part of NPBB.
%
%   NPBB is free software; you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation; either version 2 of the License, or
%   (at your option) any later version.
%
%   NPBB is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You should have received a copy of the GNU General Public License
%   along with NPBB; if not, write to the Free Software
%   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

[nSamples, d] = size(X);

% initialization
[cOArray, cOnClasses, cOCounter] = ClusteringInitialization(size(X,1));
[cFArray, cFnClasses, cFCounter] = ClusteringInitialization(size(X,2));
debug = 0;
maxIter = 30;
threshold = 1e-5;

% go through the optional arguments 
for k=1:2:length(varargin)
	switch lower(varargin{k})
	case 'debug'
		debug = varargin{k+1};
	case 'clusteringobjects'
		[cOArray, cOnClasses, cOCounter] = ...
			ClusteringInitialization(size(X,1), varargin{k+1});
	case 'clusteringfeatures'
		[cFArray, cFnClasses, cFCounter] = ...
		 	ClusteringInitialization(size(X,2), varargin{k+1});
	case 'maxiter'
		maxIter = varargin{k+1};
	case 'threshold'
		threshold = varargin{k+1};
	otherwise
		error(['Unknown parameter ' varargin{k} '.']);
	end
end

if (debug)
	figure, visualizebiclustering(X, cOArray, cFArray);
	axis tight;
	drawnow;
	set(gca, 'nextplot', 'replacechildren');

	hO = [];
	hF = [];
end

hist_cO = [cOArray(:)];
hist_cF = [cFArray(:)];

iter = 1;
changeF = 2*threshold;
changeO = 2*threshold;
while ((changeO > threshold || changeF > threshold) && iter < maxIter)
	% change the assignments of the objects
	cOArray_old = cOArray;
	[cOArray, cOCounter, cOnClasses] = CRPStep(X, alphaO, distribution, cOArray, ...
			cOCounter, cOnClasses, cFArray, cFCounter, cFnClasses);
	
	% change the assignments of the features
	cFArray_old = cFArray;
	[cFArray, cFCounter, cFnClasses] = CRPStep(X', alphaF, distribution, cFArray, ...
			cFCounter, cFnClasses, cOArray, cOCounter, cOnClasses);
	
	% TODO: find a good measure of convergence
	changeO = 0.7*sum(sign(abs(cOArray_old - cOArray)))/nSamples + 0.3*changeO;
	changeF = 0.7*sum(sign(abs(cFArray_old - cFArray)))/d + 0.3*changeF;

	if (debug)
		visualizebiclustering(X, cOArray, cFArray);
		drawnow;
		
		hO = [hO changeO];
		hF = [hF changeF];
	end
	hist_cO = [hist_cO, cOArray(:)];
	hist_cF = [hist_cF, cFArray(:)];

	iter = iter+1
end

if (debug)
	figure();
	plot(hO, 'b-');
	hold on;
	plot(hF, 'r-');
	hold off;
	legend('objects clustering', 'feature clustering');
end

% store the results (and history) in a struct
results.cO = cOArray;
results.cF = cFArray;
results.hist_cO = hist_cO;
results.hist_cF = hist_cF;
results.weightsO = cOCounter./nSamples;
results.weightsF = cFCounter./d;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [cMainArray, cMainCounter, cMainnClasses] = CRPStep(X, alpha, ...
		distribution, cMainArray, cMainCounter, cMainnClasses, ...
		cOtherArray, cOtherCounter, cOthernClasses)

[dmain, dother] = size(X);
	
for i=1:dmain
	% remove previous assignment of i
	ind = cMainArray(i);
	cMainCounter(ind) = cMainCounter(ind) - 1;
	if (cMainCounter(ind) == 0)
		cMainCounter(ind) = [];
		cMainnClasses = cMainnClasses-1;
		ind = find(cMainArray > ind);
		cMainArray(ind) = cMainArray(ind)-1;
	end
	
	% compute probability of being assigned to
	% an already generated cluster
	Xtemp = X;
	Xtemp(i,:) = [];
	ctemp = cMainArray;
	ctemp(i) = [];
	q = log(cMainCounter) + ...
		BiLogEvidencePosterior(X(i,:), Xtemp, distribution, ctemp, ...
					cMainnClasses, cOtherArray, cOthernClasses);

	% compute probability of being assigned to
	% a new cluster
	r = log(alpha) + BiLogEvidence(X(i,:), distribution, cOtherArray, cOthernClasses);

	% normalize the two statistics
	temp = [q(:)' r];
	temp = temp - logsumexp(temp,2);
	q = exp(temp(1:cMainnClasses));
	r = exp(temp(cMainnClasses+1));

	% choose an already generated cluster
	if (rand() < 1-r)
		% sample a new assignment
		cdf = cumsum(q(:)./(1-r));
		ind = sum(cdf < rand*cdf(end)) + 1;

		% and assign to the cluster
		cMainArray(i) = ind;
		cMainCounter(ind) = cMainCounter(ind) + 1;
	% choose a new cluster
	else
		cMainnClasses = cMainnClasses + 1;
		ind = cMainnClasses;
		cMainArray(i) = ind;
		cMainCounter(ind) = 1;
	end

	if (mod(i, 20) == 0)
		fprintf('i: %d, cMainnClasses: %d\n', i, cMainnClasses);
	end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [cArray, cNClasses, cCounter] = ClusteringInitialization(n, cArray)

if (nargin == 1 || length(cArray) == 0)
	cArray = ceil(5*rand(n, 1));
elseif (length(cArray) ~= n)
	error('initialization of clustering must have correct length!');
end
cArrayUnique = unique(cArray);
cNClasses = length(cArrayUnique);
cCounter = zeros(1, cNClasses);
cArrayNew = zeros(size(cArray));
for i=1:cNClasses
	ind = find(cArray == cArrayUnique(i));
	cArrayNew(ind) = i;
	cCounter(i) = length(ind);
end
cArray = cArrayNew;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p = BiLogEvidencePosterior(x, X, distribution, cMain, NcMain, cOther, NcOther)

p = zeros(1, NcMain);
for i=1:NcMain,
	Xtemp = X(find(cMain == i), :);
	p(i) = BiLogProbabilityData(x, Xtemp, distribution, cOther, NcOther);
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p = BiLogEvidence(x, distribution, cOther, NcOther)

if (isequal(distribution.type, 'bernoulli'))
	p = 0;
	for i=1:NcOther,
		inds = find(cOther == i);
	
		cnt0 = sum(x(inds) == 0);
		cnt1 = sum(x(inds) == 1);
	
		p = p + betaln(cnt0+distribution.beta0, cnt1+distribution.beta1) - ...
			betaln(distribution.beta0, distribution.beta1);
	end
elseif (isequal(distribution.type, 'gaussian'))
	Spinv = 1/distribution.S0;
	Slinv = 1/distribution.S1;
	
	p = 0;
	for i=1:NcOther,
		inds = find(cOther == i);
	
		T = length(inds)*Slinv + Spinv;
		S = 1/T;
		mu = S*(Slinv*sum(x(inds)) + Spinv*distribution.mu0);
	
		temp = x(inds);
		temp = temp(:)';
		
		p = p + -0.5*temp*Slinv*temp'-0.5*distribution.mu0'*...
			Spinv*distribution.mu0+0.5*mu'*T*mu;
		p = p + 0.5*log(S) - ( (length(inds)/2)*log(2*pi) + ...
			0.5*log(distribution.S1) + 0.5*log(distribution.S0) );
	end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p = BiLogProbabilityData(x, X, distribution, cOther, NcOther)

if (isequal(distribution.type, 'bernoulli'))
	p = 0;
	for i=1:NcOther,
		inds = find(cOther == i);
		Xtemp = X(:, inds);
	
		prior0 = sum(Xtemp(:) == 0)+distribution.beta0;
		prior1 = sum(Xtemp(:) == 1)+distribution.beta1;
		
		cnt0 = sum(x(inds) == 0);
		cnt1 = sum(x(inds) == 1);
	
		p = p + betaln(cnt0+prior0, cnt1+prior1) - betaln(prior0, prior1);
	end
elseif (isequal(distribution.type, 'gaussian'))
	Spinv = inv(distribution.S0);
	Slinv = inv(distribution.S1);
	T = Slinv + Spinv;
	S = inv(T);
	
	p = 0;
	for i=1:NcOther,
		inds = find(cOther == i);
		Xtemp = X(:, inds);
	
		% compute the combined mean/covariance
		T1 = prod(size(Xtemp))*Slinv + Spinv;
		S1 = inv(T1);
		mu1 = S1*(Slinv*sum(Xtemp(:)) + Spinv*distribution.mu0);
	
		T2 = length(inds)*Slinv + T1;
		S2 = inv(T2);
		mu2 = S2*(Slinv*sum(x(inds)) + T1*mu1);
	
		temp = x(inds);
		temp = temp(:)';
	
		% compute the evidence, i.e. we integrate the prior out
		p = p + -0.5*temp*Slinv*temp'-0.5*mu1'*T1*mu1+0.5*mu2'*T2*mu2;
		p = p + 0.5*log(S2) - ( (length(inds)/2)*log(2*pi) + ...
			0.5*log(distribution.S1) + 0.5*log(S1) );
	end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function s = logsumexp(a, dim)
% Returns log(sum(exp(a),dim)) while avoiding numerical underflow.
% Default is dim = 1 (columns).
% logsumexp(a, 2) will sum across rows instead of columns.

if nargin < 2
	dim = 1;
end

% subtract the largest in each column
[y, i] = max(a,[],dim);
dims = ones(1,ndims(a));
dims(dim) = size(a,dim);
a = a - repmat(y, dims);
s = y + log(sum(exp(a),dim));
i = find(~isfinite(y));
if ~isempty(i)
	s(i) = y(i);
end
