function classArray = CRP(alpha, nSamples)
% CRP  generate samples from the Chinese restaurant process
%    classArray = CRP(alpha, nSamples)
%    sample a clustering of size nSamples from the Chinese
%    restaurant process (CRP).
%
%    Inputs:
%    alpha	concentration parameter
%    nSamples	number of samples to draw
%
%    Output:
%    classArray	a clustering sampled from the CRP

c = [];
classArray = zeros(nSamples, 1);
nClasses = 0;
for i=1:nSamples
	% select one of the already generated samples
	if (rand() < (i-1) / ((i-1) + alpha))
		cdf = cumsum(c(:)./(i-1));
		ind = sum(cdf < rand*cdf(end)) + 1;
		classArray(i,:) = ind;
		c(ind) = c(ind) + 1;
	% generate a new sample
	else
		nClasses = nClasses + 1;
		classArray(i,:) = nClasses;
		c = [c; 1];
	end
end
