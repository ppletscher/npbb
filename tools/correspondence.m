function [W, c, cpreds] = correspondence(ctrue, cpred)
% CORRESPONDENCE find an assignment for two clustering solutions
%    [W, c, cpreds] = CORRESPONDENCE(ctrue, cpred) given two clustering
%    solutions compute the label permutation that leads to the
%    smallest error. This is the classical assignment problem from
%    theoretical computer science and we use the Hungarian method
%    to solve it.
%
%    Inputs:
%    ctrue	ground-truth labels
%    cpred	labels inferred by a clustering algorithm
%
%    Outputs:
%    W		how many times does a combination (i, j) of
%		ctrue = i and cpred = j occur
%    c		costs of using the assignment returned in a
%    cpreds	mapped partition, that is as similar as possible
%		to ctrue

ctrue = ctrue(:);
cpred = cpred(:);

etrue = unique(ctrue);
epred = unique(cpred);

W = zeros(length(etrue), length(epred));

for i=1:length(etrue)
	ind = find(ctrue == etrue(i));
	for j=1:length(epred)
		W(i,j) = length(find(cpred(ind) == epred(j)));
	end
end

W0 = zeros(max(size(W)), max(size(W)));
W0(1:size(W,1), 1:size(W,2)) = W;

[a, c] = hungarian(-W0);
c = 1 + c/size(ctrue, 1);

etrue = [etrue; [max(etrue)+1:max(etrue)+1+length(epred)-length(etrue)]'];
cpreds = zeros(1, length(cpred));
for i=1:length(epred),
	ind = find(cpred == epred(i));
	cpreds(ind) = etrue(a(i));
end
