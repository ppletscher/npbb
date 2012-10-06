function visualizebiclustering(F, cO, cF)
% VISUALIZEBICLUSTERING illustrates a specified Biclustering
%    VISUALIZEBICLUSTERING(F, cO, cF, Theta) shows the 
%    Biclustering specified by cO and cF.
%
%    Inputs:
%    F		the data (rows: objects, columns: features)
%    cO		clustering of the objects
%    cF		clustering of the features

[cF, pF] = sort(cF);
[cO, pD] = sort(cO);

cOunique = unique(cO);
cFunique = unique(cF);
NcO = length(cOunique);
NcF = length(cFunique);
T = F(pD, pF);

% visualize
imagesc(T);
colormap(jet);

y = 0.5;
for i=1:NcO-1
	y = y + length(find(cO==cOunique(i)));
	line([0 size(F,2)+0.5], [y y], 'Color','r','LineWidth',1);
end

x = 0.5;
for i=1:NcF-1
	x = x + length(find(cF==cFunique(i)));
	line([x x], [0 size(F,1)+0.5], 'Color','r','LineWidth',1);
end
