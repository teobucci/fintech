function D = MixDistance(X,Y)
%
% Mixed categorical/numerical distance 
%
% INPUT:
% X = matrix of features, nObs x (nCategorical + nNumerical)
%        NOTE: categorical features must be
%                               1) grouped together
%                               2) the first block 
% Y = X (for most applications)
%
% OUTPUT:
% D = matrix of distances (nObsCat+nObsNum) x (nObsCat+nObsNum)

%% Find the number of categorical and numerical features
% The idea is that categorical variables are encoded, so they are
% represented by dummy/binary variables,
% and the sum of the possibile values == 1

nFeatures = size(X,2);
nCat = 0;
for i = 1:nFeatures
    if sum(unique(X(:,i))) == 1
        nCat = nCat + 1;
    end
end
nNum = nFeatures - nCat;

%% Compute distances, separately
DCat = pdist2(X(:,1:nCat), Y(:,1:nCat), 'hamming');
DNum = pdist2(X(:,nCat+1:end), Y(:,nCat+1:end), 'cityblock');
% Compute relative weight based on the number of categorical variables
wCat = nCat/(nCat + nNum); 
D = wCat*DCat + (1 - wCat)*DNum;
end


