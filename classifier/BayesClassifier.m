classdef BayesClassifier
%BAYESCLASSIFIER Summary of this class goes here
%   Detailed explanation goes here
    
    properties
		F
		C
		labelsSet
    end
    
    methods
		function model = BayesClassifier(trainingFeatures, trainingLabels, labelsSet)

			nClass = length(labelsSet);
            nFeat = size(trainingFeatures,1);
			
			model.F = zeros(nFeat, nClass);
			model.C = zeros(nFeat, nFeat, nClass);
			
            for idx = 1:nClass
                data = trainingFeatures(:,trainingLabels==labelsSet(idx));
                model.F(:,idx) = mean(data,2);
                model.C(:,:,idx) = cov(data');
            end
			
			model.labelsSet = labelsSet;
        end
        
        function labels = predict(model, features, prioriProbabilities)
			
            nClass = length(model.labelsSet);
            nItem = size(features, 2);
            
            if nargin < 3
				prioriProbabilities = repmat(1/nClass, [1, nClass]);
            end
            
            scores = zeros(nClass, 1);
			labels = zeros(nItem, 1);
            
			for item = 1:nItem
                for c = 1:nClass
                    scores(c) = -0.5*(features(:,item)-model.F(:,c))' ...
                        *( model.C(:,:,c)\(features(:,item)-model.F(:,c))) ...
                        + log(prioriProbabilities(c)) ...
                        - 0.5*log(abs(det(model.C(:,:,c))));
                end
				[~,num] = max(scores);
				labels(item) = model.labelsSet(num);
			end
        end
    end
    
end

