classdef LibSVMClassifier
%LibSVMClassifier implement a warpper object of LibSVM.
% USAGE:
% model = LibSVMClassifier(trainingFeatures, trainingLabels, ...
%                          params, trFeatFile); 
% [predictLabel, accuracy, decValues] = model.predict( ...
%                          testfeatures, testLabels, teFeatFile); 
% 
% some input variables are optional, their default values are: 
% parmas: '-c 1 -g 0.07' 
% trFeatFile: 'trFeature.txt'
% teFeatFile: 'teFeature.txt'
    
    properties
    end
    
    methods
		function model = LibSVMClassifier(trainingFeatures, trainingLabels, params, trFeatFile)

			if nargin < 4
				trFeatFile = 'trFeature.txt';
				if nargin < 3
					params = '-c 1 -g 0.07';
				end
			end
			
			write2txt(trainingFeatures, trainingLabels, trFeatFile);
			[trLabel, trFea]=libsvmread(trFeatFile);

			%[bestCVaccuracy,bestc,bestg]=SVMcgForClass(trLabel,TrFea,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep);
			model=svmtrain(trLabel,trFea,params);
        end
        
        function [predictLabel, accuracy, decValues] = predict(model, testfeatures, testLabels, teFeatFile)
		
			if nargin < 4
				teFeatFile = 'teFeature.txt';
			end
			
			write2txt(testfeatures, testLabels, teFeatFile);
			[teLabel, teFea]=libsvmread(teFeatFile);
			[predictLabel, accuracy, decValues] = svmpredict(teLabel, teFea, model);

        end
	
    end
	
	methods(Access = private)
		function write2txt(features, labels, filename)
		%nFeat:size of the feature vector; nSample:number of samples
			[nFeat,nSample]=size(features); 
			fid=fopen(filename,'wt');
			for s = 1:nSample
				fprintf(fid,num2str(labels(s)));
				for f=1:nFeat
					fprintf(fid,' %s:%s', num2str(f), num2str(features(f,s)));  
				end
			fprintf(fid,'\n');
			end
			fclose(fid);
		end
	end
    
end

