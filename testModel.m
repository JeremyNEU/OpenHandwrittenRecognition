function confMat = testModel(model, featExtrFunc, featSize)
%TESTMODEL 
% extract feature vectors of FEATSIZE via FEATEXTRFUNC, test MODEL and
% compute CONFMAT.

%% load test dataset
testData = loadMNISTImages('dataset/mnist/t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('dataset/mnist/t10k-labels.idx1-ubyte');

N = length(testData);
img = reshape(testData,28,28,N);

%% feature extraction

testFeatures = zeros(featSize,N);

for idx = 1:N
	testFeatures(:,idx) = featExtrFunc(img(:,:,idx));
end

%% test model

predictedLabels = model.predict(testFeatures);

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));