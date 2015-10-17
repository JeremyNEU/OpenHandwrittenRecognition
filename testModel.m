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

digits = '0':'9';
colHeadings = arrayfun(@(x)sprintf('%d',x),0:9,'UniformOutput',false);
format = repmat('%-9s',1,11);
header = sprintf(format,'digit  |',colHeadings{:});
fprintf('\n%s\n%s\n',header,repmat('-',size(header)));
for idx = 1:numel(digits)
    fprintf('%-9s',   [digits(idx) '      |']);
    fprintf('%-9.2f', confMat(idx,:));
    fprintf('\n')
end