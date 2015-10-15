% %% load model
% load('Model.mat');
% 
% %% configuration
% % load('config.mat');
% binaryFunc = @(img)(im2bw(img, graythresh(img))); %Otsu
% 
% featExtrFunc = @extractCoarseMeshFeatures;
% meshSize = [3 6];
% blockSize = [5 5];
% params = {meshSize,blockSize};
% featSize = meshSize(1)*meshSize(2);
% 
% Classifier = @BayesClassifier;

%% load dataset
testData = loadMNISTImages('dataset/mnist/t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('dataset/mnist/t10k-labels.idx1-ubyte');

% display_network(images(:,1:100)); % Show the first 100 images
N = length(testData);
img = reshape(testData,28,28,N);

% imshow(img(:,:,1)) % Show the first image

%% feature extraction

testFeatures = zeros(featSize,N);

for idx = 1:N
	bw = binaryFunc(img(:,:,idx)); % pre-processing
	testFeatures(:,idx) = featExtrFunc(bw, params{:});
end

%% test

predictedLabels = Model.predict(testFeatures);

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