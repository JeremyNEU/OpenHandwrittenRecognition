%% display benchmarkResults of testing algorithms on mnist dataset.

%% some basic processing functions.
% pre-processing
unpadding = @(img)(img(5:24,5:24));

% binary
im2bwOTSU = @(img)(im2bw(img, graythresh(img)));

% Mesh feature
countBlock = @(block_struct) sum(sum(block_struct.data));
meshfeatExtr = @(img,blockSize)(blockproc(img,blockSize,countBlock));

% convert to feature vector.
convert2Array = @(img)(img(:));

%% load mnist dataset.
mnist.trainingData = loadMNISTImages('dataset/mnist/train-images.idx3-ubyte');
mnist.trainingLabels = loadMNISTLabels('dataset/mnist/train-labels.idx1-ubyte');
mnist.testData = loadMNISTImages('dataset/mnist/t10k-images.idx3-ubyte');
mnist.testLabels = loadMNISTLabels('dataset/mnist/t10k-labels.idx1-ubyte');


%% different algorithms.

%CMB4x7 & CMB6x9 to find influence of size diff. - [4 7] is better.
UCB4x7.name = 'Unpadding-CoarseMesh-Bayes-4X7';
UCB4x7.featureExtractor = @(img)convert2Array(meshfeatExtr(unpadding(img),[4 7]));
UCB4x7.classifier = @BayesClassifier;

UCB6x9.name = 'Unpadding-CoarseMesh-Bayes-6X9';
UCB6x9.featureExtractor = @(img)convert2Array(meshfeatExtr(unpadding(img),[6 9]));
UCB6x9.classifier = @BayesClassifier;

%CMB4x7 & UCMB to find the best image preprocessing.
%nonzero, fixed threshold, adaptive threshold, edge thining 
UNMB.name = 'Unpadding-Nonzero-CoarseMesh-Bayes';
UNMB.featureExtractor = @(img)convert2Array(meshfeatExtr(unpadding(img)>0,[4 7]));
UNMB.classifier = @BayesClassifier;

UOMB.name = 'Unpadding-Otsu-CoarseMesh-Bayes';
UOMB.featureExtractor = @(img)convert2Array(meshfeatExtr(im2bwOTSU(unpadding(img)),[4 7]));
UOMB.classifier = @BayesClassifier;

%resize the image.
OACB.name = 'Otsu-AdjustBound-CoarseMesh-Bayes';
OACB.featureExtractor = @(img)extractCoarseMeshFeatures(im2bwOTSU(img),[5,3],[6,10]);
OACB.classifier = @BayesClassifier;

algorithms = {UCB4x7,UCB6x9,UNMB,OACB};

%% test and display performance table.    
colHeadings = arrayfun(@(x)sprintf('%d',x),0:9,'UniformOutput',false);
format = repmat('%-6s',1,12); % 1 row 12 col
header = sprintf(format,colHeadings{:},'| Overall | Algorithm');
fprintf('\n%s\n%s\n',header,repmat('-',size(header)));

for n = 1:length(algorithms)
    confMat = benchmark(algorithms{n}, mnist);
    confRateMat = bsxfun(@rdivide,confMat,sum(confMat,2));
    ErrorRate = 1-diag(confRateMat);
    fprintf('%-6.2f', 100*ErrorRate);
    fprintf('|  %-6.2f | ', 100*mean(ErrorRate));
    disp(algorithms{n}.name);
    algorithms{n}.confMat = confMat;
end