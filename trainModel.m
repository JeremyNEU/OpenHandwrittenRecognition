%% configuration
% load('config.mat');
binaryFunc = @(img)(im2bw(img, graythresh(img))); %Otsu

featExtrFunc = @extractCoarseMeshFeatures;
meshSize = [5,3];
blockSize = [6,10];
params = {meshSize,blockSize};
featSize = meshSize(1)*meshSize(2);

Classifier = @BayesClassifier;

%% load dataset
trainingData = loadMNISTImages('dataset/mnist/train-images.idx3-ubyte');
trainingLabels = loadMNISTLabels('dataset/mnist/train-labels.idx1-ubyte');

% display_network(images(:,1:100)); % Show the first 100 images
N = length(trainingData);
img = reshape(trainingData,28,28,N);

% imshow(img(:,:,1)) % Show the first image

%% feature extraction

trainingFeatures = zeros(featSize,N);

for idx = 1:N
	bw = binaryFunc(img(:,:,idx)); % pre-processing
	trainingFeatures(:,idx) = featExtrFunc(bw, params{:});
end

%% training

Model = Classifier(trainingFeatures, trainingLabels, 0:9);
save('Model.mat','Model');