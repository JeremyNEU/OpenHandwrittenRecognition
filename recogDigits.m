function label = recogDigits(img)

% %% load dataset
% testData = loadMNISTImages('dataset/mnist/t10k-images.idx3-ubyte');
% testLabels = loadMNISTLabels('dataset/mnist/t10k-labels.idx1-ubyte');
% 
% N = length(testData);
% img = reshape(testData,28,28,N);
% 
% img1 = img(:,:,1);

%% configuration

binaryFunc = @(img)(im2bw(img, graythresh(img))); %Otsu
featExtrFunc = @extractCoarseMeshFeatures;
meshSize = [5,3];
blockSize = [6,10];
params = {meshSize,blockSize};

%% feature extraction

bw = binaryFunc(img); % pre-processing
featureVector = featExtrFunc(bw, params{:});


%% predict
load('Model.mat');
label = Model.predict(featureVector);