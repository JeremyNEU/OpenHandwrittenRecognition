%% output images for testing feature extraction 
data = loadMNISTImages('dataset/mnist/train-images.idx3-ubyte');
labels = loadMNISTLabels('dataset/mnist/train-labels.idx1-ubyte');

N = length(data);
img = reshape(data,28,28,N);

% for idx = 1:40
%     imwrite(img(:,:,idx),num2str(labels(idx), '%d.jpg'));
% end

meshSize = [5,3];
blockSize = [6,10];
binaryFunc = @(img)(im2bw(img, graythresh(img))); %Otsu
featExtrFunc = @(img)(extractCoarseMeshFeatures(binaryFunc(img), ...
    meshSize, blockSize));
featSize = meshSize(1)*meshSize(2);

load('Model.mat');
confMat = testModel(Model, featExtrFunc, featSize);


trueRecog = diag(confMat);
falseRecog = confMat.*~eye(10); % or confMat - diag(trueRecog)

figure;
subplot(121);imshow(mat2gray(falseRecog));
subplot(122);bar(0:9,trueRecog);
