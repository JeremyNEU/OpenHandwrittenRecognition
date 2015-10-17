meshSize = [5,3];
blockSize = [6,10];
binaryFunc = @(img)(im2bw(img, graythresh(img))); %Otsu
featExtrFunc = @(img)(extractCoarseMeshFeatures(binaryFunc(img), ...
    meshSize, blockSize));
featSize = meshSize(1)*meshSize(2);

load('Model.mat');
testModel(Model, featExtrFunc, featSize);