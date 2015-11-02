setup;

dataset.trainingData = loadMNISTImages('dataset/mnist/train-images.idx3-ubyte');
dataset.trainingLabels = loadMNISTLabels('dataset/mnist/train-labels.idx1-ubyte');
dataset.testData = loadMNISTImages('dataset/mnist/t10k-images.idx3-ubyte');
dataset.testLabels = loadMNISTLabels('dataset/mnist/t10k-labels.idx1-ubyte');

algorithm.name = 'Otsu-AdjustBound-CoarseMesh';
im2bwOTSU = @(img)(im2bw(img, graythresh(img)));
algorithm.featureExtractor = @(img)extractCoarseMeshFeatures(im2bwOTSU(img),[5,3],[6,10]);
algorithm.classifier = @BayesClassifier;

benchmark(algorithm, dataset);