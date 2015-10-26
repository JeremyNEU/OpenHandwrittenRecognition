
# 数据集的读取

文件(第三方)：loadMNISTImages.m loadMNISTLabels.m display_network.m

数据集读取实例

```matlab
% assume that the path of mnist dataset is `dataset/mnist/`

%% load dataset
trainingData = loadMNISTImages('dataset/mnist/train-images.idx3-ubyte');
trainingLabels = loadMNISTLabels('dataset/mnist/train-labels.idx1-ubyte');

%% display former 100 images labeled 8
display_network(trainingData(:,find(trainingLabels==8,100)'));
```

生成测试图片，为方便在编码过程中测试，取出数据集中部分图片，注意一维的图片数据需要转为二维的图片（28*28）

```matlab
%% output 50 images for testing feature extraction 

N = length(trainingData);
img = reshape(trainingData,28,28,N);

for idx = 1:50
    imwrite(img(:,:,idx), sprintf('test/%d(%d).jpg', idex,labels(idx)));
end
```

# 特征提取函数

特征提取函数的格式为`function [featureVector, visualization] = extractXXXFeatures(img, params)`，以粗网格特征为例：

文件：extractCoarseMeshFeatures.m

function [featureVector, visualization] = extractCoarseMeshFeatures(img,meshSize,blockSize)

粗网格特征反映了空间的分布统计性，将图像划分为多个网格，统计各个网格中的像素值的和。为了统一各种尺寸的图片，首先进行图像大小的调整。称每个格子为一个图像块，给出块的大小（一个图像块含几行几列像素）和网格的大小（一副图像横纵各含几个网格）则图像调整的目标尺寸就知道了。由于手写数字集中在图片中央，周围的背景像素对粗网格提取是无意义的，所以再调整图像尺寸前先裁剪掉这些区域。裁剪的方法是计算出上下左右的边界。

输入为一个经过预处理的图片img（可以是灰度图像，也可以是黑白图像）以及网格和块的大小meshSize、blockSize；
输出为提取出的特征向量featureVector以及该特征的可视化图像visualization

由于数字的行的信息比列的分布信息更加丰富（数码管显示数字三行两列），因此网格的行数应当大于列数；另一方面，网格如果取得过多，由于手写的随机性造成的分布漂移不能够聚合，统计特征不能较好的表现，而网格取得过少则不具备区分性。综合考虑，典型网格数目取值是5×3，由于数据集图像尺寸为28×28，所以块大小取6×10(调整后的目标尺寸为30×30，比较接近28×28)

visualization 将特征可视化，方便理解和调试：

```matlab
[featureVector,visualization] = extractXXXFeatures(image);

figure,
subplot(121), imshow(image);
subplot(122), imshow(visualization);
```

实际应用时，不需要 visualization，`featureVector = extractXXXFeatures(image)`

# 分类器设计

文件：BayesClassifier.m

分类器通过matlab对象实现，该对象需要实现两个方法：

1. 构造方法：function model = XXXClassifier(trainingFeatures, trainingLabels, labelsSet) 通过训练集的特征向量trainingFeatures(#DimOfFeatureVector * #TrainingSamples)和相应类别trainingLabels(1 * #TrainingSamples)以及类别的集合labelsSet(比如手写数字的labelsSet为0到9)；
2. 分类：function labels = predict(model, features, prioriProbabilities) 根据训练出的模型，给出某个样本的特征features以及其先验概率prioriProbabilities得出该样本的类别。

调用实例：

```matlab
% training
trainingFeatures = featureExtraction(trainingData);
model = BayesClassifier(trainingFeatures, trainingLabels, 0:9);

% use model to label a sample
features = featureExtraction(sample);
labels = model.predict(features, prioriProbabilities);
```

# 训练、测试、结果显示和评估

文件：benchmark.m

benchmark(algorithm, dataset)

输入

algorithm结构体

* algorithm.name 算法名称，用于生成的模型存入文件的命名以及加载模型文件；
* algorithm.featureExtractor 特征提取函数，输入为图像，输出为特征向量，预处理工作需在函数内部完成，调用前不会对图像作任何处理；
* algorithm.classifier 算法采用的分类器

dataset结构体

* dataset.trainingData 用于训练的数据，每个样本的图像数据是一维的
* dataset.trainingLabels 训练数据的标定
* dataset.testData 用于测试的数据
* dataset.testLabels 测试数据的标定

输出

如果没有输出参数，则输出调试信息到命令窗口，否则输出混淆矩阵。调试信息包括：混淆百分比矩阵，错分样本

```matlab
% benchmark example
dataset.trainingData = loadMNISTImages('dataset/mnist/train-images.idx3-ubyte');
dataset.trainingLabels = loadMNISTLabels('dataset/mnist/train-labels.idx1-ubyte');
dataset.testData = loadMNISTImages('dataset/mnist/t10k-images.idx3-ubyte');
dataset.testLabels = loadMNISTLabels('dataset/mnist/t10k-labels.idx1-ubyte');

algorithm.name = 'Otsu-AdjustBound-CoarseMesh';
im2bwOTSU = @(img)(im2bw(img, graythresh(img)));
algorithm.featureExtractor = @(img)extractCoarseMeshFeatures(im2bwOTSU(img),[5,3],[6,10]);
algorithm.classifier = @BayesClassifier;

benchmark(algorithm, dataset);
```
