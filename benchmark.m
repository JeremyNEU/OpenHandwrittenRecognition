% % Original
% meshSize = [5,3];
% blockSize = [6,10];
% binaryFunc = @(img)(im2bw(img, graythresh(img))); %Otsu
% featExtrFunc = @(img)(extractCoarseMeshFeatures(binaryFunc(img), ...
%     meshSize, blockSize));
% featSize = meshSize(1)*meshSize(2);

% Optimized
countBlock = @(block_struct) sum(sum(block_struct.data));
% binaryFunc = @(img)(im2bw(img, graythresh(img))); %Otsu
imgProc = @(img)(blockproc(img(5:24,5:24),[4 7],countBlock));
convert2Array = @(img)(img(:));
%1:28 -> 5:24 unpadding, no binary
featExtrFunc = @(img)(convert2Array(imgProc(img)));

test0 = imread('test/0.jpg');
result0 = featExtrFunc(test0);
featSize = length(result0(:));

% process time includes load-data time.

disp('Training...');

tic
model = trainModel(featExtrFunc, featSize);
toc

disp('Testing...');

tic
confMat = testModel(model, featExtrFunc, featSize);
toc

trueRecog = diag(confMat);
falseRecog = confMat.*~eye(10); % or confMat - diag(trueRecog)

figure;
subplot(121);imshow(mat2gray(falseRecog));
subplot(122);bar(0:9,trueRecog);

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