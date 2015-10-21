%% benchmarkResults

% % Original
% meshSize = [5,3];
% blockSize = [6,10];
% binaryFunc = @(img)(im2bw(img, graythresh(img))); %Otsu
% featExtrFunc = @(img)(extractCoarseMeshFeatures(binaryFunc(img), ...
%     meshSize, blockSize));
% featSize = meshSize(1)*meshSize(2);

% Optimized

% binary
im2bwOTSU = @(img)(im2bw(img, graythresh(img)));
unpadding = @(img)(img(5:24,5:24));

% Mesh feature
countBlock = @(block_struct) sum(sum(block_struct.data));
MeshfeatExtr = @(img,blockSize)(blockproc(img,blockSize,countBlock));

% convert to feature vector.
convert2Array = @(img)(img(:));

featExtrFuncs = {...
% @(img)convert2Array(MeshfeatExtr(unpadding(img), [4 7])), ...
% @(img)convert2Array(MeshfeatExtr(im2bwOTSU(unpadding(img)), [4 7])), ...
% @(img)convert2Array(MeshfeatExtr((unpadding(img)>0), [4 7])), ...
% ErrorRate =
% 
%     0.1052
%     0.1117
%     0.0968
@(img)extractCoarseMeshFeatures(im2bwOTSU(img),[5,3],[6,10]), ...
@(img)extractCoarseMeshFeatures(img>0,[5,3],[6,10]), ...
@(img)extractCoarseMeshFeatures(img>0,[5,3],[4,7]) ...
% ErrorRate =
% 
%     0.0702
%     0.0717
%     0.0731
};

[confMat, trainingTime, testingTime] = benchmark(featExtrFuncs);

N = length(featExtrFuncs);
trueRecog = zeros(N,10);
ErrorRate = zeros(N,1);

for n = 1:N
	confMatDiag = diag(confMat{n});
	trueRecog(n,:) = confMatDiag;
	ErrorRate(n) = 1 - mean(confMatDiag);
end

figure, 
subplot(131), plot(trainingTime);
subplot(132), plot(testingTime);
subplot(133), plot(ErrorRate);
