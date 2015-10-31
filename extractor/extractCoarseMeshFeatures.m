function [featureVector, visualization] = extractCoarseMeshFeatures(img,meshSize,blockSize)
% Note img is 2d
% extractCoarseMeshFeatures(img1,[6,3],[5,9]) % nRow, nCol

%% min bound box

sumRow = sum(img,1); % one row
sumCol = sum(img,2); % one column

cMin = find(sumRow>0,1);
cMax = find(sumRow>0,1,'last');
rMin = find(sumCol>0,1);
rMax = find(sumCol>0,1,'last');

aligned = img(rMin:rMax,cMin:cMax);
resized = imresize(aligned,meshSize.*blockSize);

countBlock = @(block_struct) sum(sum(block_struct.data));
countResult = blockproc(resized,blockSize,countBlock);

featureVector = countResult(:); % column vector

if nargout > 1
% figure, 
% subplot(2,2,1);imshow(img);
% subplot(2,2,2);imshow(aligned);
% subplot(2,2,3);imshow(resized);
% subplot(2,2,4);imshow(mat2gray(countResult));
    visualization = mat2gray(countResult);
end

