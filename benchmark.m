function [confMat, trainingTime, testingTime] = benchmark(featExtrFuncs)

N = length(featExtrFuncs);
confMat = cell(N,1);
trainingTime = zeros(N,1);
testingTime = zeros(N,1);

for n = 1:N
  
    featExtrFunc = featExtrFuncs{n};
    test0 = imread('test/0.jpg');
    result0 = featExtrFunc(test0);
    featSize = length(result0(:));

    % process time includes load-data time.

    disp('Training...');

    tic
    model = trainModel(featExtrFunc, featSize);
    trainingTime(n) = toc;

    disp('Testing...');

    tic
    confMat{n} = testModel(model, featExtrFunc, featSize);
    testingTime(n) = toc;

%     trueRecog = diag(confMat);
%     falseRecog = confMat.*~eye(10); % or confMat - diag(trueRecog)
% 
%     figure;
%     subplot(121);imshow(mat2gray(falseRecog));
%     subplot(122);bar(0:9,trueRecog);

%     digits = '0':'9';
%     colHeadings = arrayfun(@(x)sprintf('%d',x),0:9,'UniformOutput',false);
%     format = repmat('%-9s',1,11);
%     header = sprintf(format,'digit  |',colHeadings{:});
%     fprintf('\n%s\n%s\n',header,repmat('-',size(header)));
%     for idx = 1:numel(digits)
%         fprintf('%-9s',   [digits(idx) '      |']);
%         fprintf('%-9.2f', confMat(idx,:));
%         fprintf('\n')
%     end
    
end