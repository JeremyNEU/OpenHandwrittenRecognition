function confMat = benchmark(algorithm, dataset)
%benchmark test a feature extraction function 

global MODEL_FOLD_NAME;

    %% show feature extraction example

    image = reshape(dataset.trainingData(:,1),28,28);
    featureVector = algorithm.featureExtractor(image);

    featSize = length(featureVector);

%     if nargout == 0
%         figure,
%         subplot(121), imshow(image);
%         subplot(122), imshow(visualization);
%     end
    
    %% feature extraction
    function features = helperExtractFeaturesFromDataSet(data)
        N = length(data);
        images = reshape(data,28,28,N);

        features = zeros(featSize,N);
        for n = 1:N
            features(:,n) = algorithm.featureExtractor(images(:,:,n));
        end
    end


    %% training

    modelFileName = [MODEL_FOLD_NAME '/' algorithm.name '.mat'];

    if exist(modelFileName, 'file')
%         disp(['load:' modelFileName]);
        load(modelFileName, '-mat'); % model = load(modelFileName, '-mat') will convert a obj to struct
    else
        trainingFeatures = helperExtractFeaturesFromDataSet(dataset.trainingData);

        model = algorithm.classifier(trainingFeatures, dataset.trainingLabels, 0:9);
        % save training result to file.
        save(modelFileName,'model');
    end

    %% test
    
    testFeatures = helperExtractFeaturesFromDataSet(dataset.testData);
    predictedLabels = model.predict(testFeatures);

    %% Evaluation
    
    % Tabulate the results using a confusion matrix.
    confMat = confusionmat(dataset.testLabels, predictedLabels);
	
	% display performance evaluation
	
	if nargout > 0, return; end

    % Convert confusion matrix into percentage form
    confRateMat = bsxfun(@rdivide,confMat,sum(confMat,2));

    fprintf('Confusion Matrix\n');
    % digit  | Recog Results           
    % ------------------------
    % 0:9    | Percentage(Num)

    digits = '0':'9';
    colHeadings = arrayfun(@(x)sprintf('%d',x),0:9,'UniformOutput',false);
    format = repmat('%-9s',1,11);
    header = sprintf(format,'digit  |',colHeadings{:});
    fprintf('\n%s\n%s\n',header,repmat('-',size(header)));
    for idx = 1:numel(digits)
        fprintf('%-9s',   [digits(idx) '      |']);
        fprintf('%-9.2f', confRateMat(idx,:));
        fprintf('\n')
    end

    trueRecog = diag(confRateMat);
    falseRecog = confRateMat.*~eye(10); % or confMat - diag(trueRecog)

    %% visualize confMat
    
    figure;
    subplot(121);imshow(mat2gray(falseRecog));
    subplot(122);bar(0:9,trueRecog);

    ErrorRate = 1 - mean(trueRecog);
    fprintf('Error Rate:%.4f%%\n', ErrorRate);

    %% display false recognition
    labelSet = 0:9;
    for idx = 1:length(confMat)
        digit = labelSet(idx);
        figure('NumberTitle', 'off', 'Name', ...
                sprintf('%s:false recog of %d', algorithm.name, digit));
        isFalseRecog = confMat(idx,:)~=0; 
        isFalseRecog(idx) = 0; % 
        nPlot = sum(isFalseRecog);
        nPlotRow = floor(sqrt(nPlot));
        nPlotCol = ceil(nPlot/nPlotRow);
        no = 1;
        for falseDigit = labelSet(isFalseRecog) % j != i false recognition
            subplot(nPlotRow,nPlotCol,no);
            display_network(dataset.testData(:, ...
                find(dataset.testLabels==digit & predictedLabels==falseDigit)'));
            title(sprintf('%d->%d',digit,falseDigit));
            no = no + 1;
        end
    end

end
