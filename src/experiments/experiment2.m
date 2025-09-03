function experiment2(imdsTrain, imdsVal, imdsTest, YTest)
    % Experiment 2: Balanced CNN with stronger augmentation

    % Balance dataset by oversampling smaller classes
    imdsTrainBalanced = balanceDataset(imdsTrain);

    % Define augmentation
    aug = augmentData("exp2_3");

    inputSize = [128 128 1];
    trainAug = augmentedImageDatastore(inputSize, imdsTrainBalanced, ...
        'DataAugmentation', aug, 'ColorPreprocessing', 'rgb2gray');
    valAug   = augmentedImageDatastore(inputSize, imdsVal, 'ColorPreprocessing', 'rgb2gray');
    testAug  = augmentedImageDatastore(inputSize, imdsTest, 'ColorPreprocessing', 'rgb2gray');

    % Reuse baseline CNN architecture from Experiment 1
    layers = [
        imageInputLayer([128 128 1])
        convolution2dLayer(3,8,"Padding","same")
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,"Stride",2)
        convolution2dLayer(3,16,"Padding","same")
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,"Stride",2)
        convolution2dLayer(3,32,"Padding","same")
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,"Stride",2)
        convolution2dLayer(3,64,"Padding","same")
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,"Stride",2)
        convolution2dLayer(3,128,"Padding","same")
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(4)
        softmaxLayer
        classificationLayer];

    % Training options
    options = trainingOptions("adam", ...
        "MiniBatchSize",128, ...
        "MaxEpochs",20, ...
        "InitialLearnRate",0.001, ...
        "Shuffle","every-epoch", ...
        "ValidationData",valAug, ...
        "ValidationFrequency",132, ...
        "Verbose", true, ...
        "Plots", "training-progress");

    rng(0);
    fprintf('Beginning Experiment 2 Training');
    net = trainNetwork(trainAug, layers, options);

    % Evaluate
    YPred = classify(net, testAug);
    accuracy = sum(YPred == YTest)/numel(YTest);
    fprintf('Experiment 2 Accuracy (Balanced CNN): %.2f%%\n\n', accuracy*100);
    confusionchart(YTest, YPred, Normalization="row-normalized");
end
