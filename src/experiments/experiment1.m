function experiment1(imdsTrain, imdsVal, imdsTest, YTest)
    % Experiment 1: Baseline CNN
    aug = augmentData("exp1");

    inputSize = [128 128 1];
    trainAug = augmentedImageDatastore(inputSize, imdsTrain, 'DataAugmentation', aug, "ColorPreprocessing","rgb2gray");
    valAug   = augmentedImageDatastore(inputSize, imdsVal, "ColorPreprocessing","rgb2gray");
    testAug  = augmentedImageDatastore(inputSize, imdsTest, "ColorPreprocessing","rgb2gray");

    processTrain = transform(trainAug, @preProcessImages, "IncludeInfo", true);
    processVal   = transform(valAug, @preProcessImages, "IncludeInfo", true);
    processTest  = transform(testAug, @preProcessImages, "IncludeInfo", true);

    % CNN architecture
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
        "ValidationData",processVal, ...
        "ValidationFrequency",132, ...
        "Verbose", true, ...
        "Plots", "training-progress");

    rng(0);
    fprintf('Beginning Experiment 1 Training');
    net = trainNetwork(processTrain, layers, options);

    % Evaluate
    YPred = classify(net, processTest);
    accuracy = sum(YPred == YTest)/numel(YTest);
    fprintf('Experiment 1 Accuracy: %.2f%%\n\n', accuracy*100);
    confusionchart(YTest, YPred, Normalization="row-normalized");
end
