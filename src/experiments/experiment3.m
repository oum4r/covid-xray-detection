function experiment3(imdsTrain, imdsVal, imdsTest, YTest)
    % Experiment 3: Transfer Learning with VGG19

    % Balance dataset by oversampling smaller classes
    imdsTrainBalanced = balanceDataset(imdsTrain);

    % Define augmentation
    aug = augmentData("exp2_3");

    inputSize = [224 224 3]; % Standard for VGG19
    trainAug = augmentedImageDatastore(inputSize, imdsTrainBalanced, ...
        'DataAugmentation', aug, 'ColorPreprocessing', 'gray2rgb');
    valAug   = augmentedImageDatastore(inputSize, imdsVal, 'ColorPreprocessing', 'gray2rgb');
    testAug  = augmentedImageDatastore(inputSize, imdsTest, 'ColorPreprocessing', 'gray2rgb');

    % Load pretrained VGG19
    fprintf("Loading VGG19 Model...\n");
    vgg = vgg19();
    fprintf("Loaded VGG19 Model\n");

    % Modify architecture
    layers = [
        vgg.Layers(1:end-3) % Keep all except last 3
        fullyConnectedLayer(256)
        reluLayer
        dropoutLayer(0.5)
        fullyConnectedLayer(128)
        reluLayer
        dropoutLayer(0.5)
        fullyConnectedLayer(4) % Four classes
        softmaxLayer
        classificationLayer];

    % Training options
    options = trainingOptions("adam", ...
        "MiniBatchSize",32, ...
        "MaxEpochs",15, ...
        "InitialLearnRate",1e-4, ...
        "LearnRateSchedule", "piecewise", ...
        "LearnRateDropFactor", 0.8, ...
        "LearnRateDropPeriod", 5, ...
        "Shuffle", "every-epoch", ...
        "ValidationData", valAug, ...
        "ValidationFrequency", 50, ...
        "Verbose", true, ...
        "Plots", "training-progress");

    rng(0);
    fprintf('Beginning Experiment 3 Training\n');
    net = trainNetwork(trainAug, layers, options);

    % Evaluate
    YPred = classify(net, testAug);
    accuracy = sum(YPred == YTest)/numel(YTest);
    fprintf('Experiment 3 Accuracy (VGG19): %.2f%%\n\n', accuracy*100);
    confusionchart(YTest, YPred, Normalization="row-normalized");
end
