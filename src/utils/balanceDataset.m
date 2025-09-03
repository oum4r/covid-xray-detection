function imdsBalanced = balanceDataset(imdsTrain)
    % Balance dataset by oversampling underrepresented classes
    labelCounts = countEachLabel(imdsTrain);
    [maxCount, ~] = max(labelCounts.Count);
    classNames = categories(imdsTrain.Labels);
    numClasses = length(classNames);

    classIndices = cell(numClasses, 1);
    for i = 1:numClasses
        classIndices{i} = find(imdsTrain.Labels == classNames{i});
    end

    balancedIndices = [];
    balancedLabels = [];

    for i = 1:numClasses
        currentIndices = classIndices{i};
        currentCount = length(currentIndices);

        if currentCount < maxCount
            samplingIndices = randi(currentCount, maxCount, 1);
            balancedIndices = [balancedIndices; currentIndices(samplingIndices)];
            balancedLabels = [balancedLabels; repmat(classNames(i), maxCount, 1)];
        else
            balancedIndices = [balancedIndices; currentIndices];
            balancedLabels = [balancedLabels; repmat(classNames(i), length(currentIndices), 1)];
        end
    end

    balancedFiles = imdsTrain.Files(balancedIndices);
    imdsBalanced = imageDatastore(balancedFiles, 'Labels', categorical(balancedLabels));
end
