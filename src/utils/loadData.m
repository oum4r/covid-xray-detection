function [imdsTrain, imdsVal, imdsTest, YTest] = loadData(directory)
    % Load dataset into MATLAB datastores and split
    imds = imageDatastore(fullfile(directory), ...
        FileExtensions=".png", ...
        LabelSource="foldernames", ...
        IncludeSubfolders=true);
    fprintf("Data Loaded from directory.\n")
    rng(0);
    [imdsTrain, imdsVal, imdsTest] = splitEachLabel(imds, 0.8, 0.1, 0.1, 'randomized');
    YTest = imdsTest.Labels;
end
