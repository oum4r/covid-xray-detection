%% COVID-19 X-Ray Detection System
% Entry point script
clearvars; close all; clc;

% Load dataset
[imdsTrain, imdsVal, imdsTest, YTest] = loadData('..\data');

% Run Experiments
fprintf('Running Experiment 1: Baseline CNN...\n');
experiment1(imdsTrain, imdsVal, imdsTest, YTest);

fprintf('Running Experiment 2: Balanced CNN...\n');
experiment2(imdsTrain, imdsVal, imdsTest, YTest);

fprintf('Running Experiment 3: Transfer Learning (VGG19)...\n');
experiment3(imdsTrain, imdsVal, imdsTest, YTest);
