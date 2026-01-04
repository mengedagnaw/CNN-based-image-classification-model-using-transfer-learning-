%% transfer_learning_googlenet_sdss2507.m
% Transfer Learning (GoogLeNet) for Microstructure Classification
% Classes: Asprinted vs SolutionAnnealed
% Follows the Lab 2 technical report workflow. :contentReference[oaicite:1]{index=1}
%
% Expected dataset structure:
% microstructure_classification/SimpleData/Asprinted/*.png (or jpg)
% microstructure_classification/SimpleData/SolutionAnnealed/*.png (or jpg)
%
% Outputs saved to:
% microstructure_classification/outputs/

clear; close all; clc;
rng(1); % reproducibility

%% ---- Repo-relative paths ----
scriptDir = fileparts(mfilename('fullpath'));
repoRoot  = fullfile(scriptDir, "..");

dataDir = fullfile(repoRoot, "microstructure_classification", "SimpleData");
outDir  = fullfile(repoRoot, "microstructure_classification", "outputs");
if ~exist(outDir, "dir"), mkdir(outDir); end

if ~exist(dataDir, "dir")
    error("Dataset folder not found: %s\nCreate microstructure_classification/SimpleData/...", dataDir);
end

%% Step 1: Load Dataset (imageDatastore)
imds = imageDatastore(dataDir, ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

disp("=== Dataset summary ===");
disp(countEachLabel(imds));

%% Step 2: Balance + Split (80/20)
tbl = countEachLabel(imds);
minSetCount = min(tbl.Count);

imdsBalanced = splitEachLabel(imds, minSetCount, "randomized");
[imdsTrain, imdsValidation] = splitEachLabel(imdsBalanced, 0.8, "randomized");

disp("=== After balancing ===");
disp(countEachLabel(imdsBalanced));
disp("=== Train / Validation ===");
disp(countEachLabel(imdsTrain));
disp(countEachLabel(imdsValidation));

%% Step 3: Preprocess Images (resize + gray2rgb)
inputSize = [224 224 3]; % GoogLeNet input
augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    "ColorPreprocessing", "gray2rgb");

augimdsValidation = augmentedImageDatastore(inputSize, imdsValidation, ...
    "ColorPreprocessing", "gray2rgb");

%% Step 4: Modify GoogLeNet for 2-class classification
% Requires: Deep Learning Toolbox + GoogLeNet support package installed
net = googlenet;
lgraph = layerGraph(net);

% Remove final classification layers
lgraph = removeLayers(lgraph, {"loss3-classifier","prob","output"});

numClasses = numel(categories(imdsTrain.Labels)); % should be 2

newLayers = [
    fullyConnectedLayer(numClasses, "Name","new_fc", ...
        "WeightLearnRateFactor",10, "BiasLearnRateFactor",10)
    softmaxLayer("Name","new_softmax")
    classificationLayer("Name","new_output")];

lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, "pool5-drop_7x7_s1", "new_fc");

%% Step 5: Training options (SGDM)
options = trainingOptions("sgdm", ...
    "MiniBatchSize", 10, ...
    "MaxEpochs", 10, ...
    "InitialLearnRate", 1e-4, ...
    "ValidationData", augimdsValidation, ...
    "ValidationFrequency", 5, ...
    "Verbose", false, ...
    "Plots", "training-progress");

%% Step 6: Train
[trainedNet, trainInfo] = trainNetwork(augimdsTrain, lgraph, options);

% Save model
stamp = datestr(now, "yyyymmdd_HHMMSS");
save(fullfile(outDir, "trainedGoogLeNet_" + stamp + ".mat"), "trainedNet", "trainInfo");

%% Step 7: Evaluate accuracy
YPred = classify(trainedNet, augimdsValidation);
YValidation = imdsValidation.Labels;

accuracy = mean(YPred == YValidation);
fprintf("\nValidation Accuracy: %.2f%%\n", accuracy * 100);

% Confusion matrix (nice and modern)
fig1 = figure("Name","Confusion Matrix");
confusionchart(YValidation, YPred);
title(sprintf("Confusion Matrix (Accuracy = %.2f%%)", accuracy*100));
exportgraphics(fig1, fullfile(outDir, "confusion_matrix_" + stamp + ".png"));

%% Step 8: Visualize sample predictions (12)
numImages = min(12, numel(imdsValidation.Files));
perm = randperm(numel(imdsValidation.Files), numImages);

fig2 = figure("Name","Sample Predictions");
for i = 1:numImages
    subplot(3,4,i);

    img = readimage(imdsValidation, perm(i));

    % Resize + ensure 3 channels
    resizedImg = imresize(img, inputSize(1:2));
    if size(resizedImg,3) == 1
        resizedImg = repmat(resizedImg, [1 1 3]);
    end

    predLabel = classify(trainedNet, resizedImg);
    trueLabel = imdsValidation.Labels(perm(i));

    imshow(img);
    title(string(predLabel) + " / " + string(trueLabel), "FontSize", 8);
end
sgtitle("Sample Predictions (Predicted / True)");
exportgraphics(fig2, fullfile(outDir, "sample_predictions_" + stamp + ".png"));

%% Step 9: Classify a new image (UI file picker)
[fileName, pathName] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp'}, "Select an image to classify");

if isequal(fileName, 0)
    disp("No file selected. Skipping single-image classification.");
else
    img = imread(fullfile(pathName, fileName));
    img2 = im2gray(img);

    resizedImg = imresize(img2, inputSize(1:2));
    resizedImg = repmat(resizedImg, [1 1 3]);

    [pred, scores] = classify(trainedNet, resizedImg);

    fig3 = figure("Name","Single Image Prediction");
    imshow(img2);
    title("Prediction: " + string(pred), "FontSize", 12);
    exportgraphics(fig3, fullfile(outDir, "single_prediction_" + stamp + ".png"));

    fig4 = figure("Name","Classification Confidence");
    bar(scores);
    ylim([0 1]);
    xticklabels(trainedNet.Layers(end).Classes);
    ylabel("Confidence");
    xlabel("Class");
    title("Classification Confidence");
    exportgraphics(fig4, fullfile(outDir, "confidence_bar_" + stamp + ".png"));
end

disp("Done. Results saved in: " + outDir);
