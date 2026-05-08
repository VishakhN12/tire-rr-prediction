clc;
clear;
close all;

%% 1: LOAD DATA
data = readtable('Tire_Data.xlsx');

disp("Column Names:");
disp(data.Properties.VariableNames')

%% 2: NUMERIC CONVERSION
toNum = @(x) str2double(string(x));

%% 3: BASIC INPUT FEATURES (NO REDUNDANCY)

weight = toNum(data.TYRE_WEIGHT_KG);
load   = toNum(data.MAX_LOAD_SINGLE_KG);
psi    = toNum(data.PSI);

land = toNum(data.LAND_100);
sea  = toNum(data.SEA_100);

arc_b2 = toNum(data.ARC_WIDTH_B2);
arc_b3 = toNum(data.ARC_WIDTH_B3);

%% 4: PHYSICS-BASED FEATURES (IMPORTANT FIX)

OD_Growth = toNum(data.INFLATED_OD_CROWN) - toNum(data.UNINFLATED_OD_CROWN);
SW_Growth = toNum(data.INFLATED_SW) - toNum(data.UNINFLATED_SW);

%% 5: FINAL FEATURE MATRIX (CLEAN & NON-REDUNDANT)

X = [
    weight,...
    load,...
    psi,...
    OD_Growth,...
    SW_Growth,...
    land,...
    sea,...
    arc_b2,...
    arc_b3
];

%% 6: TARGET VARIABLE
Y = toNum(data.RR_80KMPH);

%% 7: REMOVE INVALID ROWS (VERY IMPORTANT)

validRows = all(~isnan(X),2) & ~isnan(Y);

X = X(validRows,:);
Y = Y(validRows,:);

%% 8: NORMALIZATION
X = normalize(X);

%% 9: TRAIN-TEST SPLIT

n = size(X,1);
idx = randperm(n);

trainRatio = 0.8;
nTrain = round(trainRatio*n);

trainIdx = idx(1:nTrain);
testIdx  = idx(nTrain+1:end);

XTrain = X(trainIdx,:);
XTest  = X(testIdx,:);

YTrain = Y(trainIdx,:);
YTest  = Y(testIdx,:);

%% 10: LINEAR MODEL - BASELINE

lmModel = fitlm(XTrain,YTrain);

YPred_LM = predict(lmModel,XTest);

RMSE_LM = sqrt(mean((YTest - YPred_LM).^2));
R2_LM = 1 - sum((YTest - YPred_LM).^2) / sum((YTest - mean(YTest)).^2);

disp("Linear RMSE: " + RMSE_LM);
disp("Linear R2: " + R2_LM);

%% 11: ENSEMBLE MODEL

rfModel = fitrensemble(XTrain,YTrain,...
    'Method','Bag',...
    'NumLearningCycles',300);

YPred_RF = predict(rfModel,XTest);

RMSE_RF = sqrt(mean((YTest - YPred_RF).^2));
R2_RF = 1 - sum((YTest - YPred_RF).^2) / sum((YTest - mean(YTest)).^2);

disp("Ensemble RMSE: " + RMSE_RF);
disp("Ensemble R2: " + R2_RF);

%% 12: RESULTS VISUALIZATION

figure;
scatter(YTest, YPred_RF, 'filled');
grid on;
xlabel('Actual RR');
ylabel('Predicted RR');
title('Final Clean Model: Actual vs Predicted RR');

%% 13: PSI vs RR

figure;
scatter(psi(validRows), Y, 'filled');
grid on;
xlabel('PSI');
ylabel('Rolling Resistance');
title('PSI vs RR');

%% 14: SAVE MODEL

save('RR_Model_FINAL_CLEAN.mat','lmModel','rfModel');
