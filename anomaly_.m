

% An autoencoder is a type of model that is trained to replicate its input by transforming the input to a lower 
% dimensional space (the encoding step) and reconstructing the input from the lower dimensional representation 
% (the decoding step). Training an autoencoder does not require labeled data.

% An autoencoder itself does not detect anomalies. Training an autoencoder using only representative data yields 
% a model that can reconstruct its input data by using features learned from the representative data only.


clear all; close all;


%% Load Training Data
load WaveformData

numChannels = size(data{1},2)

figure
tiledlayout(2,2)
for i = 1:4
    nexttile
    stackedplot(data{i},DisplayLabels="Channel " + (1:numChannels));
    title("Observation " + i)
    xlabel("Time Step")
end

numObservations = numel(data);
XTrain = data(1:floor(0.9*numObservations));
XValidation = data(floor(0.9*numObservations)+1:end);

%% Prepare Data for Training

numDownsamples = 2;

sequenceLengths = zeros(1,numel(XTrain));

for n = 1:numel(XTrain)
    X = XTrain{n};
    if n < 4
        fprintf('original size %s\n', num2str(size(X)))
    end
    cropping = mod(size(X,1), 2^numDownsamples);
    X(end-cropping+1:end,:) = [];
    XTrain{n} = X;
    sequenceLengths(n) = size(X,1);
end
fprintf('downsampling size %s\n', num2str(size(XTrain{1})))

for n = 1:numel(XValidation)
    X = XValidation{n};
    cropping = mod(size(X,1),2^numDownsamples);
    X(end-cropping+1:end,:) = [];
    XValidation{n} = X;
end

%% Define Network Architecture

minLength = min(sequenceLengths);
filterSize = 7;
numFilters = 16;
dropoutProb = 0.2;

layers = sequenceInputLayer(numChannels,Normalization="zscore",MinLength=minLength);

for i = 1:numDownsamples
    layers = [
        layers
        convolution1dLayer(filterSize,(numDownsamples+1-i)*numFilters,Padding="same",Stride=2)
        reluLayer
        dropoutLayer(dropoutProb)];
end

for i = 1:numDownsamples
    layers = [
        layers
        transposedConv1dLayer(filterSize,i*numFilters,Cropping="same",Stride=2)
        reluLayer
        dropoutLayer(dropoutProb)];
end

layers = [
    layers
    transposedConv1dLayer(filterSize,numChannels,Cropping="same")];

deepNetworkDesigner(layers)

%% Specify Training Options
options = trainingOptions("adam", ...
    MaxEpochs=120, ...
    Shuffle="every-epoch", ...
    ValidationData={XValidation,XValidation}, ...
    Verbose=1, ...
    Plots="training-progress");


%% Train Network
% When you train an autoencoder, the inputs and targets are the same. For regression, use mean squared error loss

net = trainnet(XTrain,XTrain,layers,"mse",options);


%% Test Network

YValidation = minibatchpredict(net,XValidation);

for n = 1:numel(XValidation)
    T = XValidation{n};
    
    sequenceLength = size(T,1);
    Y = YValidation(1:sequenceLength,:,n);
    
    err(n) = rmse(Y,T,"all");
end

figure
histogram(err)
xlabel("Root Mean Square Error (RMSE)")
ylabel("Frequency")
title("Representative Samples")

RMSEbaseline = max(err)


%% Identify Anomalous Sequences
% reate a new set of data by manually editing some of the validation sequences to have anomalous regions.

XNew = XValidation;

numAnomalousSequences = 20;
idx = randperm(numel(XValidation),numAnomalousSequences);

for i = 1:numAnomalousSequences
    X = XNew{idx(i)};

    idxPatch = 50:60;
    XPatch = X(idxPatch,:);
    X(idxPatch,:) = 4*abs(XPatch);

    XNew{idx(i)} = X;
end

YNew = minibatchpredict(net,XNew);

% For each prediction, calculate the RMSE between the input sequence and the reconstructed sequence.
errNew = zeros(numel(XNew),1);
for n = 1:numel(XNew)
    T = XNew{n};
    
    sequenceLength = size(T,1);
    Y = YNew(1:sequenceLength,:,n);
    
    errNew(n) = rmse(Y,T,"all");
end

figure
histogram(errNew)
xlabel("Root Mean Square Error (RMSE)")
ylabel("Frequency")
title("New Samples")
hold on
xline(RMSEbaseline,"r--")
legend(["Data" "Baseline RMSE"])

% dentify the top 10 sequences with the largest RMSE values.
[~,idxTop] = sort(errNew,"descend");
idxTop(1:10)

X = XNew{idxTop(1)};
sequenceLength = size(X,1);
Y = YNew(1:sequenceLength,:,idxTop(1));

% Visualize the sequence with the largest RMSE value and its reconstruction in a plot.
figure
t = tiledlayout(numChannels,1);
title(t,"Sequence " + idxTop(1))

for i = 1:numChannels
    nexttile

    plot(X(:,i))
    box off
    ylabel("Channel " + i)

    hold on
    plot(Y(:,i),"--")
end

nexttile(1)
legend(["Original" "Reconstructed"])


%% Identify Anomalous Regions
% Set the time step window size to 7. Identify windows that have time steps with RMSE values of at least 10% 
% above the maximum error value identified using the validation data.

RMSE = rmse(Y,X,2);

windowSize = 7;
thr = 1.1*RMSEbaseline;

idxAnomaly = false(1,size(X,1));
for t = 1:(size(X,1) - windowSize + 1)
    idxWindow = t:(t + windowSize - 1);

    if all(RMSE(idxWindow) > thr)
        idxAnomaly(idxWindow) = true;
    end
end

% Display the sequence in a plot and highlight the anomalous regions.
figure
t = tiledlayout(numChannels,1);
title(t,"Anomaly Detection ")

for i = 1:numChannels
    nexttile
    plot(X(:,i));
    ylabel("Channel " + i)
    box off
    hold on

    XAnomalous = nan(1,size(X,1));
    XAnomalous(idxAnomaly) = X(idxAnomaly,i);
    plot(XAnomalous,"r",LineWidth=3)
    hold off
end

xlabel("Time Step")

nexttile(1)
legend(["Input" "Anomalous"])


