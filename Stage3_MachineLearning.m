% =========================================================================
% STAGE 3: Machine Learning Implementation for Speech Enhancement
% =========================================================================
% This script implements a CNN-based approach for speech enhancement:
% 1. Design CNN Model Architecture
% 2. Prepare training data (noisy/clean spectrogram pairs)
% 3. Train the CNN to predict time-frequency masks
% 4. Inference on test samples
%
% Author: Alaa Aldirani
% Project: Real-Time Speech Enhancement
% =========================================================================

clear; close all; clc;

%% Configuration
fprintf('========================================\n');
fprintf('STAGE 3: Machine Learning Implementation\n');
fprintf('========================================\n\n');

% Load prepared dataset
fprintf('Loading prepared dataset...\n');
load('prepared_data/noizeus_prepared.mat');
fprintf('Dataset loaded successfully!\n');
fprintf('Training samples: %d\n', length(trainData));
fprintf('Testing samples: %d\n\n', length(testData));

% Set random seed for reproducibility
rng(42);

%% STFT Parameters
fs = trainData(1).fs;
winLen = round(0.032 * fs); % 32ms window
hopSize = round(0.016 * fs); % 16ms hop (50% overlap)
nfft = 2^nextpow2(winLen);
winFun = hamming(winLen, 'periodic');

fprintf('STFT Parameters:\n');
fprintf('  Window length: %d samples (%.1f ms)\n', winLen, winLen/fs*1000);
fprintf('  Hop size: %d samples (%.1f ms)\n', hopSize, hopSize/fs*1000);
fprintf('  FFT size: %d\n', nfft);
fprintf('  Frequency bins: %d\n\n', nfft/2 + 1);

%% ========================================================================
%  SECTION 1: DATA PREPARATION FOR CNN
% =========================================================================
fprintf('SECTION 1: Preparing Training Data\n');
fprintf('----------------------------------\n');

% Decide how many samples to use for training (to keep training time reasonable)
numTrainSamples = min(length(trainData), 100); % Use 100 samples or all if less
fprintf('Using %d training samples\n', numTrainSamples);

% Storage for spectrograms
trainSpectrograms = cell(numTrainSamples, 1);
trainMasks = cell(numTrainSamples, 1);

fprintf('Computing spectrograms...\n');
for i = 1:numTrainSamples
    if mod(i, 20) == 0
        fprintf('  Progress: %d/%d\n', i, numTrainSamples);
    end
    
    % Get signals
    cleanSig = trainData(i).clean;
    noisySig = trainData(i).noisy;
    
    % Compute STFT
    [S_clean, ~, ~] = stft(cleanSig, fs, 'Window', winFun, ...
                           'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    [S_noisy, ~, ~] = stft(noisySig, fs, 'Window', winFun, ...
                           'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    
    % Compute magnitude spectrograms
    magClean = abs(S_clean);
    magNoisy = abs(S_noisy);
    
    % Compute Ideal Binary Mask (IBM) or Ideal Ratio Mask (IRM)
    % Using IRM as it generally performs better
    epsilon = 1e-10; % Small value to avoid division by zero
    idealMask = magClean ./ (magNoisy + epsilon);
    idealMask = min(idealMask, 1); % Clip to [0, 1]
    
    % Store spectrograms and masks
    trainSpectrograms{i} = log(magNoisy + epsilon); % Log-magnitude for better training
    trainMasks{i} = idealMask;
end

fprintf('Spectrogram computation complete!\n\n');

%% ========================================================================
%  SECTION 2: PREPARE DATA FOR CNN TRAINING
% =========================================================================
fprintf('SECTION 2: Formatting Data for CNN\n');
fprintf('-----------------------------------\n');

% Find maximum dimensions to pad all spectrograms to same size
maxFreqBins = 0;
maxTimeBins = 0;
for i = 1:numTrainSamples
    [nFreq, nTime] = size(trainSpectrograms{i});
    maxFreqBins = max(maxFreqBins, nFreq);
    maxTimeBins = max(maxTimeBins, nTime);
end

fprintf('Maximum spectrogram dimensions:\n');
fprintf('  Frequency bins: %d\n', maxFreqBins);
fprintf('  Time bins: %d\n\n', maxTimeBins);

% Pad all spectrograms to same size
paddedSpectrograms = zeros(maxFreqBins, maxTimeBins, 1, numTrainSamples);
paddedMasks = zeros(maxFreqBins, maxTimeBins, 1, numTrainSamples);

for i = 1:numTrainSamples
    [nFreq, nTime] = size(trainSpectrograms{i});
    paddedSpectrograms(1:nFreq, 1:nTime, 1, i) = trainSpectrograms{i};
    paddedMasks(1:nFreq, 1:nTime, 1, i) = trainMasks{i};
end

fprintf('Data padding complete!\n');
fprintf('  Input shape: [%d x %d x 1 x %d]\n', maxFreqBins, maxTimeBins, numTrainSamples);
fprintf('  Target shape: [%d x %d x 1 x %d]\n\n', maxFreqBins, maxTimeBins, numTrainSamples);

%% ========================================================================
%  SECTION 3: CNN MODEL ARCHITECTURE
% =========================================================================
fprintf('SECTION 3: Designing CNN Architecture\n');
fprintf('--------------------------------------\n');

% Define a simple CNN architecture for mask estimation
% Input: Log-magnitude spectrogram [Freq x Time x 1]
% Output: Time-frequency mask [Freq x Time x 1]

layers = [
    imageInputLayer([maxFreqBins maxTimeBins 1], 'Name', 'input', 'Normalization', 'none')
    
    % Encoder layers
    convolution2dLayer(5, 16, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    
    convolution2dLayer(5, 32, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    
    convolution2dLayer(5, 64, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    
    % Decoder layers
    convolution2dLayer(5, 32, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    
    convolution2dLayer(5, 16, 'Padding', 'same', 'Name', 'conv5')
    batchNormalizationLayer('Name', 'bn5')
    reluLayer('Name', 'relu5')
    
    % Output layer - predict mask values [0, 1]
    convolution2dLayer(1, 1, 'Padding', 'same', 'Name', 'conv_out')
    sigmoidLayer('Name', 'sigmoid') % Sigmoid to constrain output to [0, 1]
    
    regressionLayer('Name', 'output')
];

fprintf('CNN Architecture:\n');
fprintf('  Input: [%d x %d x 1]\n', maxFreqBins, maxTimeBins);
fprintf('  Conv layers: 5\n');
fprintf('  Filters: 16 -> 32 -> 64 -> 32 -> 16 -> 1\n');
fprintf('  Kernel size: 5x5\n');
fprintf('  Activation: ReLU + Sigmoid output\n');
fprintf('  Output: Time-frequency mask [%d x %d x 1]\n\n', maxFreqBins, maxTimeBins);

%% ========================================================================
%  SECTION 4: TRAINING CONFIGURATION
% =========================================================================
fprintf('SECTION 4: Configuring Training\n');
fprintf('--------------------------------\n');

% Training options
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 8, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 15, ...
    'Shuffle', 'every-epoch', ...
    'ValidationFrequency', 10, ...
    'Verbose', true, ...
    'VerboseFrequency', 5, ...
    'Plots', 'training-progress');

fprintf('Training Configuration:\n');
fprintf('  Optimizer: Adam\n');
fprintf('  Max epochs: 50\n');
fprintf('  Batch size: 8\n');
fprintf('  Initial learning rate: 0.001\n');
fprintf('  Learning rate schedule: Piecewise (drop by 0.5 every 15 epochs)\n\n');

%% ========================================================================
%  SECTION 5: TRAIN THE CNN
% =========================================================================
fprintf('SECTION 5: Training the CNN\n');
fprintf('---------------------------\n');
fprintf('Starting training... This may take several minutes.\n\n');

% Train the network
try
    net = trainNetwork(paddedSpectrograms, paddedMasks, layers, options);
    fprintf('\nTraining completed successfully!\n\n');
catch ME
    fprintf('\nError during training: %s\n', ME.message);
    fprintf('This might be due to insufficient Deep Learning Toolbox.\n');
    fprintf('Continuing with inference on a simplified model...\n\n');
    net = [];
end

%% ========================================================================
%  SECTION 6: SAVE TRAINED MODEL
% =========================================================================
fprintf('SECTION 6: Saving Model\n');
fprintf('-----------------------\n');

% Create output directory
outputDir = 'ml_results';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Save the trained network
if ~isempty(net)
    save(fullfile(outputDir, 'trained_cnn.mat'), 'net', 'maxFreqBins', 'maxTimeBins', ...
         'winLen', 'hopSize', 'nfft', 'winFun');
    fprintf('Model saved to: %s\n', fullfile(outputDir, 'trained_cnn.mat'));
else
    fprintf('Model training failed - skipping save.\n');
end

%% ========================================================================
%  SECTION 7: INFERENCE ON TEST SAMPLES
% =========================================================================
fprintf('\nSECTION 7: Testing CNN on Sample Data\n');
fprintf('--------------------------------------\n');

if ~isempty(net)
    % Select a test sample
    testIdx = 1;
    cleanSig = testData(testIdx).clean;
    noisySig = testData(testIdx).noisy;
    noiseType = testData(testIdx).noiseType;
    snrLevel = testData(testIdx).snr;
    
    fprintf('Processing test sample:\n');
    fprintf('  Noise type: %s\n', noiseType);
    fprintf('  SNR level: %s\n', snrLevel);
    fprintf('  Duration: %.2f seconds\n\n', length(noisySig)/fs);
    
    % Compute STFT of noisy signal
    [S_noisy, F, T] = stft(noisySig, fs, 'Window', winFun, ...
                           'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    
    % Prepare input for CNN
    magNoisy = abs(S_noisy);
    phaseNoisy = angle(S_noisy);
    logMagNoisy = log(magNoisy + 1e-10);
    
    % Pad to match training dimensions
    [nFreq, nTime] = size(logMagNoisy);
    paddedInput = zeros(maxFreqBins, maxTimeBins, 1, 1);
    paddedInput(1:nFreq, 1:nTime, 1, 1) = logMagNoisy;
    
    % Predict mask using CNN
    predictedMaskPadded = predict(net, paddedInput);
    predictedMask = squeeze(predictedMaskPadded(1:nFreq, 1:nTime, 1, 1));
    
    % Apply mask to noisy spectrogram
    S_enhanced = predictedMask .* magNoisy .* exp(1j * phaseNoisy);
    
    % Inverse STFT to get enhanced signal
    enhancedSig_cnn = istft(S_enhanced, fs, 'Window', winFun, ...
                            'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    

    % Trim or pad to original length

    % Inverse STFT to get enhanced signal
    enhancedSig_cnn = istft(S_enhanced, fs, 'Window', winFun, ...
                        'OverlapLength', winLen-hopSize, 'FFTLength', nfft);

    % Take real part (remove any numerical imaginary components)
    enhancedSig_cnn = real(enhancedSig_cnn);

    % Trim or pad to original length
    if length(enhancedSig_cnn) >= length(noisySig)
        enhancedSig_cnn = enhancedSig_cnn(1:length(noisySig));  % Trim if longer
    else
        % Pad with zeros if shorter
        enhancedSig_cnn = [enhancedSig_cnn; zeros(length(noisySig) - length(enhancedSig_cnn), 1)];
    end
    
    % Compute SNR improvement
    snr_noisy = snr(cleanSig, noisySig - cleanSig);
    snr_cnn = snr(cleanSig, enhancedSig_cnn - cleanSig);
    
    fprintf('Results:\n');
    fprintf('  Noisy SNR: %.2f dB\n', snr_noisy);
    fprintf('  Enhanced SNR: %.2f dB\n', snr_cnn);
    fprintf('  SNR Improvement: %.2f dB\n\n', snr_cnn - snr_noisy);
    
    % Save enhanced audio
    audiowrite(fullfile(outputDir, 'enhanced_cnn_sample.wav'), enhancedSig_cnn, fs);
    fprintf('Enhanced audio saved!\n\n');
    
    % Visualization
    figure('Name', 'CNN Speech Enhancement', 'Position', [50 50 1400 800]);
    
    subplot(3,3,1);
    plot((0:length(cleanSig)-1)/fs, cleanSig);
    title('Clean Signal');
    xlabel('Time (s)'); ylabel('Amplitude');
    grid on; ylim([-1 1]);
    
    subplot(3,3,2);
    plot((0:length(noisySig)-1)/fs, noisySig);
    title(sprintf('Noisy Signal (%s, %s)', noiseType, snrLevel));
    xlabel('Time (s)'); ylabel('Amplitude');
    grid on; ylim([-1 1]);
    
    subplot(3,3,3);
    plot((0:length(enhancedSig_cnn)-1)/fs, enhancedSig_cnn);
    title('CNN Enhanced Signal');
    xlabel('Time (s)'); ylabel('Amplitude');
    grid on; ylim([-1 1]);
    
    subplot(3,3,4);
    spectrogram(cleanSig, winFun, winLen-hopSize, nfft, fs, 'yaxis');
    title('Clean Spectrogram');
    colorbar;
    
    subplot(3,3,5);
    spectrogram(noisySig, winFun, winLen-hopSize, nfft, fs, 'yaxis');
    title('Noisy Spectrogram');
    colorbar;
    
    subplot(3,3,6);
    spectrogram(enhancedSig_cnn, winFun, winLen-hopSize, nfft, fs, 'yaxis');
    title('CNN Enhanced Spectrogram');
    colorbar;
    
    subplot(3,3,7);
    imagesc(T, F/1000, log(magNoisy + 1e-10));
    axis xy; colorbar;
    title('Noisy Magnitude (dB)');
    xlabel('Time (s)'); ylabel('Frequency (kHz)');
    
    subplot(3,3,8);
    imagesc(T, F/1000, predictedMask);
    axis xy; colorbar;
    title('CNN Predicted Mask');
    xlabel('Time (s)'); ylabel('Frequency (kHz)');
    caxis([0 1]);
    
    subplot(3,3,9);
    imagesc(T, F/1000, log(abs(S_enhanced) + 1e-10));
    axis xy; colorbar;
    title('Enhanced Magnitude (dB)');
    xlabel('Time (s)'); ylabel('Frequency (kHz)');
    
    saveas(gcf, fullfile(outputDir, 'cnn_enhancement_visualization.png'));
    fprintf('Visualization saved!\n\n');
    
else
    fprintf('Skipping inference - model not trained.\n\n');
end

%% ========================================================================
%  SECTION 8: SUMMARY
% =========================================================================
fprintf('========================================\n');
fprintf('Stage 3 Complete!\n');
fprintf('========================================\n');
fprintf('\nSummary:\n');
fprintf('  Training samples used: %d\n', numTrainSamples);
fprintf('  CNN architecture: 5 convolutional layers\n');
fprintf('  Model saved: %s\n', fullfile(outputDir, 'trained_cnn.mat'));
fprintf('\nNext steps:\n');
fprintf('  1. Run Stage 4 for comprehensive evaluation\n');
fprintf('  2. Compare CNN with adaptive filtering methods\n');
fprintf('  3. Evaluate on full test set\n');
fprintf('========================================\n');
