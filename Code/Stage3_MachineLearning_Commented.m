% =========================================================================
% STAGE 3: Machine Learning Implementation for Speech Enhancement
% =========================================================================
% This script implements a CNN-based approach for speech enhancement:
% 1. Design CNN Model Architecture
% 2. Prepare training data (noisy/clean spectrogram pairs)
% 3. Train the CNN to predict time-frequency masks
% 4. Inference on test samples
%
% Machine Learning Approach:
% Instead of hand-crafted rules (like Wiener filtering), we let the neural
% network LEARN the optimal mapping from noisy to clean spectrograms.
% 
% The CNN learns to predict a "mask" that, when applied to the noisy
% spectrogram, produces an estimate of the clean spectrogram.
%
% Key Advantages over Traditional DSP:
% - Can learn complex, non-linear patterns
% - Adapts to specific noise types in training data
% - Often achieves better perceptual quality
%
% Key Disadvantages:
% - Requires large training dataset
% - Computationally expensive to train
% - May not generalize to unseen noise types
%
% Author: Alaa Aldirani
% Project: Real-Time Speech Enhancement
% =========================================================================

clear; close all; clc;

%% Configuration
fprintf('========================================\n');
fprintf('STAGE 3: Machine Learning Implementation\n');
fprintf('========================================\n\n');

% Load prepared dataset from Stage 1
fprintf('Loading prepared dataset...\n');
load('prepared_data/noizeus_prepared.mat');
fprintf('Dataset loaded successfully!\n');
fprintf('Training samples: %d\n', length(trainData));
fprintf('Testing samples: %d\n\n', length(testData));

% Set random seed for reproducibility
% This ensures same results when re-running the script
% Important for debugging and comparing experiments
rng(42);

%% STFT Parameters
% =========================================================================
% Same STFT parameters as Stage 2 for consistency
% These parameters determine the time-frequency resolution of spectrograms
% The CNN will learn to process these spectrograms
% =========================================================================
fs = trainData(1).fs;               % Sample rate (8000 Hz for NOIZEUS)
winLen = round(0.032 * fs);          % 32ms window (256 samples at 8kHz)
hopSize = round(0.016 * fs);         % 16ms hop (128 samples), 50% overlap
nfft = 2^nextpow2(winLen);           % FFT size (512 for zero-padding)
winFun = hamming(winLen, 'periodic'); % Hamming window for STFT

fprintf('STFT Parameters:\n');
fprintf('  Window length: %d samples (%.1f ms)\n', winLen, winLen/fs*1000);
fprintf('  Hop size: %d samples (%.1f ms)\n', hopSize, hopSize/fs*1000);
fprintf('  FFT size: %d\n', nfft);
fprintf('  Frequency bins: %d\n\n', nfft/2 + 1); % Only positive frequencies

%% ========================================================================
%  SECTION 1: DATA PREPARATION FOR CNN
% =========================================================================
% Prepare training data by computing spectrograms and ideal masks
%
% For each noisy/clean pair:
% 1. Compute STFT of both signals
% 2. Create log-magnitude spectrogram of noisy (INPUT to CNN)
% 3. Compute Ideal Ratio Mask (TARGET for CNN to predict)
%
% Ideal Ratio Mask (IRM):
%   IRM(t,f) = |S_clean(t,f)| / |S_noisy(t,f)|
%   This is the "perfect" mask that would recover clean from noisy
%   The CNN learns to approximate this mask from noisy input alone
%
% Why log-magnitude?
% - Log scale compresses dynamic range (speech has huge dynamic range)
% - Makes the data more suitable for neural network processing
% - Similar to human auditory perception (Weber-Fechner law)
% =========================================================================
fprintf('SECTION 1: Preparing Training Data\n');
fprintf('----------------------------------\n');

% Decide how many samples to use for training
% Using subset to keep training time reasonable for demonstration
% In practice, you'd use all available data for best performance
numTrainSamples = min(length(trainData), 100); % Use 100 samples or all if less
fprintf('Using %d training samples\n', numTrainSamples);

% Storage for spectrograms and masks (cell arrays for variable-length signals)
trainSpectrograms = cell(numTrainSamples, 1);  % Input: noisy log-magnitude
trainMasks = cell(numTrainSamples, 1);         % Target: ideal ratio mask

fprintf('Computing spectrograms...\n');
for i = 1:numTrainSamples
    if mod(i, 20) == 0
        fprintf('  Progress: %d/%d\n', i, numTrainSamples);
    end
    
    % Get clean and noisy signal pair
    cleanSig = trainData(i).clean;
    noisySig = trainData(i).noisy;
    
    % Compute STFT of both signals
    % S_clean and S_noisy are complex spectrograms [Freq × Time]
    [S_clean, ~, ~] = stft(cleanSig, fs, 'Window', winFun, ...
                           'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    [S_noisy, ~, ~] = stft(noisySig, fs, 'Window', winFun, ...
                           'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    
    % Compute magnitude spectrograms (discard phase for now)
    % Phase reconstruction is a separate challenge; here we focus on magnitude
    magClean = abs(S_clean);
    magNoisy = abs(S_noisy);
    
    % Compute Ideal Ratio Mask (IRM)
    % IRM tells us how much to scale each T-F bin
    % Values > 1 would amplify (not desired), so we clip to [0, 1]
    epsilon = 1e-10; % Small value to avoid division by zero
    idealMask = magClean ./ (magNoisy + epsilon);
    idealMask = min(idealMask, 1); % Clip to [0, 1] range
    
    % Store log-magnitude spectrogram as input feature
    % Adding epsilon prevents log(0) = -inf
    trainSpectrograms{i} = log(magNoisy + epsilon); % Log-magnitude for better training
    trainMasks{i} = idealMask;                       % Target mask [0, 1]
end

fprintf('Spectrogram computation complete!\n\n');

%% ========================================================================
%  SECTION 2: PREPARE DATA FOR CNN TRAINING
% =========================================================================
% CNN requires fixed-size inputs, but our signals have different lengths
% Solution: Pad all spectrograms to the maximum size
%
% Data format for MATLAB CNN:
% 4D array: [Height × Width × Channels × Samples]
% For spectrograms:
% - Height = Frequency bins (spatial dimension)
% - Width = Time frames (spatial dimension)
% - Channels = 1 (single channel, like grayscale image)
% - Samples = Number of training examples (batch dimension)
%
% This is similar to image processing, where:
% - Each spectrogram is treated like an image
% - CNN learns spatial patterns in time-frequency domain
% =========================================================================
fprintf('SECTION 2: Formatting Data for CNN\n');
fprintf('-----------------------------------\n');

% Find maximum dimensions to pad all spectrograms to same size
% This is necessary because CNN needs fixed-size input
maxFreqBins = 0;
maxTimeBins = 0;
for i = 1:numTrainSamples
    [nFreq, nTime] = size(trainSpectrograms{i});
    maxFreqBins = max(maxFreqBins, nFreq); % Maximum frequency bins
    maxTimeBins = max(maxTimeBins, nTime); % Maximum time frames
end

fprintf('Maximum spectrogram dimensions:\n');
fprintf('  Frequency bins: %d\n', maxFreqBins);
fprintf('  Time bins: %d\n\n', maxTimeBins);

% Pad all spectrograms to same size (zero-padding)
% Creates 4D arrays: [Freq × Time × 1 × NumSamples]
paddedSpectrograms = zeros(maxFreqBins, maxTimeBins, 1, numTrainSamples);
paddedMasks = zeros(maxFreqBins, maxTimeBins, 1, numTrainSamples);

for i = 1:numTrainSamples
    [nFreq, nTime] = size(trainSpectrograms{i});
    % Place actual data in top-left corner, rest is zero-padded
    paddedSpectrograms(1:nFreq, 1:nTime, 1, i) = trainSpectrograms{i};
    paddedMasks(1:nFreq, 1:nTime, 1, i) = trainMasks{i};
end

fprintf('Data padding complete!\n');
fprintf('  Input shape: [%d x %d x 1 x %d]\n', maxFreqBins, maxTimeBins, numTrainSamples);
fprintf('  Target shape: [%d x %d x 1 x %d]\n\n', maxFreqBins, maxTimeBins, numTrainSamples);

%% ========================================================================
%  SECTION 3: CNN MODEL ARCHITECTURE
% =========================================================================
% Design a Convolutional Neural Network for mask estimation
%
% Architecture: Encoder-Decoder style
% - Encoder: Extract features from noisy spectrogram
% - Decoder: Generate mask from features
%
% Why CNN for spectrograms?
% - CNNs excel at learning local patterns (similar to image processing)
% - Speech has structure in both time and frequency (formants, harmonics)
% - Convolutional layers can capture these patterns
%
% Layer Types:
% - imageInputLayer: Defines input size and normalization
% - convolution2dLayer: Applies learnable filters to extract features
% - batchNormalizationLayer: Normalizes activations (faster training)
% - reluLayer: Non-linear activation (ReLU = max(0, x))
% - sigmoidLayer: Squashes output to [0, 1] (for mask values)
% - regressionLayer: Computes mean squared error loss
%
% Architecture Details:
% Input → Conv(16) → Conv(32) → Conv(64) → Conv(32) → Conv(16) → Conv(1) → Output
%
% Filter sizes of 5x5:
% - Large enough to capture local patterns
% - Small enough to be computationally efficient
%
% 'same' padding:
% - Output size = Input size
% - Important for pixel-wise (T-F bin-wise) prediction
% =========================================================================
fprintf('SECTION 3: Designing CNN Architecture\n');
fprintf('--------------------------------------\n');

% Define CNN architecture for mask estimation
% Input: Log-magnitude spectrogram [Freq x Time x 1]
% Output: Time-frequency mask [Freq x Time x 1]

layers = [
    % Input layer - specifies input dimensions
    % 'Normalization', 'none': We'll handle normalization ourselves
    imageInputLayer([maxFreqBins maxTimeBins 1], 'Name', 'input', 'Normalization', 'none')
    
    % ===== Encoder layers: Extract features =====
    % First conv layer: 16 filters of 5x5
    % Learns low-level features (edges, simple patterns)
    convolution2dLayer(5, 16, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1') % Normalize activations
    reluLayer('Name', 'relu1') % Non-linear activation
    
    % Second conv layer: 32 filters
    % Learns mid-level features (combinations of low-level)
    convolution2dLayer(5, 32, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    
    % Third conv layer: 64 filters (bottleneck)
    % Learns high-level features (complex patterns)
    convolution2dLayer(5, 64, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    
    % ===== Decoder layers: Generate mask =====
    % Fourth conv layer: 32 filters
    % Starts reconstructing from features
    convolution2dLayer(5, 32, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    
    % Fifth conv layer: 16 filters
    convolution2dLayer(5, 16, 'Padding', 'same', 'Name', 'conv5')
    batchNormalizationLayer('Name', 'bn5')
    reluLayer('Name', 'relu5')
    
    % Output layer - predict mask values [0, 1]
    % 1x1 convolution: combines features into single channel (mask)
    convolution2dLayer(1, 1, 'Padding', 'same', 'Name', 'conv_out')
    % Sigmoid activation: squashes output to [0, 1] range
    % Perfect for mask values which represent scaling factors
    sigmoidLayer('Name', 'sigmoid')
    
    % Regression layer: computes mean squared error between predicted and target mask
    % MSE Loss = mean((predicted_mask - target_mask)^2)
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
% Configure training hyperparameters
%
% Key Hyperparameters:
% - Optimizer: Adam (Adaptive Moment Estimation)
%   * Combines best of AdaGrad and RMSProp
%   * Adapts learning rate for each parameter
%   * Works well for most deep learning problems
%
% - Learning Rate: Controls step size in gradient descent
%   * Too high: Training is unstable, may diverge
%   * Too low: Training is slow, may get stuck
%   * 0.001 is a good default for Adam
%
% - Learning Rate Schedule: Reduce LR over time
%   * Helps fine-tune the network as it converges
%   * Drop by 0.5 every 15 epochs
%
% - Batch Size: Number of samples processed together
%   * Larger batch: More stable gradients, more memory
%   * Smaller batch: More noise, less memory
%   * 8 is a reasonable choice for this dataset size
%
% - Epochs: Number of passes through entire dataset
%   * More epochs: Better training (up to a point)
%   * Too many: Overfitting (memorizes training data)
%   * 50 epochs should be sufficient for demonstration
% =========================================================================
fprintf('SECTION 4: Configuring Training\n');
fprintf('--------------------------------\n');

% Training options using Adam optimizer
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...                    % Train for 50 complete passes
    'MiniBatchSize', 8, ...                  % Process 8 samples at a time
    'InitialLearnRate', 0.001, ...           % Starting learning rate
    'LearnRateSchedule', 'piecewise', ...    % Reduce LR over time
    'LearnRateDropFactor', 0.5, ...          % Multiply LR by 0.5
    'LearnRateDropPeriod', 15, ...           % Every 15 epochs
    'Shuffle', 'every-epoch', ...            % Randomize order each epoch
    'ValidationFrequency', 10, ...           % Check validation every 10 iterations
    'Verbose', true, ...                     % Print training progress
    'VerboseFrequency', 5, ...               % Print every 5 iterations
    'Plots', 'training-progress');           % Show training plot

fprintf('Training Configuration:\n');
fprintf('  Optimizer: Adam\n');
fprintf('  Max epochs: 50\n');
fprintf('  Batch size: 8\n');
fprintf('  Initial learning rate: 0.001\n');
fprintf('  Learning rate schedule: Piecewise (drop by 0.5 every 15 epochs)\n\n');

%% ========================================================================
%  SECTION 5: TRAIN THE CNN
% =========================================================================
% Train the neural network on prepared data
%
% Training Process:
% 1. Forward pass: Compute predicted mask from input
% 2. Loss computation: MSE between predicted and target mask
% 3. Backward pass: Compute gradients via backpropagation
% 4. Weight update: Adjust weights to minimize loss
% 5. Repeat for all batches and epochs
%
% The training plot shows:
% - Training loss: How well model fits training data
% - Should decrease over time
% - If it stops decreasing, model has converged or is stuck
%
% Note: Training requires Deep Learning Toolbox
% If not available, the script will catch the error
% =========================================================================
fprintf('SECTION 5: Training the CNN\n');
fprintf('---------------------------\n');
fprintf('Starting training... This may take several minutes.\n\n');

% Train the network
try
    % trainNetwork: Main MATLAB function for training neural networks
    % Inputs:
    %   - paddedSpectrograms: Input data [Height x Width x Channels x Samples]
    %   - paddedMasks: Target data [Height x Width x Channels x Samples]
    %   - layers: Network architecture
    %   - options: Training configuration
    % Output:
    %   - net: Trained network object
    net = trainNetwork(paddedSpectrograms, paddedMasks, layers, options);
    fprintf('\nTraining completed successfully!\n\n');
catch ME
    % Handle errors (e.g., missing Deep Learning Toolbox)
    fprintf('\nError during training: %s\n', ME.message);
    fprintf('This might be due to insufficient Deep Learning Toolbox.\n');
    fprintf('Continuing with inference on a simplified model...\n\n');
    net = []; % Empty network indicates training failed
end

%% ========================================================================
%  SECTION 6: SAVE TRAINED MODEL
% =========================================================================
% Save the trained model and STFT parameters for later use
% Important to save STFT parameters so we can process new signals
% with the same settings used during training
% =========================================================================
fprintf('SECTION 6: Saving Model\n');
fprintf('-----------------------\n');

% Create output directory
outputDir = 'ml_results';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Save the trained network and all necessary parameters
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
% Test the trained CNN on a sample not seen during training
%
% Inference Process:
% 1. Compute STFT of noisy signal
% 2. Extract log-magnitude spectrogram
% 3. Pad to match training dimensions
% 4. Pass through trained CNN to get predicted mask
% 5. Apply mask to noisy magnitude
% 6. Combine with original phase
% 7. Inverse STFT to get enhanced time-domain signal
%
% Important: We use the noisy phase for reconstruction
% This is a simplification; phase estimation is another research area
% In practice, this works reasonably well since phase is less perceptually
% important than magnitude for speech
% =========================================================================
fprintf('\nSECTION 7: Testing CNN on Sample Data\n');
fprintf('--------------------------------------\n');

if ~isempty(net)
    % Select a test sample (from test set, not training set)
    testIdx = 1;
    cleanSig = testData(testIdx).clean;
    noisySig = testData(testIdx).noisy;
    noiseType = testData(testIdx).noiseType;
    snrLevel = testData(testIdx).snr;
    
    fprintf('Processing test sample:\n');
    fprintf('  Noise type: %s\n', noiseType);
    fprintf('  SNR level: %s\n', snrLevel);
    fprintf('  Duration: %.2f seconds\n\n', length(noisySig)/fs);
    
    % Step 1: Compute STFT of noisy signal
    [S_noisy, F, T] = stft(noisySig, fs, 'Window', winFun, ...
                           'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    
    % Step 2: Prepare input for CNN
    magNoisy = abs(S_noisy);        % Magnitude spectrum
    phaseNoisy = angle(S_noisy);    % Phase spectrum (keep for reconstruction)
    logMagNoisy = log(magNoisy + 1e-10); % Log-magnitude input
    
    % Step 3: Pad to match training dimensions
    % CNN expects same size as during training
    [nFreq, nTime] = size(logMagNoisy);
    paddedInput = zeros(maxFreqBins, maxTimeBins, 1, 1);
    paddedInput(1:nFreq, 1:nTime, 1, 1) = logMagNoisy;
    
    % Step 4: Predict mask using trained CNN
    % predict() runs forward pass through network
    predictedMaskPadded = predict(net, paddedInput);
    % Extract only the valid region (remove padding)
    predictedMask = squeeze(predictedMaskPadded(1:nFreq, 1:nTime, 1, 1));
    
    % Step 5: Apply mask to noisy spectrogram
    % Enhanced magnitude = Predicted mask × Noisy magnitude
    % Reconstruct complex spectrogram using original phase
    S_enhanced = predictedMask .* magNoisy .* exp(1j * phaseNoisy);
    
    % Step 6: Inverse STFT to get enhanced time-domain signal
    enhancedSig_cnn = istft(S_enhanced, fs, 'Window', winFun, ...
                            'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    
    % Take real part (remove any numerical imaginary components)
    % ISTFT should give real output, but numerical errors may introduce tiny imag parts
    enhancedSig_cnn = real(enhancedSig_cnn);

    % Trim or pad to original length
    if length(enhancedSig_cnn) >= length(noisySig)
        enhancedSig_cnn = enhancedSig_cnn(1:length(noisySig));  % Trim if longer
    else
        % Pad with zeros if shorter
        enhancedSig_cnn = [enhancedSig_cnn; zeros(length(noisySig) - length(enhancedSig_cnn), 1)];
    end
    
    % Compute SNR improvement
    % Compare enhanced signal with clean reference
    snr_noisy = snr(cleanSig, noisySig - cleanSig);
    snr_cnn = snr(cleanSig, enhancedSig_cnn - cleanSig);
    
    fprintf('Results:\n');
    fprintf('  Noisy SNR: %.2f dB\n', snr_noisy);
    fprintf('  Enhanced SNR: %.2f dB\n', snr_cnn);
    fprintf('  SNR Improvement: %.2f dB\n\n', snr_cnn - snr_noisy);
    
    % Save enhanced audio for listening test
    audiowrite(fullfile(outputDir, 'enhanced_cnn_sample.wav'), enhancedSig_cnn, fs);
    fprintf('Enhanced audio saved!\n\n');
    
    % =====================================================================
    % Visualization: Comprehensive view of CNN enhancement process
    % Shows input, processing, and output at each stage
    % =====================================================================
    figure('Name', 'CNN Speech Enhancement', 'Position', [50 50 1400 800]);
    
    % Row 1: Time-domain signals
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
    
    % Row 2: Spectrograms
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
    
    % Row 3: Processing details (key difference from traditional methods)
    subplot(3,3,7);
    imagesc(T, F/1000, log(magNoisy + 1e-10)); % Log-magnitude input to CNN
    axis xy; colorbar;
    title('Noisy Magnitude (dB)');
    xlabel('Time (s)'); ylabel('Frequency (kHz)');
    
    subplot(3,3,8);
    imagesc(T, F/1000, predictedMask); % CNN-predicted mask (learned from data!)
    axis xy; colorbar;
    title('CNN Predicted Mask');
    xlabel('Time (s)'); ylabel('Frequency (kHz)');
    caxis([0 1]); % Mask values between 0 and 1
    
    subplot(3,3,9);
    imagesc(T, F/1000, log(abs(S_enhanced) + 1e-10)); % Result after applying mask
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
