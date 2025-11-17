% =========================================================================
% STAGE 4: Comparative Analysis and Evaluation (FIXED)
% =========================================================================
% This script performs comprehensive evaluation and comparison of:
% 1. Adaptive Filtering methods (Wiener, Improved Wiener, NLMS)
% 2. Machine Learning approach (CNN)
%
% FIXES APPLIED:
% - Explicit CNN loading with proper error reporting
% - Robust CNN processing with actual inference (not fallback)
% - Proper STFT parameter usage from trained model
%
% Purpose:
% Objectively compare all enhancement methods using multiple metrics
% to understand strengths and weaknesses of each approach.
%
% Evaluation Metrics:
% - Scale-Invariant Signal-to-Noise Ratio (SI-SNR)
% - Short-Time Objective Intelligibility (STOI)
% - Log-Spectral Distortion (LSD)
% - Segmental SNR
%
% Why Multiple Metrics?
% Each metric captures different aspects of speech quality:
% - SNR: Overall noise reduction
% - SI-SNR: Scale-invariant version (robust to amplitude differences)
% - STOI: Speech intelligibility (how understandable is the speech?)
% - LSD: Spectral fidelity (how well is spectrum preserved?)
% - SegSNR: Local SNR (varies over time)
%
% Author: Alaa Aldirani
% Project: Real-Time Speech Enhancement
% =========================================================================

clear; close all; clc;

%% Configuration
fprintf('========================================\n');
fprintf('STAGE 4: Comparative Analysis (FIXED)\n');
fprintf('========================================\n\n');

% Load prepared dataset from Stage 1
fprintf('Loading prepared dataset...\n');
load('prepared_data/noizeus_prepared.mat');
fprintf('Dataset loaded successfully!\n\n');

%% Set default STFT parameters FIRST (always define these)
% =========================================================================
% These parameters must match what was used in training (Stage 3)
% If CNN model was trained with different parameters, they'll be overridden
% =========================================================================
fs = testData(1).fs;                    % Sample rate (8000 Hz)
winLen = round(0.032 * fs);              % 32ms window
hopSize = round(0.016 * fs);             % 16ms hop
nfft = 2^nextpow2(winLen);               % FFT size
winFun = hamming(winLen, 'periodic');    % Hamming window

%% Load trained CNN model (if available)
% =========================================================================
% Attempt to load the CNN model trained in Stage 3
% Need to load:
% - net: The trained network object
% - maxFreqBins, maxTimeBins: Dimensions used during training (for padding)
% - STFT parameters: To ensure consistency with training
%
% Robust loading:
% 1. Check what variables are in the file
% 2. Load required variables
% 3. Load optional variables (use defaults if missing)
% =========================================================================
fprintf('Trying to load trained CNN model...\n');
hasCNN = false;
try
    % First check what variables are in the file
    % whos('-file', ...) lists variables without loading them
    cnnFileInfo = whos('-file', 'ml_results/trained_cnn.mat');
    cnnVarNames = {cnnFileInfo.name};
    fprintf('  Variables in trained_cnn.mat: %s\n', strjoin(cnnVarNames, ', '));
    
    % Load the network (required - this is the trained model)
    if ismember('net', cnnVarNames)
        load('ml_results/trained_cnn.mat', 'net');
    else
        error('net variable not found in trained_cnn.mat');
    end
    
    % Load max dimensions (required for padding input to match training size)
    if ismember('maxFreqBins', cnnVarNames) && ismember('maxTimeBins', cnnVarNames)
        load('ml_results/trained_cnn.mat', 'maxFreqBins', 'maxTimeBins');
    else
        error('maxFreqBins or maxTimeBins not found in trained_cnn.mat');
    end
    
    % Load STFT parameters if available (optional - use defaults if not present)
    % These ensure we process test data the same way as training data
    if ismember('winLen', cnnVarNames)
        load('ml_results/trained_cnn.mat', 'winLen');
        fprintf('  Loaded winLen from CNN file: %d\n', winLen);
    else
        fprintf('  Using default winLen: %d\n', winLen);
    end
    
    if ismember('hopSize', cnnVarNames)
        load('ml_results/trained_cnn.mat', 'hopSize');
        fprintf('  Loaded hopSize from CNN file: %d\n', hopSize);
    else
        fprintf('  Using default hopSize: %d\n', hopSize);
    end
    
    if ismember('nfft', cnnVarNames)
        load('ml_results/trained_cnn.mat', 'nfft');
        fprintf('  Loaded nfft from CNN file: %d\n', nfft);
    else
        fprintf('  Using default nfft: %d\n', nfft);
    end
    
    if ismember('winFun', cnnVarNames)
        load('ml_results/trained_cnn.mat', 'winFun');
        fprintf('  Loaded winFun from CNN file\n');
    else
        fprintf('  Using default winFun (Hamming)\n');
    end
    
    hasCNN = true;
    fprintf('  CNN model loaded successfully.\n');
    fprintf('  CNN STFT settings: winLen=%d, hopSize=%d, nfft=%d\n', ...
            winLen, hopSize, nfft);
catch ME
    fprintf('  WARNING: Could not load CNN model: %s\n', ME.message);
    fprintf('  Continuing with adaptive filtering only.\n\n');
    hasCNN = false;
end

% Decide how many test samples to process
% Using subset for faster processing; increase for more reliable statistics
numSamplesToProcess = min(50, length(testData));
fprintf('Processing %d test samples...\n\n', numSamplesToProcess);

%% ========================================================================
%  SECTION 1: EVALUATION METRICS IMPLEMENTATION
% =========================================================================
% Define the evaluation metrics used to compare methods
%
% Why These Specific Metrics?
%
% 1. SI-SNR (Scale-Invariant SNR):
%    - Robust to amplitude scaling
%    - Commonly used in source separation
%    - Measures how well signal is separated from noise
%
% 2. STOI (Short-Time Objective Intelligibility):
%    - Correlates highly with human intelligibility ratings
%    - Range: 0 to 1 (higher is better)
%    - Important for practical speech communication
%
% 3. LSD (Log-Spectral Distortion):
%    - Measures distortion in spectral domain
%    - Lower is better
%    - Captures how well spectral shape is preserved
%
% 4. Segmental SNR:
%    - Computes SNR in short segments, then averages
%    - More sensitive to local variations than global SNR
%    - Reflects perceptual quality better than global SNR
%
% Note: Full implementations of these metrics are at end of script
% =========================================================================
fprintf('SECTION 1: Implementing Evaluation Metrics\n');
fprintf('-------------------------------------------\n\n');

fprintf('Evaluation metrics implemented:\n');
fprintf('  - SI-SNR (Scale-Invariant SNR)\n');
fprintf('  - STOI (Speech Intelligibility - simplified)\n');
fprintf('  - LSD (Log-Spectral Distortion)\n');
fprintf('  - Segmental SNR\n\n');

%% ========================================================================
%  SECTION 2: BATCH PROCESSING AND EVALUATION
% =========================================================================
% Process all test samples with each enhancement method and compute metrics
%
% For each test sample:
% 1. Apply all enhancement methods
% 2. Compute all metrics for each method
% 3. Store results for statistical analysis
%
% This gives us a comprehensive view of performance across:
% - Different noise types
% - Different SNR levels
% - Different enhancement methods
% =========================================================================
fprintf('SECTION 2: Batch Processing\n');
fprintf('---------------------------\n');

% Initialize results storage
% Pre-allocate arrays for all metrics and all methods
results = struct();

% Noisy (baseline) metrics
results.snr_noisy = zeros(numSamplesToProcess, 1);
results.sisnr_noisy = zeros(numSamplesToProcess, 1);
results.stoi_noisy = zeros(numSamplesToProcess, 1);
results.lsd_noisy = zeros(numSamplesToProcess, 1);
results.segsnr_noisy = zeros(numSamplesToProcess, 1);

% Wiener filter results
results.snr_wiener = zeros(numSamplesToProcess, 1);
results.sisnr_wiener = zeros(numSamplesToProcess, 1);
results.stoi_wiener = zeros(numSamplesToProcess, 1);
results.lsd_wiener = zeros(numSamplesToProcess, 1);
results.segsnr_wiener = zeros(numSamplesToProcess, 1);

% Improved Wiener filter results (with oversubtraction)
results.snr_wiener_imp = zeros(numSamplesToProcess, 1);
results.sisnr_wiener_imp = zeros(numSamplesToProcess, 1);
results.stoi_wiener_imp = zeros(numSamplesToProcess, 1);
results.lsd_wiener_imp = zeros(numSamplesToProcess, 1);
results.segsnr_wiener_imp = zeros(numSamplesToProcess, 1);

% NLMS filter results (adaptive filtering)
results.snr_nlms = zeros(numSamplesToProcess, 1);
results.sisnr_nlms = zeros(numSamplesToProcess, 1);
results.stoi_nlms = zeros(numSamplesToProcess, 1);
results.lsd_nlms = zeros(numSamplesToProcess, 1);
results.segsnr_nlms = zeros(numSamplesToProcess, 1);

% CNN results (if available)
if hasCNN
    results.snr_cnn = zeros(numSamplesToProcess, 1);
    results.sisnr_cnn = zeros(numSamplesToProcess, 1);
    results.stoi_cnn = zeros(numSamplesToProcess, 1);
    results.lsd_cnn = zeros(numSamplesToProcess, 1);
    results.segsnr_cnn = zeros(numSamplesToProcess, 1);
    results.cnn_processing_success = false(numSamplesToProcess, 1); % Track successes
end

% Metadata for analysis
results.noiseTypes = cell(numSamplesToProcess, 1);
results.snrLevels = cell(numSamplesToProcess, 1);

fprintf('Starting batch processing...\n');
fprintf('Progress: ');

cnnSuccessCount = 0; % Track CNN processing successes
cnnFailCount = 0;    % Track CNN processing failures

% Main processing loop - iterate through all test samples
for idx = 1:numSamplesToProcess
    if mod(idx, 10) == 0
        fprintf('%d ', idx); % Progress indicator
    end
    
    % Extract signals and metadata for current sample
    cleanSig = testData(idx).clean;        % Ground truth (clean speech)
    noisySig = testData(idx).noisy;        % Input (noisy speech)
    noiseType = testData(idx).noiseType;   % Type of noise added
    snrLevel = testData(idx).snr;          % SNR level
    
    % Store metadata for later analysis by noise type and SNR
    results.noiseTypes{idx} = noiseType;
    results.snrLevels{idx} = snrLevel;
    
    %% Process with adaptive filtering methods
    % =====================================================================
    % Apply traditional DSP methods (same as Stage 2, but automated)
    % =====================================================================
    
    % Compute STFT of noisy signal
    [S_noisy, ~, ~] = stft(noisySig, fs, 'Window', winFun, ...
                           'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    
    % VAD for noise estimation (same approach as Stage 2)
    frameEnergy = sum(abs(S_noisy).^2, 1);
    frameEnergyDB = 10*log10(frameEnergy + eps);
    energyThreshold = mean(frameEnergyDB) - 5;
    vadDecisions = frameEnergyDB > energyThreshold;
    vadDecisions = medfilt1(double(vadDecisions), 5) > 0.5;
    
    % Estimate noise PSD from noise-only frames
    noiseFrames = S_noisy(:, ~vadDecisions);
    if ~isempty(noiseFrames)
        noisePSD = mean(abs(noiseFrames).^2, 2);
    else
        % Fallback: use first 10 frames
        noisePSD = mean(abs(S_noisy(:, 1:min(10, size(S_noisy,2)))).^2, 2);
    end
    
    % Standard Wiener filter (H = max(1 - N/Y, 0))
    noisyPSD = abs(S_noisy).^2;
    wienerGain = max(1 - bsxfun(@rdivide, noisePSD, noisyPSD), 0);
    S_wiener = wienerGain .* S_noisy;
    enhancedSig_wiener = istft(S_wiener, fs, 'Window', winFun, ...
                               'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    % Ensure output length matches input
    enhancedSig_wiener = enhancedSig_wiener(1:min(length(enhancedSig_wiener), length(noisySig)));
    if length(enhancedSig_wiener) < length(noisySig)
        enhancedSig_wiener = [enhancedSig_wiener; zeros(length(noisySig)-length(enhancedSig_wiener), 1)];
    end
    
    % Improved Wiener filter with oversubtraction (alpha=2, beta=0.01)
    alpha = 2.0; beta = 0.01;
    improvedWienerGain = max(1 - alpha * bsxfun(@rdivide, noisePSD, noisyPSD), beta);
    S_wiener_imp = improvedWienerGain .* S_noisy;
    enhancedSig_wiener_imp = istft(S_wiener_imp, fs, 'Window', winFun, ...
                                    'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    enhancedSig_wiener_imp = enhancedSig_wiener_imp(1:min(length(enhancedSig_wiener_imp), length(noisySig)));
    if length(enhancedSig_wiener_imp) < length(noisySig)
        enhancedSig_wiener_imp = [enhancedSig_wiener_imp; zeros(length(noisySig)-length(enhancedSig_wiener_imp), 1)];
    end
    
    % NLMS adaptive filter
    filterOrder = 32; mu = 0.1; delta = 0.01;
    w = zeros(filterOrder, 1);
    enhancedSig_nlms = zeros(size(noisySig));
    refSignal = [zeros(filterOrder, 1); noisySig(1:end-filterOrder)];
    
    % NLMS algorithm (sample-by-sample)
    for n = filterOrder+1:length(noisySig)
        x = refSignal(n:-1:n-filterOrder+1);
        y = w' * x;
        e = noisySig(n) - y;
        enhancedSig_nlms(n) = e;
        w = w + (mu / (x'*x + delta)) * e * x;
    end
    % Remove initial transient
    enhancedSig_nlms = enhancedSig_nlms(filterOrder+1:end);
    cleanSig_nlms = cleanSig(filterOrder+1:end);
    
    %% Process with CNN (if available) - FIXED VERSION
    % =====================================================================
    % Apply CNN-based enhancement (learned from data in Stage 3)
    % Robust error handling to track success/failure
    % =====================================================================
    if hasCNN
        try
            % Compute STFT using the SAME parameters as during training
            % This is critical for proper CNN inference
            [S_noisy_cnn, ~, ~] = stft(noisySig, fs, 'Window', winFun, ...
                                       'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
            magNoisy = abs(S_noisy_cnn);
            phaseNoisy = angle(S_noisy_cnn);
            logMagNoisy = log(magNoisy + 1e-10);
            
            % Pad to match training dimensions
            [nFreq, nTime] = size(logMagNoisy);
            
            % Ensure we don't exceed the maximum dimensions
            % If signal is too long, CNN can't process it (fixed architecture)
            if nFreq > maxFreqBins || nTime > maxTimeBins
                error('Input dimensions (%d x %d) exceed max (%d x %d)', ...
                      nFreq, nTime, maxFreqBins, maxTimeBins);
            end
            
            % Create padded input (zeros in unused areas)
            paddedInput = zeros(maxFreqBins, maxTimeBins, 1, 1);
            paddedInput(1:nFreq, 1:nTime, 1, 1) = logMagNoisy;
            
            % Predict mask using the trained CNN
            predictedMaskPadded = predict(net, paddedInput);
            predictedMask = squeeze(predictedMaskPadded(1:nFreq, 1:nTime, 1, 1));
            
            % Apply mask to magnitude spectrum
            S_enhanced_cnn = predictedMask .* magNoisy .* exp(1j * phaseNoisy);
            
            % Inverse STFT to reconstruct time-domain signal
            enhancedSig_cnn = istft(S_enhanced_cnn, fs, 'Window', winFun, ...
                                    'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
            
            % Match length to original signal
            if length(enhancedSig_cnn) >= length(noisySig)
                enhancedSig_cnn = enhancedSig_cnn(1:length(noisySig));
            else
                enhancedSig_cnn = [enhancedSig_cnn; zeros(length(noisySig)-length(enhancedSig_cnn), 1)];
            end
            
            % Mark as successful processing
            results.cnn_processing_success(idx) = true;
            cnnSuccessCount = cnnSuccessCount + 1;
            
        catch ME
            % Print the error for debugging (only for first few failures)
            if cnnFailCount < 3
                fprintf('\n  CNN processing error for sample %d: %s\n', idx, ME.message);
            end
            % DO NOT fall back to noisy signal - set to NaN instead
            % This prevents biased results from failed processing
            enhancedSig_cnn = NaN(size(noisySig));
            results.cnn_processing_success(idx) = false;
            cnnFailCount = cnnFailCount + 1;
        end
    end
    
    %% Compute all metrics for this sample
    % =====================================================================
    % Evaluate each method using multiple objective metrics
    % Error handling ensures one bad sample doesn't crash entire analysis
    % =====================================================================
    try
        % Noisy metrics (baseline performance)
        results.snr_noisy(idx) = snr(cleanSig, noisySig - cleanSig);
        results.sisnr_noisy(idx) = compute_sisnr(cleanSig, noisySig);
        results.stoi_noisy(idx) = compute_stoi_simplified(cleanSig, noisySig, fs);
        results.lsd_noisy(idx) = compute_lsd(cleanSig, noisySig, fs, winLen, hopSize, nfft);
        results.segsnr_noisy(idx) = computeSegmentalSNR(cleanSig, noisySig, fs);
        
        % Wiener filter metrics
        results.snr_wiener(idx) = snr(cleanSig, enhancedSig_wiener - cleanSig);
        results.sisnr_wiener(idx) = compute_sisnr(cleanSig, enhancedSig_wiener);
        results.stoi_wiener(idx) = compute_stoi_simplified(cleanSig, enhancedSig_wiener, fs);
        results.lsd_wiener(idx) = compute_lsd(cleanSig, enhancedSig_wiener, fs, winLen, hopSize, nfft);
        results.segsnr_wiener(idx) = computeSegmentalSNR(cleanSig, enhancedSig_wiener, fs);
        
        % Improved Wiener filter metrics
        results.snr_wiener_imp(idx) = snr(cleanSig, enhancedSig_wiener_imp - cleanSig);
        results.sisnr_wiener_imp(idx) = compute_sisnr(cleanSig, enhancedSig_wiener_imp);
        results.stoi_wiener_imp(idx) = compute_stoi_simplified(cleanSig, enhancedSig_wiener_imp, fs);
        results.lsd_wiener_imp(idx) = compute_lsd(cleanSig, enhancedSig_wiener_imp, fs, winLen, hopSize, nfft);
        results.segsnr_wiener_imp(idx) = computeSegmentalSNR(cleanSig, enhancedSig_wiener_imp, fs);
        
        % NLMS filter metrics (using trimmed signals due to transient removal)
        results.snr_nlms(idx) = snr(cleanSig_nlms, enhancedSig_nlms - cleanSig_nlms);
        results.sisnr_nlms(idx) = compute_sisnr(cleanSig_nlms, enhancedSig_nlms);
        results.stoi_nlms(idx) = compute_stoi_simplified(cleanSig_nlms, enhancedSig_nlms, fs);
        results.lsd_nlms(idx) = compute_lsd(cleanSig_nlms, enhancedSig_nlms, fs, winLen, hopSize, nfft);
        results.segsnr_nlms(idx) = computeSegmentalSNR(cleanSig_nlms, enhancedSig_nlms, fs);
        
        % CNN metrics (if available AND processing was successful)
        if hasCNN && results.cnn_processing_success(idx)
            results.snr_cnn(idx) = snr(cleanSig, enhancedSig_cnn - cleanSig);
            results.sisnr_cnn(idx) = compute_sisnr(cleanSig, enhancedSig_cnn);
            results.stoi_cnn(idx) = compute_stoi_simplified(cleanSig, enhancedSig_cnn, fs);
            results.lsd_cnn(idx) = compute_lsd(cleanSig, enhancedSig_cnn, fs, winLen, hopSize, nfft);
            results.segsnr_cnn(idx) = computeSegmentalSNR(cleanSig, enhancedSig_cnn, fs);
        elseif hasCNN
            % Set to NaN if CNN processing failed (don't bias results)
            results.snr_cnn(idx) = NaN;
            results.sisnr_cnn(idx) = NaN;
            results.stoi_cnn(idx) = NaN;
            results.lsd_cnn(idx) = NaN;
            results.segsnr_cnn(idx) = NaN;
        end
    catch ME
        % If any metric computation fails, set to NaN
        warning('Error computing metrics for sample %d: %s', idx, ME.message);
    end
end

fprintf('\nBatch processing complete!\n\n');

% Report CNN processing statistics
if hasCNN
    fprintf('CNN Processing Summary:\n');
    fprintf('  Successful: %d/%d samples\n', cnnSuccessCount, numSamplesToProcess);
    fprintf('  Failed: %d/%d samples\n\n', cnnFailCount, numSamplesToProcess);
end

%% ========================================================================
%  SECTION 3: STATISTICAL ANALYSIS
% =========================================================================
% Compute summary statistics (mean, std) for each method
% Statistical analysis reveals:
% - Average performance (mean)
% - Consistency/reliability (standard deviation)
% - Relative improvements over baseline (Delta)
% =========================================================================
fprintf('SECTION 3: Statistical Analysis\n');
fprintf('--------------------------------\n\n');

% Remove NaN values for valid indices (samples that processed correctly)
validIdx = ~isnan(results.snr_noisy);

% For CNN, only use samples where processing was successful
if hasCNN
    validCNNIdx = validIdx & results.cnn_processing_success;
    fprintf('Valid CNN samples for analysis: %d\n\n', sum(validCNNIdx));
end

% Display all metrics with mean ± standard deviation
% Delta shows improvement over noisy baseline (higher is better for SNR, STOI)
fprintf('SNR (dB):\n');
fprintf('  Noisy:           %.3f +/- %.3f\n', mean(results.snr_noisy(validIdx)), std(results.snr_noisy(validIdx)));
fprintf('  Wiener:          %.3f +/- %.3f (Delta = %.3f)\n', ...
        mean(results.snr_wiener(validIdx)), std(results.snr_wiener(validIdx)), mean(results.snr_wiener(validIdx)) - mean(results.snr_noisy(validIdx)));
fprintf('  Improved Wiener: %.3f +/- %.3f (Delta = %.3f)\n', ...
        mean(results.snr_wiener_imp(validIdx)), std(results.snr_wiener_imp(validIdx)), mean(results.snr_wiener_imp(validIdx)) - mean(results.snr_noisy(validIdx)));
fprintf('  NLMS:            %.3f +/- %.3f (Delta = %.3f)\n', ...
        mean(results.snr_nlms(validIdx)), std(results.snr_nlms(validIdx)), mean(results.snr_nlms(validIdx)) - mean(results.snr_noisy(validIdx)));
if hasCNN && sum(validCNNIdx) > 0
    fprintf('  CNN:             %.3f +/- %.3f (Delta = %.3f)\n', ...
            mean(results.snr_cnn(validCNNIdx)), std(results.snr_cnn(validCNNIdx)), mean(results.snr_cnn(validCNNIdx)) - mean(results.snr_noisy(validCNNIdx)));
end
fprintf('\n');

fprintf('SI-SNR (dB):\n');
fprintf('  Noisy:           %.3f +/- %.3f\n', mean(results.sisnr_noisy(validIdx)), std(results.sisnr_noisy(validIdx)));
fprintf('  Wiener:          %.3f +/- %.3f (Delta = %.3f)\n', ...
        mean(results.sisnr_wiener(validIdx)), std(results.sisnr_wiener(validIdx)), mean(results.sisnr_wiener(validIdx)) - mean(results.sisnr_noisy(validIdx)));
fprintf('  Improved Wiener: %.3f +/- %.3f (Delta = %.3f)\n', ...
        mean(results.sisnr_wiener_imp(validIdx)), std(results.sisnr_wiener_imp(validIdx)), mean(results.sisnr_wiener_imp(validIdx)) - mean(results.sisnr_noisy(validIdx)));
fprintf('  NLMS:            %.3f +/- %.3f (Delta = %.3f)\n', ...
        mean(results.sisnr_nlms(validIdx)), std(results.sisnr_nlms(validIdx)), mean(results.sisnr_nlms(validIdx)) - mean(results.sisnr_noisy(validIdx)));
if hasCNN && sum(validCNNIdx) > 0
    fprintf('  CNN:             %.3f +/- %.3f (Delta = %.3f)\n', ...
            mean(results.sisnr_cnn(validCNNIdx)), std(results.sisnr_cnn(validCNNIdx)), mean(results.sisnr_cnn(validCNNIdx)) - mean(results.sisnr_noisy(validCNNIdx)));
end
fprintf('\n');

fprintf('STOI Score:\n');
fprintf('  Noisy:           %.3f +/- %.3f\n', mean(results.stoi_noisy(validIdx)), std(results.stoi_noisy(validIdx)));
fprintf('  Wiener:          %.3f +/- %.3f (Delta = %.3f)\n', ...
        mean(results.stoi_wiener(validIdx)), std(results.stoi_wiener(validIdx)), mean(results.stoi_wiener(validIdx)) - mean(results.stoi_noisy(validIdx)));
fprintf('  Improved Wiener: %.3f +/- %.3f (Delta = %.3f)\n', ...
        mean(results.stoi_wiener_imp(validIdx)), std(results.stoi_wiener_imp(validIdx)), mean(results.stoi_wiener_imp(validIdx)) - mean(results.stoi_noisy(validIdx)));
fprintf('  NLMS:            %.3f +/- %.3f (Delta = %.3f)\n', ...
        mean(results.stoi_nlms(validIdx)), std(results.stoi_nlms(validIdx)), mean(results.stoi_nlms(validIdx)) - mean(results.stoi_noisy(validIdx)));
if hasCNN && sum(validCNNIdx) > 0
    fprintf('  CNN:             %.3f +/- %.3f (Delta = %.3f)\n', ...
            mean(results.stoi_cnn(validCNNIdx)), std(results.stoi_cnn(validCNNIdx)), mean(results.stoi_cnn(validCNNIdx)) - mean(results.stoi_noisy(validCNNIdx)));
end
fprintf('\n');

fprintf('LSD (lower is better):\n');
fprintf('  Noisy:           %.3f +/- %.3f\n', mean(results.lsd_noisy(validIdx)), std(results.lsd_noisy(validIdx)));
fprintf('  Wiener:          %.3f +/- %.3f (Delta = %.3f)\n', ...
        mean(results.lsd_wiener(validIdx)), std(results.lsd_wiener(validIdx)), mean(results.lsd_wiener(validIdx)) - mean(results.lsd_noisy(validIdx)));
fprintf('  Improved Wiener: %.3f +/- %.3f (Delta = %.3f)\n', ...
        mean(results.lsd_wiener_imp(validIdx)), std(results.lsd_wiener_imp(validIdx)), mean(results.lsd_wiener_imp(validIdx)) - mean(results.lsd_noisy(validIdx)));
fprintf('  NLMS:            %.3f +/- %.3f (Delta = %.3f)\n', ...
        mean(results.lsd_nlms(validIdx)), std(results.lsd_nlms(validIdx)), mean(results.lsd_nlms(validIdx)) - mean(results.lsd_noisy(validIdx)));
if hasCNN && sum(validCNNIdx) > 0
    fprintf('  CNN:             %.3f +/- %.3f (Delta = %.3f)\n', ...
            mean(results.lsd_cnn(validCNNIdx)), std(results.lsd_cnn(validCNNIdx)), mean(results.lsd_cnn(validCNNIdx)) - mean(results.lsd_noisy(validCNNIdx)));
end
fprintf('\n');

fprintf('Segmental SNR (dB):\n');
fprintf('  Noisy:           %.3f +/- %.3f\n', mean(results.segsnr_noisy(validIdx)), std(results.segsnr_noisy(validIdx)));
fprintf('  Wiener:          %.3f +/- %.3f (Delta = %.3f)\n', ...
        mean(results.segsnr_wiener(validIdx)), std(results.segsnr_wiener(validIdx)), mean(results.segsnr_wiener(validIdx)) - mean(results.segsnr_noisy(validIdx)));
fprintf('  Improved Wiener: %.3f +/- %.3f (Delta = %.3f)\n', ...
        mean(results.segsnr_wiener_imp(validIdx)), std(results.segsnr_wiener_imp(validIdx)), mean(results.segsnr_wiener_imp(validIdx)) - mean(results.segsnr_noisy(validIdx)));
fprintf('  NLMS:            %.3f +/- %.3f (Delta = %.3f)\n', ...
        mean(results.segsnr_nlms(validIdx)), std(results.segsnr_nlms(validIdx)), mean(results.segsnr_nlms(validIdx)) - mean(results.segsnr_noisy(validIdx)));
if hasCNN && sum(validCNNIdx) > 0
    fprintf('  CNN:             %.3f +/- %.3f (Delta = %.3f)\n', ...
            mean(results.segsnr_cnn(validCNNIdx)), std(results.segsnr_cnn(validCNNIdx)), mean(results.segsnr_cnn(validCNNIdx)) - mean(results.segsnr_noisy(validCNNIdx)));
end
fprintf('\n');

%% ========================================================================
%  SECTION 4: COMPREHENSIVE VISUALIZATIONS
% =========================================================================
% Create visual representations of results for better understanding
% Visualizations show:
% - Overall performance comparison (bar charts with error bars)
% - Performance breakdown by noise type
% - Performance breakdown by SNR level
% =========================================================================
fprintf('SECTION 4: Generating Visualizations\n');
fprintf('-------------------------------------\n');

% Create output directory for saving results
outputDir = 'comparative_results';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Figure 1: Overall Performance Comparison
% Shows mean and standard deviation for each metric and method
figure('Name', 'Overall Performance Comparison', 'Position', [50 50 1400 900]);

% Define method names for plotting
methods = {'Noisy', 'Wiener', 'Improved Wiener', 'NLMS'};
if hasCNN && sum(validCNNIdx) > 0
    methods{end+1} = 'CNN';
end

% SNR Comparison (higher is better)
subplot(2,3,1);
snrData = [mean(results.snr_noisy(validIdx)), mean(results.snr_wiener(validIdx)), ...
           mean(results.snr_wiener_imp(validIdx)), mean(results.snr_nlms(validIdx))];
snrStd = [std(results.snr_noisy(validIdx)), std(results.snr_wiener(validIdx)), ...
          std(results.snr_wiener_imp(validIdx)), std(results.snr_nlms(validIdx))];
if hasCNN && sum(validCNNIdx) > 0
    snrData(end+1) = mean(results.snr_cnn(validCNNIdx));
    snrStd(end+1) = std(results.snr_cnn(validCNNIdx));
end
bar(snrData);
hold on;
errorbar(1:length(snrData), snrData, snrStd, 'k.', 'LineWidth', 1.5); % Error bars show variability
hold off;
set(gca, 'XTickLabel', methods);
ylabel('SNR (dB)');
title('Signal-to-Noise Ratio');
grid on;
xtickangle(45);

% SI-SNR Comparison (higher is better)
subplot(2,3,2);
sisnrData = [mean(results.sisnr_noisy(validIdx)), mean(results.sisnr_wiener(validIdx)), ...
             mean(results.sisnr_wiener_imp(validIdx)), mean(results.sisnr_nlms(validIdx))];
sisnrStd = [std(results.sisnr_noisy(validIdx)), std(results.sisnr_wiener(validIdx)), ...
            std(results.sisnr_wiener_imp(validIdx)), std(results.sisnr_nlms(validIdx))];
if hasCNN && sum(validCNNIdx) > 0
    sisnrData(end+1) = mean(results.sisnr_cnn(validCNNIdx));
    sisnrStd(end+1) = std(results.sisnr_cnn(validCNNIdx));
end
bar(sisnrData);
hold on;
errorbar(1:length(sisnrData), sisnrData, sisnrStd, 'k.', 'LineWidth', 1.5);
hold off;
set(gca, 'XTickLabel', methods);
ylabel('SI-SNR (dB)');
title('Scale-Invariant SNR');
grid on;
xtickangle(45);

% STOI Comparison (higher is better, range 0-1)
subplot(2,3,3);
stoiData = [mean(results.stoi_noisy(validIdx)), mean(results.stoi_wiener(validIdx)), ...
            mean(results.stoi_wiener_imp(validIdx)), mean(results.stoi_nlms(validIdx))];
stoiStd = [std(results.stoi_noisy(validIdx)), std(results.stoi_wiener(validIdx)), ...
           std(results.stoi_wiener_imp(validIdx)), std(results.stoi_nlms(validIdx))];
if hasCNN && sum(validCNNIdx) > 0
    stoiData(end+1) = mean(results.stoi_cnn(validCNNIdx));
    stoiStd(end+1) = std(results.stoi_cnn(validCNNIdx));
end
bar(stoiData);
hold on;
errorbar(1:length(stoiData), stoiData, stoiStd, 'k.', 'LineWidth', 1.5);
hold off;
set(gca, 'XTickLabel', methods);
ylabel('STOI Score');
title('Speech Intelligibility');
grid on;
xtickangle(45);

% LSD Comparison (lower is better)
subplot(2,3,4);
lsdData = [mean(results.lsd_noisy(validIdx)), mean(results.lsd_wiener(validIdx)), ...
           mean(results.lsd_wiener_imp(validIdx)), mean(results.lsd_nlms(validIdx))];
lsdStd = [std(results.lsd_noisy(validIdx)), std(results.lsd_wiener(validIdx)), ...
          std(results.lsd_wiener_imp(validIdx)), std(results.lsd_nlms(validIdx))];
if hasCNN && sum(validCNNIdx) > 0
    lsdData(end+1) = mean(results.lsd_cnn(validCNNIdx));
    lsdStd(end+1) = std(results.lsd_cnn(validCNNIdx));
end
bar(lsdData);
hold on;
errorbar(1:length(lsdData), lsdData, lsdStd, 'k.', 'LineWidth', 1.5);
hold off;
set(gca, 'XTickLabel', methods);
ylabel('LSD');
title('Log-Spectral Distortion (lower is better)');
grid on;
xtickangle(45);

% Segmental SNR Comparison (higher is better)
subplot(2,3,5);
segsnrData = [mean(results.segsnr_noisy(validIdx)), mean(results.segsnr_wiener(validIdx)), ...
              mean(results.segsnr_wiener_imp(validIdx)), mean(results.segsnr_nlms(validIdx))];
segsnrStd = [std(results.segsnr_noisy(validIdx)), std(results.segsnr_wiener(validIdx)), ...
             std(results.segsnr_wiener_imp(validIdx)), std(results.segsnr_nlms(validIdx))];
if hasCNN && sum(validCNNIdx) > 0
    segsnrData(end+1) = mean(results.segsnr_cnn(validCNNIdx));
    segsnrStd(end+1) = std(results.segsnr_cnn(validCNNIdx));
end
bar(segsnrData);
hold on;
errorbar(1:length(segsnrData), segsnrData, segsnrStd, 'k.', 'LineWidth', 1.5);
hold off;
set(gca, 'XTickLabel', methods);
ylabel('Segmental SNR (dB)');
title('Segmental SNR');
grid on;
xtickangle(45);

% SNR Improvement over Noisy Baseline (key summary plot)
subplot(2,3,6);
improvementData = [0, ... % Noisy baseline (no improvement)
                   mean(results.snr_wiener(validIdx)) - mean(results.snr_noisy(validIdx)), ...
                   mean(results.snr_wiener_imp(validIdx)) - mean(results.snr_noisy(validIdx)), ...
                   mean(results.snr_nlms(validIdx)) - mean(results.snr_noisy(validIdx))];
if hasCNN && sum(validCNNIdx) > 0
    improvementData(end+1) = mean(results.snr_cnn(validCNNIdx)) - mean(results.snr_noisy(validCNNIdx));
end
bar(improvementData);
set(gca, 'XTickLabel', methods);
ylabel('SNR Improvement (dB)');
title('SNR Improvement Over Noisy');
grid on;
yline(0, 'r--', 'LineWidth', 1.5); % Zero line shows baseline
xtickangle(45);

saveas(gcf, fullfile(outputDir, 'overall_performance_comparison.png'));
fprintf('  Saved: overall_performance_comparison.png\n');

% Figure 2: Performance by Noise Type
% Shows how each method performs for different types of noise
% Some methods may be better suited for certain noise characteristics
figure('Name', 'Performance by Noise Type', 'Position', [100 100 1400 700]);

uniqueNoises = unique(results.noiseTypes); % Get list of noise types
numNoises = length(uniqueNoises);

% SNR by noise type (grouped bar chart)
subplot(1,2,1);
snrByNoise = zeros(numNoises, length(methods));
for i = 1:numNoises
    noiseIdx = strcmp(results.noiseTypes, uniqueNoises{i}) & validIdx;
    snrByNoise(i,1) = mean(results.snr_noisy(noiseIdx));
    snrByNoise(i,2) = mean(results.snr_wiener(noiseIdx));
    snrByNoise(i,3) = mean(results.snr_wiener_imp(noiseIdx));
    snrByNoise(i,4) = mean(results.snr_nlms(noiseIdx));
    if hasCNN && sum(validCNNIdx) > 0
        cnnNoiseIdx = strcmp(results.noiseTypes, uniqueNoises{i}) & validCNNIdx;
        if sum(cnnNoiseIdx) > 0
            snrByNoise(i,5) = mean(results.snr_cnn(cnnNoiseIdx));
        else
            snrByNoise(i,5) = NaN;
        end
    end
end
bar(snrByNoise);
set(gca, 'XTickLabel', uniqueNoises);
ylabel('Mean SNR (dB)');
title('SNR by Noise Type');
legend(methods, 'Location', 'best');
grid on;
xtickangle(45);

% STOI by noise type
subplot(1,2,2);
stoiByNoise = zeros(numNoises, length(methods));
for i = 1:numNoises
    noiseIdx = strcmp(results.noiseTypes, uniqueNoises{i}) & validIdx;
    stoiByNoise(i,1) = mean(results.stoi_noisy(noiseIdx));
    stoiByNoise(i,2) = mean(results.stoi_wiener(noiseIdx));
    stoiByNoise(i,3) = mean(results.stoi_wiener_imp(noiseIdx));
    stoiByNoise(i,4) = mean(results.stoi_nlms(noiseIdx));
    if hasCNN && sum(validCNNIdx) > 0
        cnnNoiseIdx = strcmp(results.noiseTypes, uniqueNoises{i}) & validCNNIdx;
        if sum(cnnNoiseIdx) > 0
            stoiByNoise(i,5) = mean(results.stoi_cnn(cnnNoiseIdx));
        else
            stoiByNoise(i,5) = NaN;
        end
    end
end
bar(stoiByNoise);
set(gca, 'XTickLabel', uniqueNoises);
ylabel('Mean STOI Score');
title('STOI by Noise Type');
legend(methods, 'Location', 'best');
grid on;
xtickangle(45);

saveas(gcf, fullfile(outputDir, 'performance_by_noise_type.png'));
fprintf('  Saved: performance_by_noise_type.png\n');

% Figure 3: Performance by SNR Level
% Shows how methods perform at different noise levels
% Important: Methods that work at low SNR are more robust
figure('Name', 'Performance by SNR Level', 'Position', [150 150 1200 700]);

uniqueSNRs = unique(results.snrLevels); % Get list of SNR levels
numSNRs = length(uniqueSNRs);

% SNR improvement by input SNR level
subplot(1,2,1);
snrImpBySNR = zeros(numSNRs, length(methods)-1); % Exclude noisy baseline
for i = 1:numSNRs
    snrIdx = strcmp(results.snrLevels, uniqueSNRs{i}) & validIdx;
    baselineSNR = mean(results.snr_noisy(snrIdx));
    snrImpBySNR(i,1) = mean(results.snr_wiener(snrIdx)) - baselineSNR;
    snrImpBySNR(i,2) = mean(results.snr_wiener_imp(snrIdx)) - baselineSNR;
    snrImpBySNR(i,3) = mean(results.snr_nlms(snrIdx)) - baselineSNR;
    if hasCNN && sum(validCNNIdx) > 0
        cnnSNRIdx = strcmp(results.snrLevels, uniqueSNRs{i}) & validCNNIdx;
        if sum(cnnSNRIdx) > 0
            snrImpBySNR(i,4) = mean(results.snr_cnn(cnnSNRIdx)) - mean(results.snr_noisy(cnnSNRIdx));
        else
            snrImpBySNR(i,4) = NaN;
        end
    end
end
bar(snrImpBySNR);
set(gca, 'XTickLabel', uniqueSNRs);
ylabel('SNR Improvement (dB)');
title('SNR Improvement by Input SNR Level');
legend(methods(2:end), 'Location', 'best'); % Exclude 'Noisy' from legend
grid on;
yline(0, 'r--', 'LineWidth', 1.5); % Zero line

% STOI improvement by input SNR level
subplot(1,2,2);
stoiImpBySNR = zeros(numSNRs, length(methods)-1);
for i = 1:numSNRs
    snrIdx = strcmp(results.snrLevels, uniqueSNRs{i}) & validIdx;
    baselineSTOI = mean(results.stoi_noisy(snrIdx));
    stoiImpBySNR(i,1) = mean(results.stoi_wiener(snrIdx)) - baselineSTOI;
    stoiImpBySNR(i,2) = mean(results.stoi_wiener_imp(snrIdx)) - baselineSTOI;
    stoiImpBySNR(i,3) = mean(results.stoi_nlms(snrIdx)) - baselineSTOI;
    if hasCNN && sum(validCNNIdx) > 0
        cnnSNRIdx = strcmp(results.snrLevels, uniqueSNRs{i}) & validCNNIdx;
        if sum(cnnSNRIdx) > 0
            stoiImpBySNR(i,4) = mean(results.stoi_cnn(cnnSNRIdx)) - mean(results.stoi_noisy(cnnSNRIdx));
        else
            stoiImpBySNR(i,4) = NaN;
        end
    end
end
bar(stoiImpBySNR);
set(gca, 'XTickLabel', uniqueSNRs);
ylabel('STOI Improvement');
title('STOI Improvement by Input SNR Level');
legend(methods(2:end), 'Location', 'best');
grid on;
yline(0, 'r--', 'LineWidth', 1.5);

saveas(gcf, fullfile(outputDir, 'performance_by_snr_level.png'));
fprintf('  Saved: performance_by_snr_level.png\n');

fprintf('\nAll visualizations complete!\n\n');

%% ========================================================================
%  SECTION 5: SAVE COMPREHENSIVE RESULTS
% =========================================================================
% Save all results and generate a detailed text report
% Report provides human-readable summary of findings
% =========================================================================
fprintf('SECTION 5: Saving Results\n');
fprintf('-------------------------\n');

% Save results structure (for further analysis in MATLAB)
save(fullfile(outputDir, 'comprehensive_results.mat'), 'results');
fprintf('Results saved to: %s\n', fullfile(outputDir, 'comprehensive_results.mat'));

% Generate detailed text report
fid = fopen(fullfile(outputDir, 'evaluation_report.txt'), 'w');
fprintf(fid, '========================================\n');
fprintf(fid, 'COMPREHENSIVE EVALUATION REPORT\n');
fprintf(fid, 'Speech Enhancement Project\n');
fprintf(fid, '========================================\n\n');

fprintf(fid, 'Dataset: NOIZEUS\n');
fprintf(fid, 'Samples evaluated: %d\n', numSamplesToProcess);
fprintf(fid, 'Methods compared: %s\n\n', strjoin(methods, ', '));

if hasCNN
    fprintf(fid, 'CNN Processing Statistics:\n');
    fprintf(fid, '  Successful samples: %d/%d (%.1f%%)\n', ...
            cnnSuccessCount, numSamplesToProcess, 100*cnnSuccessCount/numSamplesToProcess);
    fprintf(fid, '  Failed samples: %d/%d\n\n', cnnFailCount, numSamplesToProcess);
end

fprintf(fid, '----------------------------------------\n');
fprintf(fid, 'OVERALL PERFORMANCE SUMMARY\n');
fprintf(fid, '----------------------------------------\n\n');

fprintf(fid, 'SNR (dB):\n');
fprintf(fid, '  Noisy:           %.2f +/- %.2f\n', mean(results.snr_noisy(validIdx)), std(results.snr_noisy(validIdx)));
fprintf(fid, '  Wiener:          %.2f +/- %.2f (Delta = %.2f dB)\n', ...
        mean(results.snr_wiener(validIdx)), std(results.snr_wiener(validIdx)), ...
        mean(results.snr_wiener(validIdx)) - mean(results.snr_noisy(validIdx)));
fprintf(fid, '  Improved Wiener: %.2f +/- %.2f (Delta = %.2f dB)\n', ...
        mean(results.snr_wiener_imp(validIdx)), std(results.snr_wiener_imp(validIdx)), ...
        mean(results.snr_wiener_imp(validIdx)) - mean(results.snr_noisy(validIdx)));
fprintf(fid, '  NLMS:            %.2f +/- %.2f (Delta = %.2f dB)\n', ...
        mean(results.snr_nlms(validIdx)), std(results.snr_nlms(validIdx)), ...
        mean(results.snr_nlms(validIdx)) - mean(results.snr_noisy(validIdx)));
if hasCNN && sum(validCNNIdx) > 0
    fprintf(fid, '  CNN:             %.2f +/- %.2f (Delta = %.2f dB)\n', ...
            mean(results.snr_cnn(validCNNIdx)), std(results.snr_cnn(validCNNIdx)), ...
            mean(results.snr_cnn(validCNNIdx)) - mean(results.snr_noisy(validCNNIdx)));
end
fprintf(fid, '\n');

fprintf(fid, 'STOI Score:\n');
fprintf(fid, '  Noisy:           %.3f +/- %.3f\n', mean(results.stoi_noisy(validIdx)), std(results.stoi_noisy(validIdx)));
fprintf(fid, '  Wiener:          %.3f +/- %.3f (Delta = %.3f)\n', ...
        mean(results.stoi_wiener(validIdx)), std(results.stoi_wiener(validIdx)), ...
        mean(results.stoi_wiener(validIdx)) - mean(results.stoi_noisy(validIdx)));
fprintf(fid, '  Improved Wiener: %.3f +/- %.3f (Delta = %.3f)\n', ...
        mean(results.stoi_wiener_imp(validIdx)), std(results.stoi_wiener_imp(validIdx)), ...
        mean(results.stoi_wiener_imp(validIdx)) - mean(results.stoi_noisy(validIdx)));
fprintf(fid, '  NLMS:            %.3f +/- %.3f (Delta = %.3f)\n', ...
        mean(results.stoi_nlms(validIdx)), std(results.stoi_nlms(validIdx)), ...
        mean(results.stoi_nlms(validIdx)) - mean(results.stoi_noisy(validIdx)));
if hasCNN && sum(validCNNIdx) > 0
    fprintf(fid, '  CNN:             %.3f +/- %.3f (Delta = %.3f)\n', ...
            mean(results.stoi_cnn(validCNNIdx)), std(results.stoi_cnn(validCNNIdx)), ...
            mean(results.stoi_cnn(validCNNIdx)) - mean(results.stoi_noisy(validCNNIdx)));
end
fprintf(fid, '\n');

fprintf(fid, '----------------------------------------\n');
fprintf(fid, 'KEY FINDINGS\n');
fprintf(fid, '----------------------------------------\n\n');

% Determine best method for each metric
snrVals = [mean(results.snr_wiener(validIdx)), ...
           mean(results.snr_wiener_imp(validIdx)), ...
           mean(results.snr_nlms(validIdx))];
stoiVals = [mean(results.stoi_wiener(validIdx)), ...
            mean(results.stoi_wiener_imp(validIdx)), ...
            mean(results.stoi_nlms(validIdx))];

if hasCNN && sum(validCNNIdx) > 0
    snrVals(end+1) = mean(results.snr_cnn(validCNNIdx));
    stoiVals(end+1) = mean(results.stoi_cnn(validCNNIdx));
end

[~, bestSNR] = max(snrVals);
[~, bestSTOI] = max(stoiVals);

bestMethodsSNR = methods(2:end); % Exclude 'Noisy'
bestMethodsSTOI = methods(2:end);

fprintf(fid, '1. Best SNR improvement: %s\n', bestMethodsSNR{bestSNR});
fprintf(fid, '2. Best STOI improvement: %s\n', bestMethodsSTOI{bestSTOI});
fprintf(fid, '3. Most challenging noise types:\n');

% Find hardest noise types (smallest improvement)
avgImpByNoise = zeros(numNoises, 1);
for i = 1:numNoises
    noiseIdx = strcmp(results.noiseTypes, uniqueNoises{i}) & validIdx;
    avgImpByNoise(i) = mean(results.snr_wiener_imp(noiseIdx)) - mean(results.snr_noisy(noiseIdx));
end
[~, sortIdx] = sort(avgImpByNoise); % Sort ascending (smallest improvement first)
for i = 1:min(3, numNoises)
    fprintf(fid, '   - %s (%.2f dB improvement)\n', uniqueNoises{sortIdx(i)}, avgImpByNoise(sortIdx(i)));
end

fprintf(fid, '\n');
fprintf(fid, '----------------------------------------\n');
fprintf(fid, 'CONCLUSION\n');
fprintf(fid, '----------------------------------------\n\n');

if hasCNN && sum(validCNNIdx) > 0
    fprintf(fid, 'This evaluation compared traditional DSP methods\n');
    fprintf(fid, '(Wiener, Improved Wiener, NLMS) with a CNN-based\n');
    fprintf(fid, 'machine learning approach for speech enhancement.\n\n');
    
    if cnnFailCount > 0
        fprintf(fid, 'Note: CNN processing failed for %d samples.\n', cnnFailCount);
        fprintf(fid, 'CNN metrics are based on %d successful samples only.\n\n', cnnSuccessCount);
    end
else
    fprintf(fid, 'This evaluation compared three traditional DSP methods\n');
    fprintf(fid, 'for speech enhancement: Wiener filtering, Improved\n');
    fprintf(fid, 'Wiener filtering, and NLMS adaptive filtering.\n\n');
end

fclose(fid);
fprintf('Report saved to: %s\n\n', fullfile(outputDir, 'evaluation_report.txt'));

%% ========================================================================
%  SECTION 6: SUMMARY
% =========================================================================
fprintf('========================================\n');
fprintf('Stage 4 Complete!\n');
fprintf('========================================\n');
fprintf('\nAll results saved to: %s\n', outputDir);
fprintf('  - comprehensive_results.mat\n');
fprintf('  - evaluation_report.txt\n');
fprintf('  - overall_performance_comparison.png\n');
fprintf('  - performance_by_noise_type.png\n');
fprintf('  - performance_by_snr_level.png\n\n');

if hasCNN && cnnFailCount > 0
    fprintf('WARNING: CNN processing had %d failures.\n', cnnFailCount);
    fprintf('Check the error messages above for details.\n');
    fprintf('Common issues:\n');
    fprintf('  - Input dimensions exceed maxFreqBins/maxTimeBins\n');
    fprintf('  - Missing or incompatible model file\n');
    fprintf('  - STFT parameters mismatch between training and evaluation\n\n');
end

fprintf('PROJECT COMPLETE!\n');
fprintf('========================================\n');

%% ========================================================================
%  HELPER FUNCTIONS
% =========================================================================
% These functions implement the evaluation metrics used above
% Each function takes clean and enhanced signals as input
% and returns a single metric value

% SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
% =========================================================================
% Scale-invariant version of SNR that's robust to amplitude differences
% Commonly used in source separation evaluation
%
% Formula:
%   s_target = (estimate' * reference) / ||reference||^2 * reference
%   e_noise = estimate - s_target
%   SI-SNR = 10 * log10(||s_target||^2 / ||e_noise||^2)
%
% The projection step makes it invariant to scaling
% =========================================================================
function sisnr = compute_sisnr(reference, estimate)
    % Ensure same length
    minLen = min(length(reference), length(estimate));
    reference = reference(1:minLen);
    estimate = estimate(1:minLen);
    
    % Remove mean (zero-mean assumption)
    reference = reference - mean(reference);
    estimate = estimate - mean(estimate);
    
    % Compute scale-invariant projection
    % This finds the scaling factor that best aligns estimate with reference
    alpha = (reference' * estimate) / (reference' * reference + eps);
    s_target = alpha * reference; % Scaled target
    e_noise = estimate - s_target; % Residual noise
    
    % Compute SI-SNR in dB
    sisnr = 10 * log10(sum(s_target.^2) / (sum(e_noise.^2) + eps));
end

% STOI (Short-Time Objective Intelligibility) - Simplified version
% =========================================================================
% Measures speech intelligibility (how understandable is the speech)
% This is a SIMPLIFIED approximation of full STOI algorithm
% Full STOI requires STOI toolbox from DTU
%
% Approach: Compute correlation between clean and enhanced in short frames
% High correlation = good intelligibility preservation
% =========================================================================
function stoi_score = compute_stoi_simplified(clean, enhanced, fs)
    % Frame parameters (standard STOI uses 25.6ms frames)
    frameLen = round(0.030 * fs); % 30ms frames
    hopSize = round(0.015 * fs);  % 15ms hop
    
    % Ensure same length
    minLen = min(length(clean), length(enhanced));
    clean = clean(1:minLen);
    enhanced = enhanced(1:minLen);
    
    % Compute correlation in time frames
    numFrames = floor((minLen - frameLen) / hopSize) + 1;
    correlations = zeros(numFrames, 1);
    
    for i = 1:numFrames
        startIdx = (i-1) * hopSize + 1;
        endIdx = startIdx + frameLen - 1;
        
        cleanFrame = clean(startIdx:endIdx);
        enhancedFrame = enhanced(startIdx:endIdx);
        
        % Normalize frames to unit norm
        cleanFrame = cleanFrame / (norm(cleanFrame) + eps);
        enhancedFrame = enhancedFrame / (norm(enhancedFrame) + eps);
        
        % Compute correlation (dot product of normalized vectors)
        correlations(i) = abs(cleanFrame' * enhancedFrame);
    end
    
    % Average correlation as STOI approximation
    stoi_score = mean(correlations);
end

% Log-Spectral Distortion
% =========================================================================
% Measures distortion in the spectral domain
% Lower LSD means better spectral fidelity
%
% Formula: LSD = mean over frames of sqrt(mean over freq of (log diff)^2)
% Computed in log-magnitude spectrum (dB scale)
% =========================================================================
function lsd = compute_lsd(clean, enhanced, fs, winLen, hopSize, nfft)
    % Compute spectrograms
    [S_clean, ~, ~] = stft(clean, fs, 'Window', hamming(winLen, 'periodic'), ...
                           'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    [S_enhanced, ~, ~] = stft(enhanced, fs, 'Window', hamming(winLen, 'periodic'), ...
                              'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    
    % Compute magnitude spectra
    magClean = abs(S_clean);
    magEnhanced = abs(S_enhanced);
    
    % Ensure same size (handle edge effects)
    minCols = min(size(magClean, 2), size(magEnhanced, 2));
    magClean = magClean(:, 1:minCols);
    magEnhanced = magEnhanced(:, 1:minCols);
    
    % Compute log-spectral distortion
    epsilon = 1e-10; % Avoid log(0)
    logDiff = log10(magClean + epsilon) - log10(magEnhanced + epsilon);
    % For each frame: sqrt of mean squared difference across frequencies
    % Then average across all frames
    lsd = mean(sqrt(mean(logDiff.^2, 1)));
end

% computeSegmentalSNR - Segmental SNR calculation
% =========================================================================
% Computes SNR in short segments and averages
% More perceptually relevant than global SNR because:
% 1. Weights all segments equally (not dominated by high-energy regions)
% 2. Reflects local variations in enhancement quality
%
% Standard approach: 32ms frames with 16ms hop, remove outliers
% =========================================================================
function segSNR = computeSegmentalSNR(cleanSignal, noisySignal, fs)
    % Frame parameters
    frameLen = round(0.032 * fs); % 32ms frames
    hopSize = round(0.016 * fs);  % 16ms hop
    
    % Ensure same length
    minLen = min(length(cleanSignal), length(noisySignal));
    cleanSignal = cleanSignal(1:minLen);
    noisySignal = noisySignal(1:minLen);
    
    % Compute frame-by-frame SNR
    numFrames = floor((minLen - frameLen) / hopSize) + 1;
    frameSNR = zeros(numFrames, 1);
    
    for i = 1:numFrames
        startIdx = (i-1) * hopSize + 1;
        endIdx = startIdx + frameLen - 1;
        
        cleanFrame = cleanSignal(startIdx:endIdx);
        noisyFrame = noisySignal(startIdx:endIdx);
        
        % Compute noise as difference
        noise = noisyFrame - cleanFrame;
        
        % Compute powers
        signalPower = sum(cleanFrame.^2);
        noisePower = sum(noise.^2);
        
        % Compute SNR for this frame (avoid division by zero)
        if noisePower > 0 && signalPower > 0
            frameSNR(i) = 10 * log10(signalPower / noisePower);
        else
            frameSNR(i) = 0;
        end
    end
    
    % Remove outliers and compute mean
    % Outliers can occur in silent regions or pure noise frames
    % Typical range: -20 dB to 40 dB
    frameSNR = frameSNR(frameSNR > -20 & frameSNR < 40);
    segSNR = mean(frameSNR);
end
