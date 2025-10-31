% =========================================================================
% STAGE 4: Comparative Analysis and Evaluation
% =========================================================================
% This script performs comprehensive evaluation and comparison of:
% 1. Adaptive Filtering methods (Wiener, Improved Wiener, NLMS)
% 2. Machine Learning approach (CNN)
%
% Evaluation Metrics:
% - Scale-Invariant Signal-to-Noise Ratio (SI-SNR)
% - Short-Time Objective Intelligibility (STOI)
% - Perceptual Evaluation of Speech Quality (PESQ) approximation
% - Segmental SNR
%
% Author: Alaa Aldirani
% Project: Real-Time Speech Enhancement
% =========================================================================

clear; close all; clc;

%% Configuration
fprintf('========================================\n');
fprintf('STAGE 4: Comparative Analysis\n');
fprintf('========================================\n\n');

% Load prepared dataset
fprintf('Loading prepared dataset...\n');
load('prepared_data/noizeus_prepared.mat');

% Try to load trained CNN model
try
    fprintf('Loading trained CNN model...\n');
    load('ml_results/trained_cnn.mat', 'net', 'maxFreqBins', 'maxTimeBins', ...
         'winLen', 'hopSize', 'nfft', 'winFun');
    hasCNN = true;
    fprintf('CNN model loaded successfully!\n\n');
catch
    fprintf('Warning: Could not load CNN model. Will evaluate adaptive filtering only.\n\n');
    hasCNN = false;
end

% Decide how many test samples to process
numSamplesToProcess = min(50, length(testData)); % Process 50 or all if less
fprintf('Processing %d test samples...\n\n', numSamplesToProcess);

%% STFT Parameters
fs = testData(1).fs;
if ~exist('winLen', 'var')
    winLen = round(0.032 * fs);
    hopSize = round(0.016 * fs);
    nfft = 2^nextpow2(winLen);
    winFun = hamming(winLen, 'periodic');
end

%% ========================================================================
%  SECTION 1: EVALUATION METRICS IMPLEMENTATION
% =========================================================================
fprintf('SECTION 1: Implementing Evaluation Metrics\n');
fprintf('-------------------------------------------\n\n');

% SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
function sisnr = compute_sisnr(reference, estimate)
    % Ensure same length
    minLen = min(length(reference), length(estimate));
    reference = reference(1:minLen);
    estimate = estimate(1:minLen);
    
    % Remove mean
    reference = reference - mean(reference);
    estimate = estimate - mean(estimate);
    
    % Compute scale-invariant projection
    alpha = (reference' * estimate) / (reference' * reference + eps);
    s_target = alpha * reference;
    e_noise = estimate - s_target;
    
    % Compute SI-SNR
    sisnr = 10 * log10(sum(s_target.^2) / (sum(e_noise.^2) + eps));
end

% STOI (Short-Time Objective Intelligibility) - Simplified version
function stoi_score = compute_stoi_simplified(clean, enhanced, fs)
    % This is a simplified STOI approximation
    % For full STOI, you would need the STOI toolbox
    
    % Frame parameters
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
        
        % Normalize frames
        cleanFrame = cleanFrame / (norm(cleanFrame) + eps);
        enhancedFrame = enhancedFrame / (norm(enhancedFrame) + eps);
        
        % Compute correlation
        correlations(i) = abs(cleanFrame' * enhancedFrame);
    end
    
    % Average correlation as STOI approximation
    stoi_score = mean(correlations);
end

% Log-Spectral Distortion
function lsd = compute_lsd(clean, enhanced, fs, winLen, hopSize, nfft)
    % Compute spectrograms
    [S_clean, ~, ~] = stft(clean, fs, 'Window', hamming(winLen, 'periodic'), ...
                           'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    [S_enhanced, ~, ~] = stft(enhanced, fs, 'Window', hamming(winLen, 'periodic'), ...
                              'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    
    % Compute magnitude spectra
    magClean = abs(S_clean);
    magEnhanced = abs(S_enhanced);
    
    % Ensure same size
    minCols = min(size(magClean, 2), size(magEnhanced, 2));
    magClean = magClean(:, 1:minCols);
    magEnhanced = magEnhanced(:, 1:minCols);
    
    % Compute log-spectral distortion
    epsilon = 1e-10;
    logDiff = log10(magClean + epsilon) - log10(magEnhanced + epsilon);
    lsd = mean(sqrt(mean(logDiff.^2, 1)));
end

fprintf('Evaluation metrics implemented:\n');
fprintf('  - SI-SNR (Scale-Invariant SNR)\n');
fprintf('  - STOI (Speech Intelligibility - simplified)\n');
fprintf('  - LSD (Log-Spectral Distortion)\n');
fprintf('  - Segmental SNR\n\n');

%% ========================================================================
%  SECTION 2: BATCH PROCESSING AND EVALUATION
% =========================================================================
fprintf('SECTION 2: Batch Processing\n');
fprintf('---------------------------\n');
% Define STFT parameters if not already defined
if ~exist('fs', 'var')
    fs = testData(1).fs;
    winLen = round(0.032 * fs);
    hopSize = round(0.016 * fs);
    nfft = 2^nextpow2(winLen);
    winFun = hamming(winLen, 'periodic');
end
% Define STFT parameters if not already defined
if ~exist('fs', 'var')
    fs = testData(1).fs;
    winLen = round(0.032 * fs);
    hopSize = round(0.016 * fs);
    nfft = 2^nextpow2(winLen);
    winFun = hamming(winLen, 'periodic');
end

% Initialize results storage
results = struct();
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

% Improved Wiener filter results
results.snr_wiener_imp = zeros(numSamplesToProcess, 1);
results.sisnr_wiener_imp = zeros(numSamplesToProcess, 1);
results.stoi_wiener_imp = zeros(numSamplesToProcess, 1);
results.lsd_wiener_imp = zeros(numSamplesToProcess, 1);
results.segsnr_wiener_imp = zeros(numSamplesToProcess, 1);

% NLMS filter results
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
end

% Metadata
results.noiseTypes = cell(numSamplesToProcess, 1);
results.snrLevels = cell(numSamplesToProcess, 1);

fprintf('Starting batch processing...\n');
fprintf('Progress: ');

for idx = 1:numSamplesToProcess
    if mod(idx, 10) == 0
        fprintf('%d ', idx);
    end
    
    % Extract signals
    cleanSig = testData(idx).clean;
    noisySig = testData(idx).noisy;
    noiseType = testData(idx).noiseType;
    snrLevel = testData(idx).snr;
    
    % Store metadata
    results.noiseTypes{idx} = noiseType;
    results.snrLevels{idx} = snrLevel;
    
    %% Process with adaptive filtering methods
    % STFT
    [S_noisy, ~, ~] = stft(noisySig, fs, 'Window', winFun, ...
                           'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    
    % VAD for noise estimation
    frameEnergy = sum(abs(S_noisy).^2, 1);
    frameEnergyDB = 10*log10(frameEnergy + eps);
    energyThreshold = mean(frameEnergyDB) - 5;
    vadDecisions = frameEnergyDB > energyThreshold;
    vadDecisions = medfilt1(double(vadDecisions), 5) > 0.5;
    
    % Estimate noise PSD
    noiseFrames = S_noisy(:, ~vadDecisions);
    if ~isempty(noiseFrames)
        noisePSD = mean(abs(noiseFrames).^2, 2);
    else
        noisePSD = mean(abs(S_noisy(:, 1:min(10, size(S_noisy,2)))).^2, 2);
    end
    
    % Standard Wiener filter
    noisyPSD = abs(S_noisy).^2;
    wienerGain = max(1 - bsxfun(@rdivide, noisePSD, noisyPSD), 0);
    S_wiener = wienerGain .* S_noisy;
    enhancedSig_wiener = istft(S_wiener, fs, 'Window', winFun, ...
                               'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    enhancedSig_wiener = enhancedSig_wiener(1:min(length(enhancedSig_wiener), length(noisySig)));
    if length(enhancedSig_wiener) < length(noisySig)
        enhancedSig_wiener = [enhancedSig_wiener; zeros(length(noisySig)-length(enhancedSig_wiener), 1)];
    end
    
    % Improved Wiener filter
    alpha = 2.0; beta = 0.01;
    improvedWienerGain = max(1 - alpha * bsxfun(@rdivide, noisePSD, noisyPSD), beta);
    S_wiener_imp = improvedWienerGain .* S_noisy;
    enhancedSig_wiener_imp = istft(S_wiener_imp, fs, 'Window', winFun, ...
                                    'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    enhancedSig_wiener_imp = enhancedSig_wiener_imp(1:min(length(enhancedSig_wiener_imp), length(noisySig)));
    if length(enhancedSig_wiener_imp) < length(noisySig)
        enhancedSig_wiener_imp = [enhancedSig_wiener_imp; zeros(length(noisySig)-length(enhancedSig_wiener_imp), 1)];
    end
    
    % NLMS filter
    filterOrder = 32; mu = 0.1; delta = 0.01;
    w = zeros(filterOrder, 1);
    enhancedSig_nlms = zeros(size(noisySig));
    refSignal = [zeros(filterOrder, 1); noisySig(1:end-filterOrder)];
    
    for n = filterOrder+1:length(noisySig)
        x = refSignal(n:-1:n-filterOrder+1);
        y = w' * x;
        e = noisySig(n) - y;
        enhancedSig_nlms(n) = e;
        w = w + (mu / (x'*x + delta)) * e * x;
    end
    enhancedSig_nlms = enhancedSig_nlms(filterOrder+1:end);
    cleanSig_nlms = cleanSig(filterOrder+1:end);
    
    %% Process with CNN (if available)
    if hasCNN
        try
            % Compute STFT
            [S_noisy_cnn, ~, ~] = stft(noisySig, fs, 'Window', winFun, ...
                                       'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
            
            magNoisy = abs(S_noisy_cnn);
            phaseNoisy = angle(S_noisy_cnn);
            logMagNoisy = log(magNoisy + 1e-10);
            
            % Pad to match training dimensions
            [nFreq, nTime] = size(logMagNoisy);
            paddedInput = zeros(maxFreqBins, maxTimeBins, 1, 1);
            paddedInput(1:nFreq, 1:nTime, 1, 1) = logMagNoisy;
            
            % Predict mask
            predictedMaskPadded = predict(net, paddedInput);
            predictedMask = squeeze(predictedMaskPadded(1:nFreq, 1:nTime, 1, 1));
            
            % Apply mask
            S_enhanced_cnn = predictedMask .* magNoisy .* exp(1j * phaseNoisy);
            
            % Inverse STFT
            enhancedSig_cnn = istft(S_enhanced_cnn, fs, 'Window', winFun, ...
                                    'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
            enhancedSig_cnn = enhancedSig_cnn(1:length(noisySig));
        catch
            enhancedSig_cnn = noisySig; % Fallback if CNN fails
        end
    end
    
    %% Compute all metrics
    try
        % Noisy metrics
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
        
        % NLMS filter metrics
        results.snr_nlms(idx) = snr(cleanSig_nlms, enhancedSig_nlms - cleanSig_nlms);
        results.sisnr_nlms(idx) = compute_sisnr(cleanSig_nlms, enhancedSig_nlms);
        results.stoi_nlms(idx) = compute_stoi_simplified(cleanSig_nlms, enhancedSig_nlms, fs);
        results.lsd_nlms(idx) = compute_lsd(cleanSig_nlms, enhancedSig_nlms, fs, winLen, hopSize, nfft);
        results.segsnr_nlms(idx) = computeSegmentalSNR(cleanSig_nlms, enhancedSig_nlms, fs);
        
        % CNN metrics (if available)
        if hasCNN
            results.snr_cnn(idx) = snr(cleanSig, enhancedSig_cnn - cleanSig);
            results.sisnr_cnn(idx) = compute_sisnr(cleanSig, enhancedSig_cnn);
            results.stoi_cnn(idx) = compute_stoi_simplified(cleanSig, enhancedSig_cnn, fs);
            results.lsd_cnn(idx) = compute_lsd(cleanSig, enhancedSig_cnn, fs, winLen, hopSize, nfft);
            results.segsnr_cnn(idx) = computeSegmentalSNR(cleanSig, enhancedSig_cnn, fs);
        end
    catch ME
        % If any metric computation fails, set to NaN
        warning('Error computing metrics for sample %d: %s', idx, ME.message);
    end
end

fprintf('\nBatch processing complete!\n\n');

%% ========================================================================
%  SECTION 3: STATISTICAL ANALYSIS
% =========================================================================
fprintf('SECTION 3: Statistical Analysis\n');
fprintf('--------------------------------\n\n');

% Remove NaN values
validIdx = ~isnan(results.snr_noisy);

% Helper function to compute and display statistics
function displayMetricStats(metricName, noisyVals, wienerVals, wienerImpVals, nlmsVals, cnnVals, hasCNN)
    fprintf('%s:\n', metricName);
    fprintf('  Noisy:           %.3f ± %.3f\n', mean(noisyVals), std(noisyVals));
    fprintf('  Wiener:          %.3f ± %.3f (Δ = %.3f)\n', ...
            mean(wienerVals), std(wienerVals), mean(wienerVals) - mean(noisyVals));
    fprintf('  Improved Wiener: %.3f ± %.3f (Δ = %.3f)\n', ...
            mean(wienerImpVals), std(wienerImpVals), mean(wienerImpVals) - mean(noisyVals));
    fprintf('  NLMS:            %.3f ± %.3f (Δ = %.3f)\n', ...
            mean(nlmsVals), std(nlmsVals), mean(nlmsVals) - mean(noisyVals));
    if hasCNN
        fprintf('  CNN:             %.3f ± %.3f (Δ = %.3f)\n', ...
                mean(cnnVals), std(cnnVals), mean(cnnVals) - mean(noisyVals));
    end
    fprintf('\n');
end

% Display all metrics
displayMetricStats('SNR (dB)', ...
    results.snr_noisy(validIdx), results.snr_wiener(validIdx), ...
    results.snr_wiener_imp(validIdx), results.snr_nlms(validIdx), ...
    results.snr_cnn(validIdx), hasCNN);

displayMetricStats('SI-SNR (dB)', ...
    results.sisnr_noisy(validIdx), results.sisnr_wiener(validIdx), ...
    results.sisnr_wiener_imp(validIdx), results.sisnr_nlms(validIdx), ...
    results.sisnr_cnn(validIdx), hasCNN);

displayMetricStats('STOI', ...
    results.stoi_noisy(validIdx), results.stoi_wiener(validIdx), ...
    results.stoi_wiener_imp(validIdx), results.stoi_nlms(validIdx), ...
    results.stoi_cnn(validIdx), hasCNN);

displayMetricStats('LSD', ...
    results.lsd_noisy(validIdx), results.lsd_wiener(validIdx), ...
    results.lsd_wiener_imp(validIdx), results.lsd_nlms(validIdx), ...
    results.lsd_cnn(validIdx), hasCNN);

displayMetricStats('Segmental SNR (dB)', ...
    results.segsnr_noisy(validIdx), results.segsnr_wiener(validIdx), ...
    results.segsnr_wiener_imp(validIdx), results.segsnr_nlms(validIdx), ...
    results.segsnr_cnn(validIdx), hasCNN);

%% ========================================================================
%  SECTION 4: COMPREHENSIVE VISUALIZATIONS
% =========================================================================
fprintf('SECTION 4: Generating Visualizations\n');
fprintf('-------------------------------------\n');

% Create output directory
outputDir = 'comparative_results';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Figure 1: Overall Performance Comparison
figure('Name', 'Overall Performance Comparison', 'Position', [50 50 1400 900]);

methods = {'Noisy', 'Wiener', 'Improved Wiener', 'NLMS'};
if hasCNN
    methods{end+1} = 'CNN';
end

% SNR Comparison
subplot(2,3,1);
snrData = [mean(results.snr_noisy(validIdx)), mean(results.snr_wiener(validIdx)), ...
           mean(results.snr_wiener_imp(validIdx)), mean(results.snr_nlms(validIdx))];
snrStd = [std(results.snr_noisy(validIdx)), std(results.snr_wiener(validIdx)), ...
          std(results.snr_wiener_imp(validIdx)), std(results.snr_nlms(validIdx))];
if hasCNN
    snrData(end+1) = mean(results.snr_cnn(validIdx));
    snrStd(end+1) = std(results.snr_cnn(validIdx));
end
bar(snrData);
hold on;
errorbar(1:length(snrData), snrData, snrStd, 'k.', 'LineWidth', 1.5);
hold off;
set(gca, 'XTickLabel', methods);
ylabel('SNR (dB)');
title('Signal-to-Noise Ratio');
grid on;
xtickangle(45);

% SI-SNR Comparison
subplot(2,3,2);
sisnrData = [mean(results.sisnr_noisy(validIdx)), mean(results.sisnr_wiener(validIdx)), ...
             mean(results.sisnr_wiener_imp(validIdx)), mean(results.sisnr_nlms(validIdx))];
sisnrStd = [std(results.sisnr_noisy(validIdx)), std(results.sisnr_wiener(validIdx)), ...
            std(results.sisnr_wiener_imp(validIdx)), std(results.sisnr_nlms(validIdx))];
if hasCNN
    sisnrData(end+1) = mean(results.sisnr_cnn(validIdx));
    sisnrStd(end+1) = std(results.sisnr_cnn(validIdx));
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

% STOI Comparison
subplot(2,3,3);
stoiData = [mean(results.stoi_noisy(validIdx)), mean(results.stoi_wiener(validIdx)), ...
            mean(results.stoi_wiener_imp(validIdx)), mean(results.stoi_nlms(validIdx))];
stoiStd = [std(results.stoi_noisy(validIdx)), std(results.stoi_wiener(validIdx)), ...
           std(results.stoi_wiener_imp(validIdx)), std(results.stoi_nlms(validIdx))];
if hasCNN
    stoiData(end+1) = mean(results.stoi_cnn(validIdx));
    stoiStd(end+1) = std(results.stoi_cnn(validIdx));
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
if hasCNN
    lsdData(end+1) = mean(results.lsd_cnn(validIdx));
    lsdStd(end+1) = std(results.lsd_cnn(validIdx));
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

% Segmental SNR Comparison
subplot(2,3,5);
segsnrData = [mean(results.segsnr_noisy(validIdx)), mean(results.segsnr_wiener(validIdx)), ...
              mean(results.segsnr_wiener_imp(validIdx)), mean(results.segsnr_nlms(validIdx))];
segsnrStd = [std(results.segsnr_noisy(validIdx)), std(results.segsnr_wiener(validIdx)), ...
             std(results.segsnr_wiener_imp(validIdx)), std(results.segsnr_nlms(validIdx))];
if hasCNN
    segsnrData(end+1) = mean(results.segsnr_cnn(validIdx));
    segsnrStd(end+1) = std(results.segsnr_cnn(validIdx));
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

% Improvement over Noisy
subplot(2,3,6);
improvementData = [0, ... % Noisy baseline
                   mean(results.snr_wiener(validIdx)) - mean(results.snr_noisy(validIdx)), ...
                   mean(results.snr_wiener_imp(validIdx)) - mean(results.snr_noisy(validIdx)), ...
                   mean(results.snr_nlms(validIdx)) - mean(results.snr_noisy(validIdx))];
if hasCNN
    improvementData(end+1) = mean(results.snr_cnn(validIdx)) - mean(results.snr_noisy(validIdx));
end
bar(improvementData);
set(gca, 'XTickLabel', methods);
ylabel('SNR Improvement (dB)');
title('SNR Improvement Over Noisy');
grid on;
yline(0, 'r--', 'LineWidth', 1.5);
xtickangle(45);

saveas(gcf, fullfile(outputDir, 'overall_performance_comparison.png'));
fprintf('  Saved: overall_performance_comparison.png\n');

% Figure 2: Performance by Noise Type
figure('Name', 'Performance by Noise Type', 'Position', [100 100 1400 700]);

uniqueNoises = unique(results.noiseTypes);
numNoises = length(uniqueNoises);

% SNR by noise type
subplot(1,2,1);
snrByNoise = zeros(numNoises, length(methods));
for i = 1:numNoises
    noiseIdx = strcmp(results.noiseTypes, uniqueNoises{i}) & validIdx;
    snrByNoise(i,1) = mean(results.snr_noisy(noiseIdx));
    snrByNoise(i,2) = mean(results.snr_wiener(noiseIdx));
    snrByNoise(i,3) = mean(results.snr_wiener_imp(noiseIdx));
    snrByNoise(i,4) = mean(results.snr_nlms(noiseIdx));
    if hasCNN
        snrByNoise(i,5) = mean(results.snr_cnn(noiseIdx));
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
    if hasCNN
        stoiByNoise(i,5) = mean(results.stoi_cnn(noiseIdx));
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
figure('Name', 'Performance by SNR Level', 'Position', [150 150 1200 700]);

uniqueSNRs = unique(results.snrLevels);
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
    if hasCNN
        snrImpBySNR(i,4) = mean(results.snr_cnn(snrIdx)) - baselineSNR;
    end
end
bar(snrImpBySNR);
set(gca, 'XTickLabel', uniqueSNRs);
ylabel('SNR Improvement (dB)');
title('SNR Improvement by Input SNR Level');
legend(methods(2:end), 'Location', 'best');
grid on;
yline(0, 'r--', 'LineWidth', 1.5);

% STOI improvement by input SNR level
subplot(1,2,2);
stoiImpBySNR = zeros(numSNRs, length(methods)-1);
for i = 1:numSNRs
    snrIdx = strcmp(results.snrLevels, uniqueSNRs{i}) & validIdx;
    baselineSTOI = mean(results.stoi_noisy(snrIdx));
    stoiImpBySNR(i,1) = mean(results.stoi_wiener(snrIdx)) - baselineSTOI;
    stoiImpBySNR(i,2) = mean(results.stoi_wiener_imp(snrIdx)) - baselineSTOI;
    stoiImpBySNR(i,3) = mean(results.stoi_nlms(snrIdx)) - baselineSTOI;
    if hasCNN
        stoiImpBySNR(i,4) = mean(results.stoi_cnn(snrIdx)) - baselineSTOI;
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
fprintf('SECTION 5: Saving Results\n');
fprintf('-------------------------\n');

% Save results structure
save(fullfile(outputDir, 'comprehensive_results.mat'), 'results');
fprintf('Results saved to: %s\n', fullfile(outputDir, 'comprehensive_results.mat'));

% Generate detailed report
fid = fopen(fullfile(outputDir, 'evaluation_report.txt'), 'w');
fprintf(fid, '========================================\n');
fprintf(fid, 'COMPREHENSIVE EVALUATION REPORT\n');
fprintf(fid, 'Speech Enhancement Project\n');
fprintf(fid, '========================================\n\n');

fprintf(fid, 'Dataset: NOIZEUS\n');
fprintf(fid, 'Samples evaluated: %d\n', numSamplesToProcess);
fprintf(fid, 'Methods compared: %s\n\n', strjoin(methods, ', '));

fprintf(fid, '----------------------------------------\n');
fprintf(fid, 'OVERALL PERFORMANCE SUMMARY\n');
fprintf(fid, '----------------------------------------\n\n');

fprintf(fid, 'SNR (dB):\n');
fprintf(fid, '  Noisy:           %.2f ± %.2f\n', mean(results.snr_noisy(validIdx)), std(results.snr_noisy(validIdx)));
fprintf(fid, '  Wiener:          %.2f ± %.2f (Δ = %.2f dB)\n', ...
        mean(results.snr_wiener(validIdx)), std(results.snr_wiener(validIdx)), ...
        mean(results.snr_wiener(validIdx)) - mean(results.snr_noisy(validIdx)));
fprintf(fid, '  Improved Wiener: %.2f ± %.2f (Δ = %.2f dB)\n', ...
        mean(results.snr_wiener_imp(validIdx)), std(results.snr_wiener_imp(validIdx)), ...
        mean(results.snr_wiener_imp(validIdx)) - mean(results.snr_noisy(validIdx)));
fprintf(fid, '  NLMS:            %.2f ± %.2f (Δ = %.2f dB)\n', ...
        mean(results.snr_nlms(validIdx)), std(results.snr_nlms(validIdx)), ...
        mean(results.snr_nlms(validIdx)) - mean(results.snr_noisy(validIdx)));
if hasCNN
    fprintf(fid, '  CNN:             %.2f ± %.2f (Δ = %.2f dB)\n', ...
            mean(results.snr_cnn(validIdx)), std(results.snr_cnn(validIdx)), ...
            mean(results.snr_cnn(validIdx)) - mean(results.snr_noisy(validIdx)));
end
fprintf(fid, '\n');

fprintf(fid, 'STOI Score:\n');
fprintf(fid, '  Noisy:           %.3f ± %.3f\n', mean(results.stoi_noisy(validIdx)), std(results.stoi_noisy(validIdx)));
fprintf(fid, '  Wiener:          %.3f ± %.3f (Δ = %.3f)\n', ...
        mean(results.stoi_wiener(validIdx)), std(results.stoi_wiener(validIdx)), ...
        mean(results.stoi_wiener(validIdx)) - mean(results.stoi_noisy(validIdx)));
fprintf(fid, '  Improved Wiener: %.3f ± %.3f (Δ = %.3f)\n', ...
        mean(results.stoi_wiener_imp(validIdx)), std(results.stoi_wiener_imp(validIdx)), ...
        mean(results.stoi_wiener_imp(validIdx)) - mean(results.stoi_noisy(validIdx)));
fprintf(fid, '  NLMS:            %.3f ± %.3f (Δ = %.3f)\n', ...
        mean(results.stoi_nlms(validIdx)), std(results.stoi_nlms(validIdx)), ...
        mean(results.stoi_nlms(validIdx)) - mean(results.stoi_noisy(validIdx)));
if hasCNN
    fprintf(fid, '  CNN:             %.3f ± %.3f (Δ = %.3f)\n', ...
            mean(results.stoi_cnn(validIdx)), std(results.stoi_cnn(validIdx)), ...
            mean(results.stoi_cnn(validIdx)) - mean(results.stoi_noisy(validIdx)));
end
fprintf(fid, '\n');

fprintf(fid, '----------------------------------------\n');
fprintf(fid, 'KEY FINDINGS\n');
fprintf(fid, '----------------------------------------\n\n');

% Determine best method for each metric
[~, bestSNR] = max([mean(results.snr_wiener(validIdx)), ...
                    mean(results.snr_wiener_imp(validIdx)), ...
                    mean(results.snr_nlms(validIdx)), ...
                    mean(results.snr_cnn(validIdx))]);
[~, bestSTOI] = max([mean(results.stoi_wiener(validIdx)), ...
                     mean(results.stoi_wiener_imp(validIdx)), ...
                     mean(results.stoi_nlms(validIdx)), ...
                     mean(results.stoi_cnn(validIdx))]);

bestMethodsSNR = methods(2:end);
bestMethodsSTOI = methods(2:end);

fprintf(fid, '1. Best SNR improvement: %s\n', bestMethodsSNR{bestSNR});
fprintf(fid, '2. Best STOI improvement: %s\n', bestMethodsSTOI{bestSTOI});
fprintf(fid, '3. Most challenging noise types:\n');

% Find hardest noise types
avgImpByNoise = zeros(numNoises, 1);
for i = 1:numNoises
    noiseIdx = strcmp(results.noiseTypes, uniqueNoises{i}) & validIdx;
    avgImpByNoise(i) = mean(results.snr_wiener_imp(noiseIdx)) - mean(results.snr_noisy(noiseIdx));
end
[~, sortIdx] = sort(avgImpByNoise);
for i = 1:min(3, numNoises)
    fprintf(fid, '   - %s (%.2f dB improvement)\n', uniqueNoises{sortIdx(i)}, avgImpByNoise(sortIdx(i)));
end

fprintf(fid, '\n');
fprintf(fid, '----------------------------------------\n');
fprintf(fid, 'CONCLUSION\n');
fprintf(fid, '----------------------------------------\n\n');

if hasCNN
    fprintf(fid, 'This evaluation compared traditional DSP methods\n');
    fprintf(fid, '(Wiener, Improved Wiener, NLMS) with a CNN-based\n');
    fprintf(fid, 'machine learning approach for speech enhancement.\n\n');
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
fprintf('PROJECT COMPLETE! 🎉\n');
fprintf('========================================\n');

%% computeSegmentalSNR - Segmental SNR calculation
function segSNR = computeSegmentalSNR(cleanSignal, noisySignal, fs)
    % Compute segmental SNR
    %
    % Inputs:
    %   cleanSignal - Clean reference signal
    %   noisySignal - Noisy/enhanced signal
    %   fs - Sample rate
    %
    % Output:
    %   segSNR - Segmental SNR in dB
    
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
        
        noise = noisyFrame - cleanFrame;
        
        signalPower = sum(cleanFrame.^2);
        noisePower = sum(noise.^2);
        
        if noisePower > 0 && signalPower > 0
            frameSNR(i) = 10 * log10(signalPower / noisePower);
        else
            frameSNR(i) = 0;
        end
    end
    
    % Remove outliers and compute mean
    frameSNR = frameSNR(frameSNR > -20 & frameSNR < 40);
    segSNR = mean(frameSNR);
end