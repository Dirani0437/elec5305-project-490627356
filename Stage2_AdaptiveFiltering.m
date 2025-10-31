% =========================================================================
% STAGE 2: Adaptive Filtering for Speech Enhancement
% =========================================================================
% This script implements traditional DSP approaches for speech enhancement:
% 1. Voice Activity Detection (VAD)
% 2. Short-Time Fourier Transform (STFT) processing
% 3. Wiener filtering
% 4. Normalized Least Mean Squares (NLMS) adaptive filtering
%
% Author: Alaa Aldirani
% Project: Real-Time Speech Enhancement
% =========================================================================

clear; close all; clc;

%% Configuration
fprintf('========================================\n');
fprintf('STAGE 2: Adaptive Filtering Implementation\n');
fprintf('========================================\n\n');

% Load prepared dataset
fprintf('Loading prepared dataset...\n');
load('prepared_data/noizeus_prepared.mat');
fprintf('Dataset loaded successfully!\n\n');

% Select a test sample for processing
testIdx = 1; % You can change this to test different samples
cleanSig = testData(testIdx).clean;
noisySig = testData(testIdx).noisy;
fs = testData(testIdx).fs;
noiseType = testData(testIdx).noiseType;
snrLevel = testData(testIdx).snr;

fprintf('Processing sample:\n');
fprintf('  Noise type: %s\n', noiseType);
fprintf('  SNR level: %s\n', snrLevel);
fprintf('  Sample rate: %d Hz\n', fs);
fprintf('  Duration: %.2f seconds\n\n', length(noisySig)/fs);

%% STFT Parameters
winLen = round(0.032 * fs); % 32ms window
hopSize = round(0.016 * fs); % 16ms hop (50% overlap)
nfft = 2^nextpow2(winLen);
winFun = hamming(winLen, 'periodic');

fprintf('STFT Parameters:\n');
fprintf('  Window length: %d samples (%.1f ms)\n', winLen, winLen/fs*1000);
fprintf('  Hop size: %d samples (%.1f ms)\n', hopSize, hopSize/fs*1000);
fprintf('  FFT size: %d\n', nfft);
fprintf('  Overlap: %.1f%%\n\n', (1-hopSize/winLen)*100);

%% ========================================================================
%  SECTION 1: VOICE ACTIVITY DETECTION (VAD)
% =========================================================================
fprintf('SECTION 1: Voice Activity Detection\n');
fprintf('----------------------------------\n');

% Compute STFT of noisy signal for VAD
[S_noisy, F, T] = stft(noisySig, fs, 'Window', winFun, ...
                       'OverlapLength', winLen-hopSize, 'FFTLength', nfft);

% Energy-based VAD
frameEnergy = sum(abs(S_noisy).^2, 1);
frameEnergyDB = 10*log10(frameEnergy + eps);

% Simple threshold-based VAD
energyThreshold = mean(frameEnergyDB) - 5; % 5 dB below mean
vadDecisions = frameEnergyDB > energyThreshold;

% Smooth VAD decisions with median filter
vadDecisions = medfilt1(double(vadDecisions), 5) > 0.5;

fprintf('VAD Statistics:\n');
fprintf('  Total frames: %d\n', length(vadDecisions));
fprintf('  Speech frames: %d (%.1f%%)\n', sum(vadDecisions), ...
        sum(vadDecisions)/length(vadDecisions)*100);
fprintf('  Noise frames: %d (%.1f%%)\n\n', sum(~vadDecisions), ...
        sum(~vadDecisions)/length(vadDecisions)*100);

%% ========================================================================
%  SECTION 2: WIENER FILTERING
% =========================================================================
fprintf('SECTION 2: Wiener Filtering\n');
fprintf('---------------------------\n');

% Estimate noise spectrum from noise-only frames
noiseFrames = S_noisy(:, ~vadDecisions);
if ~isempty(noiseFrames)
    noisePSD = mean(abs(noiseFrames).^2, 2);
else
    % Fallback: use first few frames as noise estimate
    noisePSD = mean(abs(S_noisy(:, 1:min(10, size(S_noisy,2)))).^2, 2);
end

% Compute Wiener gain
noisyPSD = abs(S_noisy).^2;
wienerGain = max(1 - bsxfun(@rdivide, noisePSD, noisyPSD), 0);

% Apply Wiener filter
S_wiener = wienerGain .* S_noisy;

% Reconstruct signal using inverse STFT
enhancedSig_wiener = istft(S_wiener, fs, 'Window', winFun, ...
                           'OverlapLength', winLen-hopSize, ...
                           'FFTLength', nfft);

% Trim or pad to original length
if length(enhancedSig_wiener) >= length(noisySig)
    enhancedSig_wiener = enhancedSig_wiener(1:length(noisySig));
else
    % Pad with zeros if shorter
    enhancedSig_wiener = [enhancedSig_wiener; zeros(length(noisySig) - length(enhancedSig_wiener), 1)];
end

fprintf('Wiener filtering completed!\n');
fprintf('  Noise frames used for estimation: %d\n\n', size(noiseFrames, 2));

%% ========================================================================
%  SECTION 3: IMPROVED WIENER FILTER WITH OVERSUBTRACTION
% =========================================================================
fprintf('SECTION 3: Improved Wiener Filter\n');
fprintf('----------------------------------\n');

% Parameters for improved Wiener filter
alpha = 2.0; % Oversubtraction factor
beta = 0.01; % Spectral floor

% Improved Wiener gain with oversubtraction
improvedWienerGain = max(1 - alpha * bsxfun(@rdivide, noisePSD, noisyPSD), beta);

% Apply improved Wiener filter
S_wiener_improved = improvedWienerGain .* S_noisy;

% Reconstruct signal
enhancedSig_wiener_improved = istft(S_wiener_improved, fs, 'Window', winFun, ...
                                    'OverlapLength', winLen-hopSize, ...
                                    'FFTLength', nfft);

% Trim or pad to original length
if length(enhancedSig_wiener_improved) >= length(noisySig)
    enhancedSig_wiener_improved = enhancedSig_wiener_improved(1:length(noisySig));
else
    % Pad with zeros if shorter
    enhancedSig_wiener_improved = [enhancedSig_wiener_improved; zeros(length(noisySig) - length(enhancedSig_wiener_improved), 1)];
end

fprintf('Improved Wiener filtering completed!\n');
fprintf('  Oversubtraction factor: %.2f\n', alpha);
fprintf('  Spectral floor: %.3f\n\n', beta);

%% ========================================================================
%  SECTION 4: NLMS ADAPTIVE FILTERING
% =========================================================================
fprintf('SECTION 4: NLMS Adaptive Filtering\n');
fprintf('-----------------------------------\n');

% NLMS parameters
filterOrder = 32; % Filter length
mu = 0.1; % Step size (learning rate)
delta = 0.01; % Regularization parameter

% Initialize adaptive filter
w = zeros(filterOrder, 1); % Filter weights
enhancedSig_nlms = zeros(size(noisySig));

% Create delayed reference (assuming noise is somewhat stationary)
% In practice, you might have a reference noise signal
refSignal = [zeros(filterOrder, 1); noisySig(1:end-filterOrder)];

fprintf('NLMS Parameters:\n');
fprintf('  Filter order: %d\n', filterOrder);
fprintf('  Step size (mu): %.3f\n', mu);
fprintf('  Regularization: %.3f\n', delta);

% NLMS algorithm
for n = filterOrder+1:length(noisySig)
    % Extract input vector
    x = refSignal(n:-1:n-filterOrder+1);
    
    % Filter output (noise estimate)
    y = w' * x;
    
    % Error signal (enhanced speech)
    e = noisySig(n) - y;
    enhancedSig_nlms(n) = e;
    
    % Update weights using NLMS
    w = w + (mu / (x'*x + delta)) * e * x;
end

% Remove initial transient
enhancedSig_nlms = enhancedSig_nlms(filterOrder+1:end);
cleanSig_trimmed = cleanSig(filterOrder+1:end);
noisySig_trimmed = noisySig(filterOrder+1:end);

fprintf('NLMS filtering completed!\n');
fprintf('  Processed %d samples\n\n', length(enhancedSig_nlms));

%% ========================================================================
%  SECTION 5: PERFORMANCE EVALUATION
% =========================================================================
fprintf('SECTION 5: Performance Evaluation\n');
fprintf('----------------------------------\n');

% Compute SNR improvement
snr_noisy = snr(cleanSig, noisySig - cleanSig);
snr_wiener = snr(cleanSig, enhancedSig_wiener - cleanSig);
snr_wiener_improved = snr(cleanSig, enhancedSig_wiener_improved - cleanSig);
snr_nlms = snr(cleanSig_trimmed, enhancedSig_nlms - cleanSig_trimmed);

fprintf('SNR Results:\n');
fprintf('  Noisy signal: %.2f dB\n', snr_noisy);
fprintf('  Wiener filter: %.2f dB (%.2f dB improvement)\n', ...
        snr_wiener, snr_wiener - snr_noisy);
fprintf('  Improved Wiener: %.2f dB (%.2f dB improvement)\n', ...
        snr_wiener_improved, snr_wiener_improved - snr_noisy);
fprintf('  NLMS filter: %.2f dB (%.2f dB improvement)\n\n', ...
        snr_nlms, snr_nlms - snr_noisy);

%% ========================================================================
%  SECTION 6: VISUALIZATION
% =========================================================================
fprintf('SECTION 6: Generating Visualizations\n');
fprintf('-------------------------------------\n');

% Figure 1: Time-domain signals
figure('Name', 'Time Domain Comparison', 'Position', [50 50 1400 900]);

t = (0:length(noisySig)-1) / fs;
t_nlms = (0:length(enhancedSig_nlms)-1) / fs;

subplot(3,2,1);
plot(t, cleanSig);
title('Clean Signal');
xlabel('Time (s)'); ylabel('Amplitude');
grid on; ylim([-1 1]);

subplot(3,2,2);
plot(t, noisySig);
title(sprintf('Noisy Signal (%s, %s)', noiseType, snrLevel));
xlabel('Time (s)'); ylabel('Amplitude');
grid on; ylim([-1 1]);

subplot(3,2,3);
plot(t, enhancedSig_wiener);
title('Enhanced - Wiener Filter');
xlabel('Time (s)'); ylabel('Amplitude');
grid on; ylim([-1 1]);

subplot(3,2,4);
plot(t, enhancedSig_wiener_improved);
title('Enhanced - Improved Wiener');
xlabel('Time (s)'); ylabel('Amplitude');
grid on; ylim([-1 1]);

subplot(3,2,5);
plot(t_nlms, enhancedSig_nlms);
title('Enhanced - NLMS Filter');
xlabel('Time (s)'); ylabel('Amplitude');
grid on; ylim([-1 1]);

subplot(3,2,6);
t_vad = linspace(0, length(noisySig)/fs, length(vadDecisions));
plot(t_vad, vadDecisions);
title('VAD Decisions');
xlabel('Time (s)'); ylabel('Speech/Noise');
grid on; ylim([-0.1 1.1]);
yticks([0 1]); yticklabels({'Noise', 'Speech'});

% Figure 2: Spectrograms
figure('Name', 'Spectrogram Comparison', 'Position', [100 100 1400 900]);

subplot(3,2,1);
spectrogram(cleanSig, winFun, winLen-hopSize, nfft, fs, 'yaxis');
title('Clean Signal'); colorbar;

subplot(3,2,2);
spectrogram(noisySig, winFun, winLen-hopSize, nfft, fs, 'yaxis');
title('Noisy Signal'); colorbar;

subplot(3,2,3);
spectrogram(enhancedSig_wiener, winFun, winLen-hopSize, nfft, fs, 'yaxis');
title('Wiener Filter'); colorbar;

subplot(3,2,4);
spectrogram(enhancedSig_wiener_improved, winFun, winLen-hopSize, nfft, fs, 'yaxis');
title('Improved Wiener'); colorbar;

subplot(3,2,5);
spectrogram(enhancedSig_nlms, winFun, winLen-hopSize, nfft, fs, 'yaxis');
title('NLMS Filter'); colorbar;

subplot(3,2,6);
imagesc(T, F/1000, 10*log10(abs(S_noisy).^2 + eps));
axis xy; colorbar;
title('Noisy Signal STFT Magnitude (dB)');
xlabel('Time (s)'); ylabel('Frequency (kHz)');

% Figure 3: Power Spectral Density
figure('Name', 'PSD Comparison', 'Position', [150 150 1200 600]);

[pxx_clean, f] = pwelch(cleanSig, hamming(512), 256, 1024, fs);
[pxx_noisy, ~] = pwelch(noisySig, hamming(512), 256, 1024, fs);
[pxx_wiener, ~] = pwelch(enhancedSig_wiener, hamming(512), 256, 1024, fs);
[pxx_wiener_imp, ~] = pwelch(enhancedSig_wiener_improved, hamming(512), 256, 1024, fs);
[pxx_nlms, ~] = pwelch(enhancedSig_nlms, hamming(512), 256, 1024, fs);

plot(f, 10*log10(pxx_clean), 'k', 'LineWidth', 1.5, 'DisplayName', 'Clean');
hold on;
plot(f, 10*log10(pxx_noisy), 'r', 'LineWidth', 1, 'DisplayName', 'Noisy');
plot(f, 10*log10(pxx_wiener), 'b', 'LineWidth', 1, 'DisplayName', 'Wiener');
plot(f, 10*log10(pxx_wiener_imp), 'g', 'LineWidth', 1, 'DisplayName', 'Improved Wiener');
plot(f, 10*log10(pxx_nlms), 'm', 'LineWidth', 1, 'DisplayName', 'NLMS');
hold off;
grid on;
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
title('Power Spectral Density Comparison');
legend('Location', 'northeast');
xlim([0 fs/2]);

fprintf('Visualizations complete!\n\n');

%% ========================================================================
%  SECTION 7: SAVE RESULTS
% =========================================================================
fprintf('SECTION 7: Saving Results\n');
fprintf('-------------------------\n');

% Create output directory
outputDir = 'adaptive_filtering_results';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Save enhanced audio files
audiowrite(fullfile(outputDir, 'clean.wav'), cleanSig, fs);
audiowrite(fullfile(outputDir, 'noisy.wav'), noisySig, fs);
audiowrite(fullfile(outputDir, 'enhanced_wiener.wav'), enhancedSig_wiener, fs);
audiowrite(fullfile(outputDir, 'enhanced_wiener_improved.wav'), enhancedSig_wiener_improved, fs);
audiowrite(fullfile(outputDir, 'enhanced_nlms.wav'), enhancedSig_nlms, fs);

% Save results
results = struct();
results.snr_noisy = snr_noisy;
results.snr_wiener = snr_wiener;
results.snr_wiener_improved = snr_wiener_improved;
results.snr_nlms = snr_nlms;
results.noiseType = noiseType;
results.snrLevel = snrLevel;
results.vadDecisions = vadDecisions;

save(fullfile(outputDir, 'results.mat'), 'results');

fprintf('Results saved to: %s\n', outputDir);
fprintf('  - Audio files (.wav)\n');
fprintf('  - Results structure (.mat)\n\n');

%% ========================================================================
%  SECTION 8: SUMMARY
% =========================================================================
fprintf('========================================\n');
fprintf('Processing Complete!\n');
fprintf('========================================\n');
fprintf('\nSummary:\n');
fprintf('  Test sample: %s @ %s SNR\n', noiseType, snrLevel);
fprintf('  Best method: ');
[~, bestIdx] = max([snr_wiener, snr_wiener_improved, snr_nlms]);
methods = {'Wiener Filter', 'Improved Wiener Filter', 'NLMS Filter'};
fprintf('%s\n', methods{bestIdx});
fprintf('\nNext steps:\n');
fprintf('  1. Process multiple samples to get average performance\n');
fprintf('  2. Optimize filter parameters\n');
fprintf('  3. Proceed to Stage 3: Machine Learning Implementation\n');
fprintf('========================================\n');