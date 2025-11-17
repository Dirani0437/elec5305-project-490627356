% =========================================================================
% STAGE 2: Adaptive Filtering for Speech Enhancement (UPDATED)
% =========================================================================
% This script implements traditional DSP approaches for speech enhancement:
% 1. Voice Activity Detection (VAD) - Identify speech vs. noise segments
% 2. Short-Time Fourier Transform (STFT) processing - Time-frequency analysis
% 3. Wiener filtering - Optimal linear filter for noise reduction
% 4. Normalized Least Mean Squares (NLMS) adaptive filtering - Online learning
%
% UPDATED: Added comprehensive visualization similar to Stage 3 ML script
%
% Theory:
% Speech enhancement aims to recover clean speech s(n) from noisy observation
% y(n) = s(n) + d(n), where d(n) is additive noise.
%
% Key DSP Concepts:
% - STFT: Analyzes signal in overlapping time windows, applying FFT to each
% - Wiener Filter: Minimizes mean square error between estimate and clean signal
% - NLMS: Adaptive filter that learns to predict and cancel noise
%
% Author: Alaa Aldirani
% Project: Real-Time Speech Enhancement
% =========================================================================

clear; close all; clc;

%% Configuration
fprintf('========================================\n');
fprintf('STAGE 2: Adaptive Filtering Implementation\n');
fprintf('========================================\n\n');

% Load prepared dataset from Stage 1
fprintf('Loading prepared dataset...\n');
load('prepared_data/noizeus_prepared.mat');
fprintf('Dataset loaded successfully!\n\n');

% Select a test sample for processing
% Using test set (not training) to evaluate on unseen data
testIdx = 1; % You can change this to test different samples
cleanSig = testData(testIdx).clean;      % Ground truth clean speech
noisySig = testData(testIdx).noisy;      % Observed noisy speech
fs = testData(testIdx).fs;                % Sample rate (8000 Hz)
noiseType = testData(testIdx).noiseType;  % Type of noise added
snrLevel = testData(testIdx).snr;         % SNR level string (e.g., '5dB')

fprintf('Processing sample:\n');
fprintf('  Noise type: %s\n', noiseType);
fprintf('  SNR level: %s\n', snrLevel);
fprintf('  Sample rate: %d Hz\n', fs);
fprintf('  Duration: %.2f seconds\n\n', length(noisySig)/fs);

%% STFT Parameters
% =========================================================================
% Short-Time Fourier Transform (STFT) parameters
% The STFT segments the signal into overlapping frames and applies FFT
% to each frame, revealing how frequency content changes over time.
%
% Key trade-offs:
% - Longer window → Better frequency resolution, worse time resolution
% - Shorter window → Better time resolution, worse frequency resolution
% - More overlap → Smoother output, more computation
% =========================================================================

% Window length: 32ms is standard for speech processing
% At 8kHz: 0.032 * 8000 = 256 samples
% This captures ~4 pitch periods for typical male speech (125Hz fundamental)
winLen = round(0.032 * fs); % 32ms window

% Hop size (frame shift): 16ms gives 50% overlap
% 50% overlap balances smoothness with computational efficiency
% Ensures smooth reconstruction after inverse STFT
hopSize = round(0.016 * fs); % 16ms hop (50% overlap)

% FFT size: Next power of 2 for computational efficiency
% Zero-padding to power of 2 enables fast FFT algorithms
nfft = 2^nextpow2(winLen);

% Window function: Hamming window reduces spectral leakage
% 'periodic' option is optimal for STFT overlap-add reconstruction
% Hamming has good main lobe width and side lobe suppression
winFun = hamming(winLen, 'periodic');

fprintf('STFT Parameters:\n');
fprintf('  Window length: %d samples (%.1f ms)\n', winLen, winLen/fs*1000);
fprintf('  Hop size: %d samples (%.1f ms)\n', hopSize, hopSize/fs*1000);
fprintf('  FFT size: %d\n', nfft);
fprintf('  Overlap: %.1f%%\n\n', (1-hopSize/winLen)*100);

%% ========================================================================
%  SECTION 1: VOICE ACTIVITY DETECTION (VAD)
% =========================================================================
% VAD identifies which frames contain speech vs. only noise
% This is critical because:
% 1. We need noise-only frames to estimate noise characteristics
% 2. Different processing might be applied to speech vs. non-speech regions
%
% Method: Energy-based VAD
% - Compute short-time energy of each frame
% - Speech frames have higher energy than noise-only frames
% - Simple but effective for many scenarios
% =========================================================================
fprintf('SECTION 1: Voice Activity Detection\n');
fprintf('----------------------------------\n');

% Compute STFT of noisy signal for VAD
% stft() returns complex spectrogram, frequency vector, and time vector
% S_noisy: Complex matrix [Frequency bins × Time frames]
% F: Frequency vector (Hz)
% T: Time vector (seconds)
[S_noisy, F, T] = stft(noisySig, fs, 'Window', winFun, ...
                       'OverlapLength', winLen-hopSize, 'FFTLength', nfft);

% Energy-based VAD
% Sum squared magnitudes across all frequency bins for each frame
% This gives total energy in each time frame
frameEnergy = sum(abs(S_noisy).^2, 1);

% Convert to decibel scale for better dynamic range
% eps prevents log(0) = -inf
frameEnergyDB = 10*log10(frameEnergy + eps);

% Simple threshold-based VAD
% Threshold is set 5 dB below mean energy
% Assumes noise-only frames bring down the average
% This is a heuristic that works well for moderate SNRs
energyThreshold = mean(frameEnergyDB) - 5; % 5 dB below mean
vadDecisions = frameEnergyDB > energyThreshold; % 1 = speech, 0 = noise

% Smooth VAD decisions with median filter
% Removes isolated detection errors (single frame mistakes)
% Window of 5 frames smooths rapid fluctuations
% > 0.5 converts back to logical array
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
% Wiener Filter Theory:
% The optimal linear filter that minimizes mean square error:
%   H(f) = S_xx(f) / [S_xx(f) + S_nn(f)]
%   where S_xx = clean signal PSD, S_nn = noise PSD
%
% Since we don't have S_xx, we estimate:
%   H(f) ≈ 1 - S_nn(f) / S_yy(f)
%   where S_yy = noisy signal PSD (observed)
%
% This assumes: S_yy = S_xx + S_nn (independence)
%
% Gain Interpretation:
% - Where noise dominates: S_nn ≈ S_yy → Gain ≈ 0 (attenuate)
% - Where signal dominates: S_nn << S_yy → Gain ≈ 1 (preserve)
% =========================================================================
fprintf('SECTION 2: Wiener Filtering\n');
fprintf('---------------------------\n');

% Estimate noise spectrum from noise-only frames (identified by VAD)
% Use frames where VAD decided there's no speech
noiseFrames = S_noisy(:, ~vadDecisions);
if ~isempty(noiseFrames)
    % Average noise PSD across noise-only frames
    % Assumes noise is stationary (statistics don't change over time)
    noisePSD = mean(abs(noiseFrames).^2, 2);
else
    % Fallback: use first few frames as noise estimate
    % Common assumption: recording starts with silence/noise
    noisePSD = mean(abs(S_noisy(:, 1:min(10, size(S_noisy,2)))).^2, 2);
end

% Compute Wiener gain for each time-frequency bin
% noisyPSD: instantaneous power at each T-F bin
noisyPSD = abs(S_noisy).^2;

% Wiener gain: H = max(1 - noise_PSD/noisy_PSD, 0)
% bsxfun broadcasts noisePSD (column vector) across all time frames
% max(..., 0) ensures non-negative gain (no amplification of noise)
wienerGain = max(1 - bsxfun(@rdivide, noisePSD, noisyPSD), 0);

% Apply Wiener filter by multiplying gain with noisy spectrum
% This attenuates frequency bins dominated by noise
% Preserves frequency bins with strong signal
S_wiener = wienerGain .* S_noisy;

% Reconstruct time-domain signal using inverse STFT
% istft performs overlap-add synthesis with proper normalization
enhancedSig_wiener = istft(S_wiener, fs, 'Window', winFun, ...
                           'OverlapLength', winLen-hopSize, ...
                           'FFTLength', nfft);

% Trim or pad to match original signal length
% STFT/ISTFT can change length slightly due to framing
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
% Standard Wiener filter can leave residual noise (musical noise)
% Improvements:
% 1. Oversubtraction (alpha > 1): More aggressive noise removal
% 2. Spectral floor (beta): Prevent complete zeroing of bins
%
% Modified gain: H = max(1 - alpha * S_nn/S_yy, beta)
%
% alpha > 1: Subtracts more than estimated noise power
%   - Pros: Better noise reduction
%   - Cons: May distort speech
%
% beta > 0: Minimum gain (spectral floor)
%   - Pros: Reduces musical noise artifacts
%   - Cons: Some residual noise remains
%
% These parameters represent the "aggressiveness vs. distortion" trade-off
% =========================================================================
fprintf('SECTION 3: Improved Wiener Filter\n');
fprintf('----------------------------------\n');

% Parameters for improved Wiener filter
alpha = 2.0; % Oversubtraction factor (typically 1-4)
             % Higher alpha = more noise reduction but more distortion
beta = 0.01; % Spectral floor (typically 0.01-0.1)
             % Minimum gain to prevent complete spectral holes

% Improved Wiener gain with oversubtraction
% alpha = 2 means we subtract 2× the estimated noise power
% beta = 0.01 means minimum gain is 1% (not complete silence)
improvedWienerGain = max(1 - alpha * bsxfun(@rdivide, noisePSD, noisyPSD), beta);

% Apply improved Wiener filter
S_wiener_improved = improvedWienerGain .* S_noisy;

% Reconstruct signal using inverse STFT
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
% Normalized Least Mean Squares (NLMS) Adaptive Filter
%
% Theory: Adaptive noise cancellation
% - Models noise as predictable from a reference signal
% - Iteratively adjusts filter weights to minimize error
% - Error signal = desired signal - filter output
%
% In speech enhancement context:
% - Reference: delayed version of noisy signal
% - Output: estimate of noise component
% - Error: estimate of clean speech (what we want!)
%
% NLMS Update Rule:
%   w(n+1) = w(n) + [mu / (x'x + delta)] * e(n) * x(n)
%
% mu: Step size (learning rate)
%   - Too small: slow convergence
%   - Too large: instability
% delta: Regularization (prevents division by zero)
% filterOrder: Length of adaptive filter (memory)
%
% Note: This assumes noise has some temporal correlation
% Works better for stationary, correlated noise
% =========================================================================
fprintf('SECTION 4: NLMS Adaptive Filtering\n');
fprintf('-----------------------------------\n');

% NLMS parameters
filterOrder = 32; % Filter length (number of taps)
                  % Determines how much past signal to use for prediction
mu = 0.1;        % Step size (learning rate), typically 0.01-1
                  % Controls how quickly filter adapts
delta = 0.01;    % Regularization parameter (small positive constant)
                  % Prevents division by zero when input energy is small

% Initialize adaptive filter
w = zeros(filterOrder, 1); % Filter weights (coefficients), start at zero
enhancedSig_nlms = zeros(size(noisySig)); % Pre-allocate output

% Create delayed reference signal
% Uses past noisy signal as reference for noise prediction
% Assumption: noise has temporal correlation that can be exploited
refSignal = [zeros(filterOrder, 1); noisySig(1:end-filterOrder)];

fprintf('NLMS Parameters:\n');
fprintf('  Filter order: %d\n', filterOrder);
fprintf('  Step size (mu): %.3f\n', mu);
fprintf('  Regularization: %.3f\n', delta);

% NLMS algorithm - sample-by-sample processing
% This is the core adaptive filtering loop
for n = filterOrder+1:length(noisySig)
    % Extract input vector (filterOrder most recent samples)
    % Flip order so x(n) is first, x(n-1) is second, etc.
    x = refSignal(n:-1:n-filterOrder+1);
    
    % Filter output: inner product of weights and input
    % This is the linear prediction of noise
    y = w' * x;
    
    % Error signal: difference between noisy observation and predicted noise
    % e(n) = y_noisy(n) - y_predicted_noise(n)
    % If prediction is good, error ≈ clean speech
    e = noisySig(n) - y;
    enhancedSig_nlms(n) = e; % Store as enhanced output
    
    % Update weights using NLMS rule
    % Normalized by input power for stable convergence
    % delta prevents division by zero
    w = w + (mu / (x'*x + delta)) * e * x;
end

% Remove initial transient where filter hasn't converged yet
% First filterOrder samples have incomplete filter history
enhancedSig_nlms = enhancedSig_nlms(filterOrder+1:end);
cleanSig_trimmed = cleanSig(filterOrder+1:end);    % Trim clean for fair comparison
noisySig_trimmed = noisySig(filterOrder+1:end);    % Trim noisy for fair comparison

fprintf('NLMS filtering completed!\n');
fprintf('  Processed %d samples\n\n', length(enhancedSig_nlms));

%% ========================================================================
%  SECTION 5: PERFORMANCE EVALUATION
% =========================================================================
% Compute Signal-to-Noise Ratio (SNR) improvement
% SNR = 10 * log10(signal_power / noise_power)
%
% SNR improvement indicates how much noise was reduced
% Higher SNR = cleaner signal
%
% Note: This is a simple objective measure
% Doesn't capture speech distortion or perceptual quality
% Full evaluation uses multiple metrics (STOI, PESQ, etc.)
% =========================================================================
fprintf('SECTION 5: Performance Evaluation\n');
fprintf('----------------------------------\n');

% Compute SNR improvement for each method
% snr(reference, noise) computes 10*log10(sum(ref^2)/sum(noise^2))
% Noise component = enhanced - clean (residual error)
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
% Comprehensive visualization to understand algorithm performance
% Multiple views reveal different aspects:
% - Time domain: Waveform structure and amplitude
% - Spectrogram: Time-frequency patterns
% - PSD: Average frequency content
% - Gain/Mask: What the algorithm is doing
% =========================================================================
fprintf('SECTION 6: Generating Visualizations\n');
fprintf('-------------------------------------\n');

% Create output directory early for saving figures
outputDir = 'adaptive_filtering_results';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Figure 1: Time-domain signals comparison
% Shows how waveforms change after enhancement
figure('Name', 'Time Domain Comparison', 'Position', [50 50 1400 900]);

t = (0:length(noisySig)-1) / fs;                    % Time vector for full signals
t_nlms = (0:length(enhancedSig_nlms)-1) / fs;     % Time vector for NLMS (shorter)

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

saveas(gcf, fullfile(outputDir, 'time_domain_comparison.png'));

% Figure 2: Spectrogram comparison
% Shows time-frequency representation - very informative for speech
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

saveas(gcf, fullfile(outputDir, 'spectrogram_comparison.png'));

% Figure 3: Power Spectral Density comparison
% Shows average frequency content - helps identify noise characteristics
figure('Name', 'PSD Comparison', 'Position', [150 150 1200 600]);

% Compute PSD using Welch's method for each signal
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
xlim([0 fs/2]); % Nyquist frequency

saveas(gcf, fullfile(outputDir, 'psd_comparison.png'));

% =========================================================================
% NEW FIGURE 4: Comprehensive Enhancement Process Visualization
% (Similar to Stage 3 Machine Learning visualization)
% Shows the complete pipeline: Input → Processing → Output
% =========================================================================
fprintf('  Generating comprehensive enhancement visualization...\n');

figure('Name', 'Wiener Speech Enhancement Process', 'Position', [50 50 1400 800]);

% Row 1: Time domain signals (Clean, Noisy, Enhanced)
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
plot((0:length(enhancedSig_wiener_improved)-1)/fs, enhancedSig_wiener_improved);
title('Improved Wiener Enhanced');
xlabel('Time (s)'); ylabel('Amplitude');
grid on; ylim([-1 1]);

% Row 2: Spectrograms (Clean, Noisy, Enhanced)
subplot(3,3,4);
spectrogram(cleanSig, winFun, winLen-hopSize, nfft, fs, 'yaxis');
title('Clean Spectrogram');
colorbar;

subplot(3,3,5);
spectrogram(noisySig, winFun, winLen-hopSize, nfft, fs, 'yaxis');
title('Noisy Spectrogram');
colorbar;

subplot(3,3,6);
spectrogram(enhancedSig_wiener_improved, winFun, winLen-hopSize, nfft, fs, 'yaxis');
title('Enhanced Spectrogram');
colorbar;

% Row 3: Processing details (Noisy Magnitude, Gain/Mask, Enhanced Magnitude)
% This shows the internal workings of the algorithm
subplot(3,3,7);
imagesc(T, F/1000, 10*log10(abs(S_noisy).^2 + eps));
axis xy; colorbar;
title('Noisy Magnitude (dB)');
xlabel('Time (s)'); ylabel('Frequency (kHz)');

subplot(3,3,8);
imagesc(T, F/1000, improvedWienerGain);
axis xy; colorbar;
title('Improved Wiener Gain (Mask)');
xlabel('Time (s)'); ylabel('Frequency (kHz)');
caxis([0 1]); % Gain ranges from 0 to 1

subplot(3,3,9);
imagesc(T, F/1000, 10*log10(abs(S_wiener_improved).^2 + eps));
axis xy; colorbar;
title('Enhanced Magnitude (dB)');
xlabel('Time (s)'); ylabel('Frequency (kHz)');

sgtitle('Adaptive Filtering Speech Enhancement Process', 'FontSize', 14, 'FontWeight', 'bold');

saveas(gcf, fullfile(outputDir, 'wiener_enhancement_visualization.png'));
fprintf('  Saved: wiener_enhancement_visualization.png\n');

% =========================================================================
% NEW FIGURE 5: Standard Wiener Filter Enhancement Process
% Same layout for comparison with improved version
% =========================================================================
figure('Name', 'Standard Wiener Speech Enhancement', 'Position', [100 100 1400 800]);

% Row 1: Time domain signals
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
plot((0:length(enhancedSig_wiener)-1)/fs, enhancedSig_wiener);
title('Standard Wiener Enhanced');
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
spectrogram(enhancedSig_wiener, winFun, winLen-hopSize, nfft, fs, 'yaxis');
title('Enhanced Spectrogram');
colorbar;

% Row 3: Processing details
subplot(3,3,7);
imagesc(T, F/1000, 10*log10(abs(S_noisy).^2 + eps));
axis xy; colorbar;
title('Noisy Magnitude (dB)');
xlabel('Time (s)'); ylabel('Frequency (kHz)');

subplot(3,3,8);
imagesc(T, F/1000, wienerGain);
axis xy; colorbar;
title('Standard Wiener Gain (Mask)');
xlabel('Time (s)'); ylabel('Frequency (kHz)');
caxis([0 1]);

subplot(3,3,9);
imagesc(T, F/1000, 10*log10(abs(S_wiener).^2 + eps));
axis xy; colorbar;
title('Enhanced Magnitude (dB)');
xlabel('Time (s)'); ylabel('Frequency (kHz)');

sgtitle('Standard Wiener Filter Speech Enhancement Process', 'FontSize', 14, 'FontWeight', 'bold');

saveas(gcf, fullfile(outputDir, 'standard_wiener_enhancement_visualization.png'));
fprintf('  Saved: standard_wiener_enhancement_visualization.png\n');

% =========================================================================
% NEW FIGURE 6: Gain/Mask Comparison
% Direct comparison of the two Wiener filter gain functions
% Shows how oversubtraction affects the mask
% =========================================================================
figure('Name', 'Wiener Gain Comparison', 'Position', [150 150 1200 500]);

subplot(1,2,1);
imagesc(T, F/1000, wienerGain);
axis xy; colorbar;
title('Standard Wiener Gain');
xlabel('Time (s)'); ylabel('Frequency (kHz)');
caxis([0 1]);

subplot(1,2,2);
imagesc(T, F/1000, improvedWienerGain);
axis xy; colorbar;
title(sprintf('Improved Wiener Gain (α=%.1f, β=%.2f)', alpha, beta));
xlabel('Time (s)'); ylabel('Frequency (kHz)');
caxis([0 1]);

sgtitle('Comparison of Wiener Filter Gain Functions', 'FontSize', 14, 'FontWeight', 'bold');

saveas(gcf, fullfile(outputDir, 'wiener_gain_comparison.png'));
fprintf('  Saved: wiener_gain_comparison.png\n');

fprintf('Visualizations complete!\n\n');

%% ========================================================================
%  SECTION 7: SAVE RESULTS
% =========================================================================
% Save all processed signals and results for:
% 1. Listening tests (subjective evaluation)
% 2. Further analysis
% 3. Comparison with other methods (e.g., CNN in Stage 3)
% =========================================================================
fprintf('SECTION 7: Saving Results\n');
fprintf('-------------------------\n');

% Save enhanced audio files
% These can be played back to subjectively evaluate quality
audiowrite(fullfile(outputDir, 'clean.wav'), cleanSig, fs);
audiowrite(fullfile(outputDir, 'noisy.wav'), noisySig, fs);
audiowrite(fullfile(outputDir, 'enhanced_wiener.wav'), enhancedSig_wiener, fs);
audiowrite(fullfile(outputDir, 'enhanced_wiener_improved.wav'), enhancedSig_wiener_improved, fs);
audiowrite(fullfile(outputDir, 'enhanced_nlms.wav'), enhancedSig_nlms, fs);

% Save results structure with all metrics and parameters
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
fprintf('  - Results structure (.mat)\n');
fprintf('  - All visualization figures (.png)\n\n');

%% ========================================================================
%  SECTION 8: SUMMARY
% =========================================================================
fprintf('========================================\n');
fprintf('Processing Complete!\n');
fprintf('========================================\n');
fprintf('\nSummary:\n');
fprintf('  Test sample: %s @ %s SNR\n', noiseType, snrLevel);
fprintf('  Best method: ');

% Determine which method gave best SNR improvement
[~, bestIdx] = max([snr_wiener, snr_wiener_improved, snr_nlms]);
methods = {'Wiener Filter', 'Improved Wiener Filter', 'NLMS Filter'};
fprintf('%s\n', methods{bestIdx});

fprintf('\nVisualization files generated:\n');
fprintf('  1. time_domain_comparison.png\n');
fprintf('  2. spectrogram_comparison.png\n');
fprintf('  3. psd_comparison.png\n');
fprintf('  4. wiener_enhancement_visualization.png (NEW - similar to Stage 3)\n');
fprintf('  5. standard_wiener_enhancement_visualization.png (NEW)\n');
fprintf('  6. wiener_gain_comparison.png (NEW)\n');
fprintf('\nNext steps:\n');
fprintf('  1. Process multiple samples to get average performance\n');
fprintf('  2. Optimize filter parameters\n');
fprintf('  3. Proceed to Stage 3: Machine Learning Implementation\n');
fprintf('========================================\n');
