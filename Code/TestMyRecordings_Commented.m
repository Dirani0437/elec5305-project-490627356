% =========================================================================
% Test Speech Enhancement Algorithms on Personal Recordings
% =========================================================================
% This script loads your personal recordings and tests all speech
% enhancement algorithms, then plots comprehensive results.
%
% Purpose:
% Test the trained models and algorithms on real-world recordings
% that you've captured yourself. This provides a practical evaluation
% beyond the NOIZEUS benchmark dataset.
%
% Required files:
%   - myrecording_restaurant.wav (your noisy recording in restaurant)
%   - myrecording_train.wav (your noisy recording in train)
%   - Trained CNN model (optional): ml_results/trained_cnn.mat
%
% What this script does:
% 1. Loads your personal noisy recordings
% 2. Applies all enhancement methods (Wiener, NLMS, CNN, etc.)
% 3. Saves enhanced audio files for listening comparison
% 4. Generates comprehensive visualizations
% 5. Analyzes performance across different frequency bands
%
% Note: Since we don't have the clean reference for your recordings,
% we can't compute objective metrics (SNR, STOI, etc.). Instead,
% we analyze noise reduction characteristics and rely on subjective
% listening tests for quality assessment.
%
% Author: Alaa Aldirani
% =========================================================================

clear; close all; clc;

%% Configuration
fprintf('========================================\n');
fprintf('Testing Speech Enhancement on Your Recordings\n');
fprintf('========================================\n\n');

% Define your recording files
% These should be recordings you made in noisy environments
% Can be mono or stereo, any sample rate (will be processed accordingly)
recordingFiles = {
    'myrecording_restaurant.wav',
    'myrecording_train.wav'
};

% Check if files exist before proceeding
fprintf('Checking for recording files...\n');
for i = 1:length(recordingFiles)
    if exist(recordingFiles{i}, 'file')
        fprintf('  Found: %s\n', recordingFiles{i});
    else
        error('File not found: %s\nPlease place your recordings in the current directory.', recordingFiles{i});
    end
end
fprintf('\n');

%% Load CNN Model (if available)
% =========================================================================
% Attempt to load the trained CNN model from Stage 3
% If not available, we'll skip CNN enhancement and use only traditional methods
% =========================================================================
hasCNN = false;
try
    if exist('ml_results/trained_cnn.mat', 'file')
        cnnData = load('ml_results/trained_cnn.mat');
        % Check that all required variables are present
        if isfield(cnnData, 'net') && isfield(cnnData, 'maxFreqBins') && isfield(cnnData, 'maxTimeBins')
            net = cnnData.net;                    % Trained network
            maxFreqBins = cnnData.maxFreqBins;   % Input dimensions
            maxTimeBins = cnnData.maxTimeBins;
            hasCNN = true;
            fprintf('CNN model loaded successfully!\n\n');
        end
    else
        fprintf('CNN model not found. Proceeding with adaptive filtering only.\n\n');
    end
catch ME
    fprintf('Could not load CNN model: %s\n\n', ME.message);
end

%% Initialize Results Storage
% =========================================================================
% Pre-allocate storage for all enhanced signals
% Using cell arrays because recordings have different lengths
% =========================================================================
numRecordings = length(recordingFiles);
results = struct();
results.filename = recordingFiles;
results.original = cell(numRecordings, 1);          % Original noisy recordings
results.wiener = cell(numRecordings, 1);            % Wiener filtered
results.wiener_improved = cell(numRecordings, 1);   % Improved Wiener filtered
results.nlms = cell(numRecordings, 1);              % NLMS filtered
results.spectral_sub = cell(numRecordings, 1);      % Spectral subtraction
if hasCNN
    results.cnn = cell(numRecordings, 1);           % CNN enhanced
end
results.fs = zeros(numRecordings, 1);               % Sample rates

%% Process Each Recording
% =========================================================================
% Main processing loop: Apply all enhancement methods to each recording
% Each method has its own strengths and weaknesses
% =========================================================================
for recIdx = 1:numRecordings
    fprintf('Processing: %s\n', recordingFiles{recIdx});
    fprintf('----------------------------------\n');
    
    % Load audio file
    [noisySig, fs] = audioread(recordingFiles{recIdx});
    
    % Convert to mono if stereo
    % Most enhancement algorithms work on single-channel audio
    if size(noisySig, 2) > 1
        noisySig = mean(noisySig, 2); % Average left and right channels
        fprintf('  Converted stereo to mono\n');
    end
    
    % Normalize amplitude to prevent clipping
    % Scale so maximum absolute value is 1
    noisySig = noisySig / max(abs(noisySig));
    
    % Store original and metadata
    results.original{recIdx} = noisySig;
    results.fs(recIdx) = fs;
    
    fprintf('  Sample rate: %d Hz\n', fs);
    fprintf('  Duration: %.2f seconds\n', length(noisySig)/fs);
    
    %% STFT Parameters
    % =========================================================================
    % Standard speech processing parameters
    % These work well for most speech signals regardless of sample rate
    % =========================================================================
    winLen = round(0.032 * fs); % 32ms window
    hopSize = round(0.016 * fs); % 16ms hop (50% overlap)
    nfft = 2^nextpow2(winLen);   % FFT size (power of 2 for efficiency)
    winFun = hamming(winLen, 'periodic'); % Hamming window
    
    %% Compute STFT
    % Transform to time-frequency domain for spectral processing
    [S_noisy, F, T] = stft(noisySig, fs, 'Window', winFun, ...
                           'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    
    %% Voice Activity Detection
    % =========================================================================
    % Detect which frames contain speech vs. only noise
    % This helps estimate noise characteristics from non-speech regions
    % =========================================================================
    frameEnergy = sum(abs(S_noisy).^2, 1);
    frameEnergyDB = 10*log10(frameEnergy + eps);
    energyThreshold = mean(frameEnergyDB) - 5; % 5dB below mean
    vadDecisions = frameEnergyDB > energyThreshold;
    vadDecisions = medfilt1(double(vadDecisions), 5) > 0.5; % Smooth decisions
    
    speechPercent = sum(vadDecisions)/length(vadDecisions)*100;
    fprintf('  Speech content: %.1f%%\n', speechPercent);
    
    %% Estimate Noise PSD
    % =========================================================================
    % Use frames without speech to estimate noise spectrum
    % Critical step: quality of noise estimate affects all methods
    % =========================================================================
    noiseFrames = S_noisy(:, ~vadDecisions);
    if ~isempty(noiseFrames) && size(noiseFrames, 2) > 5
        % Good case: enough noise-only frames
        noisePSD = mean(abs(noiseFrames).^2, 2);
    else
        % Fallback: use beginning and end of recording
        % Common assumption: recordings start/end with ambient noise
        numNoiseFrames = min(20, floor(size(S_noisy, 2) * 0.1));
        noiseFrames = [S_noisy(:, 1:numNoiseFrames), S_noisy(:, end-numNoiseFrames+1:end)];
        noisePSD = mean(abs(noiseFrames).^2, 2);
        fprintf('  Warning: Limited noise frames detected, using boundary frames\n');
    end
    
    noisyPSD = abs(S_noisy).^2; % Power spectral density of noisy signal
    
    %% 1. Standard Wiener Filter
    % =========================================================================
    % Classic spectral subtraction-based noise reduction
    % Simple but effective for stationary noise
    % =========================================================================
    fprintf('  Applying Wiener filter...\n');
    wienerGain = max(1 - bsxfun(@rdivide, noisePSD, noisyPSD), 0);
    S_wiener = wienerGain .* S_noisy;
    enhancedSig_wiener = istft(S_wiener, fs, 'Window', winFun, ...
                               'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    enhancedSig_wiener = real(enhancedSig_wiener); % Ensure real output
    enhancedSig_wiener = matchLength(enhancedSig_wiener, noisySig);
    results.wiener{recIdx} = enhancedSig_wiener;
    
    %% 2. Improved Wiener Filter with Oversubtraction
    % =========================================================================
    % More aggressive noise removal with spectral floor
    % alpha > 1: More noise reduction (but may cause distortion)
    % beta > 0: Prevents complete spectral holes (reduces musical noise)
    % =========================================================================
    fprintf('  Applying Improved Wiener filter...\n');
    alpha = 2.0; % Oversubtraction factor
    beta = 0.01; % Spectral floor
    improvedWienerGain = max(1 - alpha * bsxfun(@rdivide, noisePSD, noisyPSD), beta);
    S_wiener_improved = improvedWienerGain .* S_noisy;
    enhancedSig_wiener_improved = istft(S_wiener_improved, fs, 'Window', winFun, ...
                                        'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    enhancedSig_wiener_improved = real(enhancedSig_wiener_improved); % Ensure real output
    enhancedSig_wiener_improved = matchLength(enhancedSig_wiener_improved, noisySig);
    results.wiener_improved{recIdx} = enhancedSig_wiener_improved;
    
    %% 3. Spectral Subtraction
    % =========================================================================
    % Classic magnitude subtraction approach
    % Different from Wiener: subtracts noise magnitude directly
    % More aggressive than Wiener, may produce "musical noise" artifacts
    % =========================================================================
    fprintf('  Applying Spectral Subtraction...\n');
    noisyMag = abs(S_noisy);
    noisyPhase = angle(S_noisy);
    % Convert noise PSD to magnitude and replicate across time
    noiseAmp = sqrt(repmat(noisePSD, [1, size(S_noisy, 2)]));
    % Subtract 1.5× noise magnitude, keep minimum of 5% original
    enhancedMag = max(noisyMag - 1.5 * noiseAmp, 0.05 * noisyMag);
    S_spectral_sub = enhancedMag .* exp(1j * noisyPhase);
    enhancedSig_spectral_sub = istft(S_spectral_sub, fs, 'Window', winFun, ...
                                     'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    enhancedSig_spectral_sub = real(enhancedSig_spectral_sub); % Ensure real output
    enhancedSig_spectral_sub = matchLength(enhancedSig_spectral_sub, noisySig);
    results.spectral_sub{recIdx} = enhancedSig_spectral_sub;
    
    %% 4. NLMS Adaptive Filter
    % =========================================================================
    % Adaptive noise cancellation approach
    % Uses delayed signal as reference to predict noise
    % Works well for temporally correlated noise
    % =========================================================================
    fprintf('  Applying NLMS filter...\n');
    filterOrder = 64; % Longer filter for higher sample rates
    mu = 0.1;         % Learning rate
    delta = 0.01;     % Regularization
    
    w = zeros(filterOrder, 1); % Filter weights
    enhancedSig_nlms = zeros(size(noisySig));
    refSignal = [zeros(filterOrder, 1); noisySig(1:end-filterOrder)];
    
    % NLMS algorithm (sample-by-sample processing)
    for n = filterOrder+1:length(noisySig)
        x = refSignal(n:-1:n-filterOrder+1);
        y = w' * x;
        e = noisySig(n) - y;
        enhancedSig_nlms(n) = e;
        w = w + (mu / (x'*x + delta)) * e * x;
    end
    results.nlms{recIdx} = enhancedSig_nlms;
    
    %% 5. CNN Enhancement (if available)
    % =========================================================================
    % Machine learning-based enhancement
    % IMPORTANT: CNN was trained on 8kHz NOIZEUS data
    % If your recording has different sample rate, we need to resample
    % =========================================================================
    if hasCNN
        fprintf('  Applying CNN enhancement...\n');
        try
            % CNN was trained on 8kHz NOIZEUS data, so resample if needed
            cnnTargetFs = 8000;
            
            if fs ~= cnnTargetFs
                fprintf('    Resampling from %d Hz to %d Hz for CNN...\n', fs, cnnTargetFs);
                % Resample signal to CNN's training sample rate
                noisySig_resampled = resample(noisySig, cnnTargetFs, fs);
            else
                noisySig_resampled = noisySig;
            end
            
            % Use CNN-appropriate STFT parameters for resampled signal
            winLen_cnn = round(0.032 * cnnTargetFs); % 32ms window at 8kHz
            hopSize_cnn = round(0.016 * cnnTargetFs); % 16ms hop
            nfft_cnn = 2^nextpow2(winLen_cnn);
            winFun_cnn = hamming(winLen_cnn, 'periodic');
            
            % Compute STFT of resampled signal
            [S_noisy_cnn, ~, ~] = stft(noisySig_resampled, cnnTargetFs, 'Window', winFun_cnn, ...
                                       'OverlapLength', winLen_cnn-hopSize_cnn, 'FFTLength', nfft_cnn);
            
            magNoisy = abs(S_noisy_cnn);
            phaseNoisy = angle(S_noisy_cnn);
            logMagNoisy = log(magNoisy + 1e-10);
            
            [nFreq, nTime] = size(logMagNoisy);
            fprintf('    CNN input size: %d x %d (max: %d x %d)\n', nFreq, nTime, maxFreqBins, maxTimeBins);
            
            % Check if input fits CNN's expected dimensions
            if nFreq <= maxFreqBins && nTime <= maxTimeBins
                % Input fits - process normally
                paddedInput = zeros(maxFreqBins, maxTimeBins, 1, 1);
                paddedInput(1:nFreq, 1:nTime, 1, 1) = logMagNoisy;
                
                % Predict mask using trained CNN
                predictedMaskPadded = predict(net, paddedInput);
                predictedMask = squeeze(predictedMaskPadded(1:nFreq, 1:nTime, 1, 1));
                
                % Apply mask to noisy spectrum
                S_cnn = predictedMask .* magNoisy .* exp(1j * phaseNoisy);
                enhancedSig_cnn_resampled = istft(S_cnn, cnnTargetFs, 'Window', winFun_cnn, ...
                                        'OverlapLength', winLen_cnn-hopSize_cnn, 'FFTLength', nfft_cnn);
                enhancedSig_cnn_resampled = real(enhancedSig_cnn_resampled); % Ensure real output
                
                % Resample back to original sample rate
                if fs ~= cnnTargetFs
                    fprintf('    Resampling CNN output back to %d Hz...\n', fs);
                    enhancedSig_cnn = resample(enhancedSig_cnn_resampled, fs, cnnTargetFs);
                else
                    enhancedSig_cnn = enhancedSig_cnn_resampled;
                end
                
                enhancedSig_cnn = matchLength(enhancedSig_cnn, noisySig);
                results.cnn{recIdx} = enhancedSig_cnn;
                fprintf('    CNN enhancement successful!\n');
            else
                % Input too large for CNN - process in overlapping segments
                % This happens when recording is longer than training samples
                fprintf('    Input still too large, processing in segments...\n');
                
                % Process in overlapping segments
                segmentSize = maxTimeBins - 20; % Leave some overlap margin
                numSegments = ceil(nTime / segmentSize);
                enhancedMag_full = zeros(size(magNoisy));
                
                % Process each segment separately
                for seg = 1:numSegments
                    startIdx = (seg-1) * segmentSize + 1;
                    endIdx = min(seg * segmentSize, nTime);
                    actualSize = endIdx - startIdx + 1;
                    
                    % Extract segment
                    segmentMag = logMagNoisy(:, startIdx:endIdx);
                    paddedInput = zeros(maxFreqBins, maxTimeBins, 1, 1);
                    paddedInput(1:nFreq, 1:actualSize, 1, 1) = segmentMag;
                    
                    % Predict mask for segment
                    predictedMaskPadded = predict(net, paddedInput);
                    predictedMask = squeeze(predictedMaskPadded(1:nFreq, 1:actualSize, 1, 1));
                    
                    % Apply mask to this segment
                    enhancedMag_full(:, startIdx:endIdx) = predictedMask .* magNoisy(:, startIdx:endIdx);
                end
                
                % Reconstruct full signal
                S_cnn = enhancedMag_full .* exp(1j * phaseNoisy);
                enhancedSig_cnn_resampled = istft(S_cnn, cnnTargetFs, 'Window', winFun_cnn, ...
                                        'OverlapLength', winLen_cnn-hopSize_cnn, 'FFTLength', nfft_cnn);
                enhancedSig_cnn_resampled = real(enhancedSig_cnn_resampled);
                
                % Resample back to original sample rate
                if fs ~= cnnTargetFs
                    enhancedSig_cnn = resample(enhancedSig_cnn_resampled, fs, cnnTargetFs);
                else
                    enhancedSig_cnn = enhancedSig_cnn_resampled;
                end
                
                enhancedSig_cnn = matchLength(enhancedSig_cnn, noisySig);
                results.cnn{recIdx} = enhancedSig_cnn;
                fprintf('    CNN segmented processing successful!\n');
            end
        catch ME
            fprintf('    CNN processing failed: %s\n', ME.message);
            % On failure, just copy original (no enhancement)
            results.cnn{recIdx} = noisySig;
        end
    end
    
    fprintf('  Done!\n\n');
end

%% Save Enhanced Audio Files
% =========================================================================
% Save all enhanced versions as WAV files for listening comparison
% This is the most important step for subjective evaluation
% Listen to each version to compare quality!
% =========================================================================
fprintf('Saving enhanced audio files...\n');
outputDir = 'enhanced_recordings';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

for recIdx = 1:numRecordings
    [~, baseName, ~] = fileparts(recordingFiles{recIdx});
    fs = results.fs(recIdx);
    
    % Save all versions with descriptive filenames
    audiowrite(fullfile(outputDir, [baseName '_original.wav']), results.original{recIdx}, fs);
    audiowrite(fullfile(outputDir, [baseName '_wiener.wav']), results.wiener{recIdx}, fs);
    audiowrite(fullfile(outputDir, [baseName '_wiener_improved.wav']), results.wiener_improved{recIdx}, fs);
    audiowrite(fullfile(outputDir, [baseName '_spectral_sub.wav']), results.spectral_sub{recIdx}, fs);
    audiowrite(fullfile(outputDir, [baseName '_nlms.wav']), results.nlms{recIdx}, fs);
    
    if hasCNN
        audiowrite(fullfile(outputDir, [baseName '_cnn.wav']), results.cnn{recIdx}, fs);
    end
    
    fprintf('  Saved enhanced versions of: %s\n', baseName);
end
fprintf('\n');

%% Generate Comprehensive Plots
% =========================================================================
% Create visualizations to understand what each algorithm is doing
% Since we don't have ground truth, we analyze:
% - Time-domain waveforms (amplitude changes)
% - Spectrograms (time-frequency patterns)
% - Power spectral density (frequency content)
% - Energy reduction by frequency band
% =========================================================================
fprintf('Generating visualizations...\n\n');

for recIdx = 1:numRecordings
    [~, baseName, ~] = fileparts(recordingFiles{recIdx});
    fs = results.fs(recIdx);
    noisySig = results.original{recIdx};
    
    % Recompute STFT parameters for this sample rate
    winLen = round(0.032 * fs);
    hopSize = round(0.016 * fs);
    nfft = 2^nextpow2(winLen);
    winFun = hamming(winLen, 'periodic');
    
    %% Figure 1: Time Domain Comparison
    % Shows amplitude waveforms - can see noise reduction in amplitude variations
    figure('Name', sprintf('%s - Time Domain', baseName), ...
           'Position', [50 50 1400 900]);
    
    t = (0:length(noisySig)-1) / fs; % Time vector in seconds
    
    numPlots = 5;
    if hasCNN
        numPlots = 6;
    end
    
    subplot(numPlots, 1, 1);
    plot(t, noisySig, 'Color', [0.5 0.5 0.5]);
    title('Original (Noisy) Recording');
    xlabel('Time (s)'); ylabel('Amplitude');
    grid on; xlim([0 max(t)]);
    
    subplot(numPlots, 1, 2);
    plot(t, results.wiener{recIdx}, 'b');
    title('Wiener Filter');
    xlabel('Time (s)'); ylabel('Amplitude');
    grid on; xlim([0 max(t)]);
    
    subplot(numPlots, 1, 3);
    plot(t, results.wiener_improved{recIdx}, 'g');
    title('Improved Wiener Filter');
    xlabel('Time (s)'); ylabel('Amplitude');
    grid on; xlim([0 max(t)]);
    
    subplot(numPlots, 1, 4);
    plot(t, results.spectral_sub{recIdx}, 'Color', [0.8 0.4 0]);
    title('Spectral Subtraction');
    xlabel('Time (s)'); ylabel('Amplitude');
    grid on; xlim([0 max(t)]);
    
    subplot(numPlots, 1, 5);
    plot(t, results.nlms{recIdx}, 'm');
    title('NLMS Adaptive Filter');
    xlabel('Time (s)'); ylabel('Amplitude');
    grid on; xlim([0 max(t)]);
    
    if hasCNN
        subplot(numPlots, 1, 6);
        plot(t, results.cnn{recIdx}, 'r');
        title('CNN Enhancement');
        xlabel('Time (s)'); ylabel('Amplitude');
        grid on; xlim([0 max(t)]);
    end
    
    sgtitle(sprintf('Time Domain Comparison - %s', strrep(baseName, '_', ' ')), 'FontSize', 14);
    saveas(gcf, fullfile(outputDir, [baseName '_time_domain.png']));
    
    %% Figure 2: Spectrogram Comparison
    % Shows time-frequency representation - most informative for speech
    % Look for: noise reduction (cleaner background), speech preservation (formants)
    figure('Name', sprintf('%s - Spectrograms', baseName), ...
           'Position', [100 100 1400 900]);
    
    if hasCNN
        nRows = 3; nCols = 2;
    else
        nRows = 3; nCols = 2;
    end
    
    subplot(nRows, nCols, 1);
    spectrogram(noisySig, winFun, winLen-hopSize, nfft, fs, 'yaxis');
    title('Original (Noisy)');
    colorbar; caxis([-120 -20]); % Fixed color scale for comparison
    
    subplot(nRows, nCols, 2);
    spectrogram(results.wiener{recIdx}, winFun, winLen-hopSize, nfft, fs, 'yaxis');
    title('Wiener Filter');
    colorbar; caxis([-120 -20]);
    
    subplot(nRows, nCols, 3);
    spectrogram(results.wiener_improved{recIdx}, winFun, winLen-hopSize, nfft, fs, 'yaxis');
    title('Improved Wiener');
    colorbar; caxis([-120 -20]);
    
    subplot(nRows, nCols, 4);
    spectrogram(results.spectral_sub{recIdx}, winFun, winLen-hopSize, nfft, fs, 'yaxis');
    title('Spectral Subtraction');
    colorbar; caxis([-120 -20]);
    
    subplot(nRows, nCols, 5);
    spectrogram(results.nlms{recIdx}, winFun, winLen-hopSize, nfft, fs, 'yaxis');
    title('NLMS Filter');
    colorbar; caxis([-120 -20]);
    
    if hasCNN
        subplot(nRows, nCols, 6);
        spectrogram(results.cnn{recIdx}, winFun, winLen-hopSize, nfft, fs, 'yaxis');
        title('CNN Enhancement');
        colorbar; caxis([-120 -20]);
    end
    
    sgtitle(sprintf('Spectrogram Comparison - %s', strrep(baseName, '_', ' ')), 'FontSize', 14);
    saveas(gcf, fullfile(outputDir, [baseName '_spectrograms.png']));
    
    %% Figure 3: Power Spectral Density
    % Shows average frequency content - reveals which frequencies are attenuated
    % Noise reduction should lower power, especially at frequencies dominated by noise
    figure('Name', sprintf('%s - PSD', baseName), ...
           'Position', [150 150 1200 600]);
    
    % Compute PSD for each signal using Welch's method
    [pxx_orig, f] = pwelch(noisySig, hamming(1024), 512, 2048, fs);
    [pxx_wiener, ~] = pwelch(results.wiener{recIdx}, hamming(1024), 512, 2048, fs);
    [pxx_wiener_imp, ~] = pwelch(results.wiener_improved{recIdx}, hamming(1024), 512, 2048, fs);
    [pxx_spec_sub, ~] = pwelch(results.spectral_sub{recIdx}, hamming(1024), 512, 2048, fs);
    [pxx_nlms, ~] = pwelch(results.nlms{recIdx}, hamming(1024), 512, 2048, fs);
    
    % Plot all PSDs on same axis
    plot(f, 10*log10(pxx_orig), 'Color', [0.5 0.5 0.5], 'LineWidth', 2, 'DisplayName', 'Original');
    hold on;
    plot(f, 10*log10(pxx_wiener), 'b', 'LineWidth', 1.5, 'DisplayName', 'Wiener');
    plot(f, 10*log10(pxx_wiener_imp), 'g', 'LineWidth', 1.5, 'DisplayName', 'Improved Wiener');
    plot(f, 10*log10(pxx_spec_sub), 'Color', [0.8 0.4 0], 'LineWidth', 1.5, 'DisplayName', 'Spectral Sub');
    plot(f, 10*log10(pxx_nlms), 'm', 'LineWidth', 1.5, 'DisplayName', 'NLMS');
    
    if hasCNN
        [pxx_cnn, ~] = pwelch(results.cnn{recIdx}, hamming(1024), 512, 2048, fs);
        plot(f, 10*log10(pxx_cnn), 'r', 'LineWidth', 1.5, 'DisplayName', 'CNN');
    end
    hold off;
    
    grid on;
    xlabel('Frequency (Hz)');
    ylabel('Power/Frequency (dB/Hz)');
    title(sprintf('Power Spectral Density - %s', strrep(baseName, '_', ' ')));
    legend('Location', 'northeast');
    xlim([0 min(fs/2, 8000)]); % Show up to 8kHz or Nyquist
    
    saveas(gcf, fullfile(outputDir, [baseName '_psd.png']));
    
    %% Figure 4: Energy Reduction Analysis
    % Analyze noise reduction in different frequency bands
    % Speech energy is concentrated in certain bands (300-3400 Hz for telephone)
    % Ideally: reduce noise in all bands, especially outside speech frequencies
    figure('Name', sprintf('%s - Energy Analysis', baseName), ...
           'Position', [200 200 1000 500]);
    
    % Define frequency bands for analysis
    bands = [0 500; 500 1000; 1000 2000; 2000 4000; 4000 min(fs/2, 8000)];
    bandNames = {'0-500 Hz', '500-1k Hz', '1-2k Hz', '2-4k Hz', '4k+ Hz'};
    
    % Compute energy change in each band relative to original
    energyReduction = zeros(length(bands), 4);
    if hasCNN
        energyReduction = zeros(length(bands), 5);
    end
    
    for b = 1:size(bands, 1)
        freqIdx = f >= bands(b,1) & f < bands(b,2);
        origEnergy = sum(pxx_orig(freqIdx));
        
        % Energy change in dB: 10*log10(enhanced/original)
        % Negative = energy reduced (good for noise)
        energyReduction(b, 1) = 10*log10(sum(pxx_wiener(freqIdx)) / origEnergy);
        energyReduction(b, 2) = 10*log10(sum(pxx_wiener_imp(freqIdx)) / origEnergy);
        energyReduction(b, 3) = 10*log10(sum(pxx_spec_sub(freqIdx)) / origEnergy);
        energyReduction(b, 4) = 10*log10(sum(pxx_nlms(freqIdx)) / origEnergy);
        
        if hasCNN
            energyReduction(b, 5) = 10*log10(sum(pxx_cnn(freqIdx)) / origEnergy);
        end
    end
    
    % Plot as grouped bar chart
    bar(energyReduction);
    set(gca, 'XTickLabel', bandNames);
    ylabel('Energy Change (dB)');
    title(sprintf('Energy Reduction by Frequency Band - %s', strrep(baseName, '_', ' ')));
    if hasCNN
        legend('Wiener', 'Improved Wiener', 'Spectral Sub', 'NLMS', 'CNN', 'Location', 'southwest');
    else
        legend('Wiener', 'Improved Wiener', 'Spectral Sub', 'NLMS', 'Location', 'southwest');
    end
    grid on;
    yline(0, 'k--', 'LineWidth', 1.5); % Zero line (no change)
    
    saveas(gcf, fullfile(outputDir, [baseName '_energy_analysis.png']));
end

%% Summary Figure: Compare Both Recordings
% =========================================================================
% If multiple recordings, compare algorithm performance across environments
% This shows which algorithms are more robust to different noise types
% =========================================================================
if numRecordings >= 2
    figure('Name', 'Recording Comparison Summary', 'Position', [50 50 1400 700]);
    
    % Compute overall noise reduction for each recording and method
    noiseReduction = zeros(numRecordings, 4);
    if hasCNN
        noiseReduction = zeros(numRecordings, 5);
    end
    
    for recIdx = 1:numRecordings
        origEnergy = sum(results.original{recIdx}.^2);
        noiseReduction(recIdx, 1) = 10*log10(sum(results.wiener{recIdx}.^2) / origEnergy);
        noiseReduction(recIdx, 2) = 10*log10(sum(results.wiener_improved{recIdx}.^2) / origEnergy);
        noiseReduction(recIdx, 3) = 10*log10(sum(results.spectral_sub{recIdx}.^2) / origEnergy);
        noiseReduction(recIdx, 4) = 10*log10(sum(results.nlms{recIdx}.^2) / origEnergy);
        if hasCNN
            noiseReduction(recIdx, 5) = 10*log10(sum(results.cnn{recIdx}.^2) / origEnergy);
        end
    end
    
    subplot(1,2,1);
    bar(noiseReduction');
    if hasCNN
        set(gca, 'XTickLabel', {'Wiener', 'Imp. Wiener', 'Spec. Sub', 'NLMS', 'CNN'});
    else
        set(gca, 'XTickLabel', {'Wiener', 'Imp. Wiener', 'Spec. Sub', 'NLMS'});
    end
    ylabel('Energy Change (dB)');
    title('Overall Signal Energy Change');
    % Create legend with noise type names (extracted from filename)
    legend(cellfun(@(x) strrep(x(12:end-4), '_', ' '), recordingFiles, 'UniformOutput', false), ...
           'Location', 'southwest');
    grid on;
    yline(0, 'k--', 'LineWidth', 1.5);
    xtickangle(45);
    
    subplot(1,2,2);
    % Compute high-frequency noise reduction (where most noise often is)
    hfNoiseReduction = zeros(numRecordings, 4);
    if hasCNN
        hfNoiseReduction = zeros(numRecordings, 5);
    end
    
    for recIdx = 1:numRecordings
        fs = results.fs(recIdx);
        % Compute PSDs
        [pxx_orig, f] = pwelch(results.original{recIdx}, hamming(1024), 512, 2048, fs);
        [pxx_wiener, ~] = pwelch(results.wiener{recIdx}, hamming(1024), 512, 2048, fs);
        [pxx_wiener_imp, ~] = pwelch(results.wiener_improved{recIdx}, hamming(1024), 512, 2048, fs);
        [pxx_spec_sub, ~] = pwelch(results.spectral_sub{recIdx}, hamming(1024), 512, 2048, fs);
        [pxx_nlms, ~] = pwelch(results.nlms{recIdx}, hamming(1024), 512, 2048, fs);
        
        % Focus on high frequencies (>2kHz)
        hfIdx = f > 2000;
        origHFEnergy = sum(pxx_orig(hfIdx));
        
        hfNoiseReduction(recIdx, 1) = 10*log10(sum(pxx_wiener(hfIdx)) / origHFEnergy);
        hfNoiseReduction(recIdx, 2) = 10*log10(sum(pxx_wiener_imp(hfIdx)) / origHFEnergy);
        hfNoiseReduction(recIdx, 3) = 10*log10(sum(pxx_spec_sub(hfIdx)) / origHFEnergy);
        hfNoiseReduction(recIdx, 4) = 10*log10(sum(pxx_nlms(hfIdx)) / origHFEnergy);
        
        if hasCNN
            [pxx_cnn, ~] = pwelch(results.cnn{recIdx}, hamming(1024), 512, 2048, fs);
            hfNoiseReduction(recIdx, 5) = 10*log10(sum(pxx_cnn(hfIdx)) / origHFEnergy);
        end
    end
    
    bar(hfNoiseReduction');
    if hasCNN
        set(gca, 'XTickLabel', {'Wiener', 'Imp. Wiener', 'Spec. Sub', 'NLMS', 'CNN'});
    else
        set(gca, 'XTickLabel', {'Wiener', 'Imp. Wiener', 'Spec. Sub', 'NLMS'});
    end
    ylabel('Energy Change (dB)');
    title('High-Frequency (>2kHz) Energy Reduction');
    legend(cellfun(@(x) strrep(x(12:end-4), '_', ' '), recordingFiles, 'UniformOutput', false), ...
           'Location', 'southwest');
    grid on;
    yline(0, 'k--', 'LineWidth', 1.5);
    xtickangle(45);
    
    sgtitle('Algorithm Performance Comparison Across Recordings', 'FontSize', 14);
    saveas(gcf, fullfile(outputDir, 'comparison_summary.png'));
end

%% Save Results
% Save all results for future analysis
save(fullfile(outputDir, 'test_results.mat'), 'results');

%% Final Summary
% =========================================================================
% Print summary of what was done and recommendations for evaluation
% =========================================================================
fprintf('========================================\n');
fprintf('Processing Complete!\n');
fprintf('========================================\n\n');

fprintf('Output files saved to: %s/\n', outputDir);
fprintf('\nFor each recording, you now have:\n');
fprintf('  - Original and enhanced audio files (.wav)\n');
fprintf('  - Time domain comparison plot\n');
fprintf('  - Spectrogram comparison plot\n');
fprintf('  - Power spectral density plot\n');
fprintf('  - Energy analysis by frequency band\n');

if numRecordings >= 2
    fprintf('  - Cross-recording comparison summary\n');
end

fprintf('\nListening Recommendations:\n');
fprintf('  1. Listen to each enhanced version to subjectively evaluate quality\n');
fprintf('  2. Pay attention to:\n');
fprintf('     - Speech clarity and intelligibility\n');
fprintf('     - Presence of musical noise or artifacts\n');
fprintf('     - Overall naturalness of the speech\n');
fprintf('  3. Compare how each algorithm handles different noise types\n');
fprintf('     (restaurant vs train noise)\n\n');

fprintf('Key Observations from Plots:\n');
fprintf('  - Spectrograms show how each method removes noise patterns\n');
fprintf('  - PSD plots show frequency-specific noise reduction\n');
fprintf('  - Energy analysis reveals which frequency bands are most affected\n\n');

fprintf('========================================\n');

%% Helper Function
% =========================================================================
% Utility function to match signal lengths
% Ensures all enhanced signals have same length as original
% Important for fair comparison and audio file saving
% =========================================================================
function out = matchLength(sig, ref)
    % Match the length of sig to ref
    if length(sig) >= length(ref)
        out = sig(1:length(ref)); % Trim if longer
    else
        out = [sig; zeros(length(ref) - length(sig), 1)]; % Pad if shorter
    end
end
