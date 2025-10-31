% =========================================================================
% Helper Functions for Speech Enhancement Project
% =========================================================================
% This file contains utility functions used across the project
%
% Functions included:
% 1. SimpleVAD - Voice Activity Detection
% 2. computeSNR - SNR calculation
% 3. computeSTOI - Short-Time Objective Intelligibility (placeholder)
% 4. wienerFilter - Wiener filtering
% 5. spectralSubtraction - Spectral subtraction
%
% Author: Alaa Aldirani
% =========================================================================

%% SimpleVAD - Voice Activity Detection
function VADdata = SimpleVAD(signalLevel, dTime, VADdata)
    % Simple Voice Activity Detector based on energy thresholding
    % 
    % Inputs:
    %   signalLevel - Current frame energy level
    %   dTime - Current time
    %   VADdata - Structure containing VAD state
    %
    % Output:
    %   VADdata - Updated VAD state structure
    
    if ~isfield(VADdata, 'initialized')
        % Initialize VAD parameters
        VADdata.initialized = true;
        VADdata.noiseLevel = signalLevel;
        VADdata.threshold = 2.0 * signalLevel; % Initial threshold
        VADdata.tSignal = false;
        VADdata.hangoverTime = 0.3; % 300ms hangover
        VADdata.lastSpeechTime = -1;
        VADdata.alpha = 0.95; % Noise estimate smoothing factor
    end
    
    % Update noise level estimate during silence
    if signalLevel < VADdata.threshold
        VADdata.noiseLevel = VADdata.alpha * VADdata.noiseLevel + ...
                            (1 - VADdata.alpha) * signalLevel;
    end
    
    % Update threshold
    VADdata.threshold = 2.5 * VADdata.noiseLevel;
    
    % Detect speech
    if signalLevel > VADdata.threshold
        VADdata.tSignal = true;
        VADdata.lastSpeechTime = dTime;
    else
        % Check hangover time
        if (dTime - VADdata.lastSpeechTime) < VADdata.hangoverTime
            VADdata.tSignal = true;
        else
            VADdata.tSignal = false;
        end
    end
end

%% computeSNR - Signal-to-Noise Ratio calculation
function snrValue = computeSNR(cleanSignal, noisySignal)
    % Compute SNR between clean and noisy signals
    %
    % Inputs:
    %   cleanSignal - Clean reference signal
    %   noisySignal - Noisy signal
    %
    % Output:
    %   snrValue - SNR in dB
    
    % Ensure signals are the same length
    minLen = min(length(cleanSignal), length(noisySignal));
    cleanSignal = cleanSignal(1:minLen);
    noisySignal = noisySignal(1:minLen);
    
    % Compute noise
    noise = noisySignal - cleanSignal;
    
    % Compute power
    signalPower = sum(cleanSignal.^2);
    noisePower = sum(noise.^2);
    
    % Compute SNR
    if noisePower == 0
        snrValue = Inf;
    else
        snrValue = 10 * log10(signalPower / noisePower);
    end
end

%% wienerFilter - Wiener filtering implementation
function [enhancedSignal, wienerGain] = wienerFilter(noisySignal, fs, ...
                                                      noisePSD, varargin)
    % Apply Wiener filter to noisy signal
    %
    % Inputs:
    %   noisySignal - Noisy input signal
    %   fs - Sample rate
    %   noisePSD - Noise power spectral density estimate
    %   varargin - Optional: 'Alpha' (oversubtraction), 'Beta' (floor)
    %
    % Outputs:
    %   enhancedSignal - Enhanced output signal
    %   wienerGain - Wiener gain function
    
    % Parse optional inputs
    p = inputParser;
    addParameter(p, 'Alpha', 1.0, @isnumeric);
    addParameter(p, 'Beta', 0.0, @isnumeric);
    addParameter(p, 'WinLen', round(0.032*fs), @isnumeric);
    addParameter(p, 'HopSize', round(0.016*fs), @isnumeric);
    parse(p, varargin{:});
    
    alpha = p.Results.Alpha;
    beta = p.Results.Beta;
    winLen = p.Results.WinLen;
    hopSize = p.Results.HopSize;
    
    % STFT parameters
    nfft = 2^nextpow2(winLen);
    winFun = hamming(winLen, 'periodic');
    
    % Compute STFT
    [S_noisy, ~, ~] = stft(noisySignal, fs, 'Window', winFun, ...
                           'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    
    % Compute Wiener gain
    noisyPSD = abs(S_noisy).^2;
    wienerGain = max(1 - alpha * bsxfun(@rdivide, noisePSD, noisyPSD), beta);
    
    % Apply gain
    S_enhanced = wienerGain .* S_noisy;
    
    % Inverse STFT
    enhancedSignal = istft(S_enhanced, fs, 'Window', winFun, ...
                          'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    
    % Trim or pad to original length
    if length(enhancedSignal) >= length(noisySignal)
        enhancedSignal = enhancedSignal(1:length(noisySignal));
    else
        enhancedSignal = [enhancedSignal; zeros(length(noisySignal) - length(enhancedSignal), 1)];
    end
end

%% spectralSubtraction - Spectral subtraction implementation
function enhancedSignal = spectralSubtraction(noisySignal, fs, noisePSD, varargin)
    % Apply spectral subtraction to noisy signal
    %
    % Inputs:
    %   noisySignal - Noisy input signal
    %   fs - Sample rate
    %   noisePSD - Noise power spectral density estimate
    %   varargin - Optional: 'Alpha' (oversubtraction), 'Beta' (floor)
    %
    % Output:
    %   enhancedSignal - Enhanced output signal
    
    % Parse optional inputs
    p = inputParser;
    addParameter(p, 'Alpha', 2.0, @isnumeric);
    addParameter(p, 'Beta', 0.01, @isnumeric);
    addParameter(p, 'WinLen', round(0.032*fs), @isnumeric);
    addParameter(p, 'HopSize', round(0.016*fs), @isnumeric);
    parse(p, varargin{:});
    
    alpha = p.Results.Alpha;
    beta = p.Results.Beta;
    winLen = p.Results.WinLen;
    hopSize = p.Results.HopSize;
    
    % STFT parameters
    nfft = 2^nextpow2(winLen);
    winFun = hamming(winLen, 'periodic');
    
    % Compute STFT
    [S_noisy, ~, ~] = stft(noisySignal, fs, 'Window', winFun, ...
                           'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    
    % Spectral subtraction
    noisyMag = abs(S_noisy);
    noisyPhase = angle(S_noisy);
    
    % Subtract noise spectrum with oversubtraction
    noiseAmp = sqrt(repmat(noisePSD, [1, size(S_noisy, 2)]));
    enhancedMag = max(noisyMag - alpha * noiseAmp, beta * noisyMag);
    
    % Reconstruct complex spectrum
    S_enhanced = enhancedMag .* exp(1j * noisyPhase);
    
    % Inverse STFT
    enhancedSignal = istft(S_enhanced, fs, 'Window', winFun, ...
                          'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    
    % Trim to original length
    enhancedSignal = enhancedSignal(1:length(noisySignal));
end

%% estimateNoisePSD - Estimate noise PSD from signal
function noisePSD = estimateNoisePSD(signal, fs, varargin)
    % Estimate noise power spectral density from signal
    %
    % Inputs:
    %   signal - Input signal
    %   fs - Sample rate
    %   varargin - Optional: 'Method' ('vad' or 'initial')
    %
    % Output:
    %   noisePSD - Estimated noise PSD
    
    % Parse optional inputs
    p = inputParser;
    addParameter(p, 'Method', 'vad', @ischar);
    addParameter(p, 'WinLen', round(0.032*fs), @isnumeric);
    addParameter(p, 'HopSize', round(0.016*fs), @isnumeric);
    addParameter(p, 'InitialFrames', 10, @isnumeric);
    parse(p, varargin{:});
    
    method = p.Results.Method;
    winLen = p.Results.WinLen;
    hopSize = p.Results.HopSize;
    initialFrames = p.Results.InitialFrames;
    
    % STFT parameters
    nfft = 2^nextpow2(winLen);
    winFun = hamming(winLen, 'periodic');
    
    % Compute STFT
    [S, ~, ~] = stft(signal, fs, 'Window', winFun, ...
                     'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    
    if strcmp(method, 'initial')
        % Use initial frames as noise estimate
        noiseFrames = S(:, 1:min(initialFrames, size(S,2)));
        noisePSD = mean(abs(noiseFrames).^2, 2);
    else
        % Use VAD to identify noise frames
        frameEnergy = sum(abs(S).^2, 1);
        frameEnergyDB = 10*log10(frameEnergy + eps);
        energyThreshold = mean(frameEnergyDB) - 5;
        vadDecisions = frameEnergyDB > energyThreshold;
        vadDecisions = medfilt1(double(vadDecisions), 5) > 0.5;
        
        % Estimate from noise frames
        noiseFrames = S(:, ~vadDecisions);
        if isempty(noiseFrames)
            % Fallback to initial frames
            noiseFrames = S(:, 1:min(initialFrames, size(S,2)));
        end
        noisePSD = mean(abs(noiseFrames).^2, 2);
    end
end

%% nlmsFilter - NLMS adaptive filter
function [enhancedSignal, weights] = nlmsFilter(noisySignal, varargin)
    % Apply NLMS adaptive filter
    %
    % Inputs:
    %   noisySignal - Noisy input signal
    %   varargin - Optional parameters
    %
    % Outputs:
    %   enhancedSignal - Enhanced signal
    %   weights - Final filter weights
    
    % Parse optional inputs
    p = inputParser;
    addParameter(p, 'FilterOrder', 32, @isnumeric);
    addParameter(p, 'StepSize', 0.1, @isnumeric);
    addParameter(p, 'Delta', 0.01, @isnumeric);
    parse(p, varargin{:});
    
    filterOrder = p.Results.FilterOrder;
    mu = p.Results.StepSize;
    delta = p.Results.Delta;
    
    % Initialize
    weights = zeros(filterOrder, 1);
    enhancedSignal = zeros(size(noisySignal));
    
    % Create reference signal (delayed version)
    refSignal = [zeros(filterOrder, 1); noisySignal(1:end-filterOrder)];
    
    % NLMS algorithm
    for n = filterOrder+1:length(noisySignal)
        x = refSignal(n:-1:n-filterOrder+1);
        y = weights' * x;
        e = noisySignal(n) - y;
        enhancedSignal(n) = e;
        weights = weights + (mu / (x'*x + delta)) * e * x;
    end
    
    % Remove initial transient
    enhancedSignal = enhancedSignal(filterOrder+1:end);
end

%% plotSpectrograms - Convenient spectrogram plotting
function plotSpectrograms(signals, titles, fs, varargin)
    % Plot spectrograms for multiple signals
    %
    % Inputs:
    %   signals - Cell array of signals
    %   titles - Cell array of titles
    %   fs - Sample rate
    %   varargin - Optional parameters
    
    % Parse optional inputs
    p = inputParser;
    addParameter(p, 'WinLen', round(0.032*fs), @isnumeric);
    addParameter(p, 'HopSize', round(0.016*fs), @isnumeric);
    addParameter(p, 'Layout', [], @isnumeric);
    parse(p, varargin{:});
    
    winLen = p.Results.WinLen;
    hopSize = p.Results.HopSize;
    layout = p.Results.Layout;
    
    nfft = 2^nextpow2(winLen);
    winFun = hamming(winLen, 'periodic');
    
    numSignals = length(signals);
    
    if isempty(layout)
        nRows = ceil(sqrt(numSignals));
        nCols = ceil(numSignals / nRows);
    else
        nRows = layout(1);
        nCols = layout(2);
    end
    
    figure('Position', [100 100 300*nCols 250*nRows]);
    
    for i = 1:numSignals
        subplot(nRows, nCols, i);
        spectrogram(signals{i}, winFun, winLen-hopSize, nfft, fs, 'yaxis');
        title(titles{i});
        colorbar;
    end
end

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