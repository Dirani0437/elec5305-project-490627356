% =========================================================================
% Batch Processing Script for Adaptive Filtering Evaluation
% =========================================================================
% This script processes multiple samples from the test set and computes
% average performance metrics for all adaptive filtering methods.
%
% Author: Alaa Aldirani
% Project: Real-Time Speech Enhancement
% =========================================================================

clear; close all; clc;

%% Configuration
fprintf('========================================\n');
fprintf('Batch Processing - Adaptive Filtering\n');
fprintf('========================================\n\n');

% Load prepared dataset
fprintf('Loading prepared dataset...\n');
load('prepared_data/noizeus_prepared.mat');
fprintf('Dataset loaded!\n\n');

% Decide how many samples to process (set to -1 for all test samples)
numSamplesToProcess = 50; % Process 50 samples, or set to -1 for all

if numSamplesToProcess == -1 || numSamplesToProcess > length(testData)
    numSamplesToProcess = length(testData);
end

fprintf('Processing %d test samples...\n\n', numSamplesToProcess);

%% STFT Parameters
fs = testData(1).fs;
winLen = round(0.032 * fs);
hopSize = round(0.016 * fs);
nfft = 2^nextpow2(winLen);
winFun = hamming(winLen, 'periodic');

%% Initialize Results Storage
results = struct();
results.snr_noisy = zeros(numSamplesToProcess, 1);
results.snr_wiener = zeros(numSamplesToProcess, 1);
results.snr_wiener_improved = zeros(numSamplesToProcess, 1);
results.snr_nlms = zeros(numSamplesToProcess, 1);
results.snr_improvement_wiener = zeros(numSamplesToProcess, 1);
results.snr_improvement_wiener_improved = zeros(numSamplesToProcess, 1);
results.snr_improvement_nlms = zeros(numSamplesToProcess, 1);
results.noiseTypes = cell(numSamplesToProcess, 1);
results.snrLevels = cell(numSamplesToProcess, 1); % Changed to cell array for strings

%% Process Each Sample
fprintf('Starting batch processing...\n');
fprintf('Progress: ');

for idx = 1:numSamplesToProcess
    % Progress indicator
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
    results.snrLevels{idx} = snrLevel; % Store as string
    
    %% STFT and VAD
    [S_noisy, ~, ~] = stft(noisySig, fs, 'Window', winFun, ...
                           'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    
    % Simple energy-based VAD
    frameEnergy = sum(abs(S_noisy).^2, 1);
    frameEnergyDB = 10*log10(frameEnergy + eps);
    energyThreshold = mean(frameEnergyDB) - 5;
    vadDecisions = frameEnergyDB > energyThreshold;
    vadDecisions = medfilt1(double(vadDecisions), 5) > 0.5;
    
    %% Wiener Filtering
    % Estimate noise spectrum
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
    
    % Trim or pad to original length
    if length(enhancedSig_wiener) >= length(noisySig)
        enhancedSig_wiener = enhancedSig_wiener(1:length(noisySig));
    else
        enhancedSig_wiener = [enhancedSig_wiener; zeros(length(noisySig) - length(enhancedSig_wiener), 1)];
    end
    
    % Improved Wiener filter
    alpha = 2.0;
    beta = 0.01;
    improvedWienerGain = max(1 - alpha * bsxfun(@rdivide, noisePSD, noisyPSD), beta);
    S_wiener_improved = improvedWienerGain .* S_noisy;
    enhancedSig_wiener_improved = istft(S_wiener_improved, fs, 'Window', winFun, ...
                                        'OverlapLength', winLen-hopSize, 'FFTLength', nfft);
    
    % Trim or pad to original length
    if length(enhancedSig_wiener_improved) >= length(noisySig)
        enhancedSig_wiener_improved = enhancedSig_wiener_improved(1:length(noisySig));
    else
        enhancedSig_wiener_improved = [enhancedSig_wiener_improved; zeros(length(noisySig) - length(enhancedSig_wiener_improved), 1)];
    end
    
    %% NLMS Filtering
    filterOrder = 32;
    mu = 0.1;
    delta = 0.01;
    
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
    cleanSig_trimmed = cleanSig(filterOrder+1:end);
    
    %% Compute SNR
    try
        results.snr_noisy(idx) = snr(cleanSig, noisySig - cleanSig);
        results.snr_wiener(idx) = snr(cleanSig, enhancedSig_wiener - cleanSig);
        results.snr_wiener_improved(idx) = snr(cleanSig, enhancedSig_wiener_improved - cleanSig);
        results.snr_nlms(idx) = snr(cleanSig_trimmed, enhancedSig_nlms - cleanSig_trimmed);
        
        results.snr_improvement_wiener(idx) = results.snr_wiener(idx) - results.snr_noisy(idx);
        results.snr_improvement_wiener_improved(idx) = results.snr_wiener_improved(idx) - results.snr_noisy(idx);
        results.snr_improvement_nlms(idx) = results.snr_nlms(idx) - results.snr_noisy(idx);
    catch
        % If SNR computation fails, set to NaN
        results.snr_noisy(idx) = NaN;
        results.snr_wiener(idx) = NaN;
        results.snr_wiener_improved(idx) = NaN;
        results.snr_nlms(idx) = NaN;
        results.snr_improvement_wiener(idx) = NaN;
        results.snr_improvement_wiener_improved(idx) = NaN;
        results.snr_improvement_nlms(idx) = NaN;
    end
end

fprintf('\nBatch processing complete!\n\n');

%% Compute Statistics
fprintf('========================================\n');
fprintf('Overall Performance Statistics\n');
fprintf('========================================\n\n');

% Remove NaN values
validIdx = ~isnan(results.snr_noisy);

fprintf('Average SNR (dB):\n');
fprintf('  Noisy:            %.2f ± %.2f\n', ...
        mean(results.snr_noisy(validIdx)), std(results.snr_noisy(validIdx)));
fprintf('  Wiener:           %.2f ± %.2f\n', ...
        mean(results.snr_wiener(validIdx)), std(results.snr_wiener(validIdx)));
fprintf('  Improved Wiener:  %.2f ± %.2f\n', ...
        mean(results.snr_wiener_improved(validIdx)), std(results.snr_wiener_improved(validIdx)));
fprintf('  NLMS:             %.2f ± %.2f\n\n', ...
        mean(results.snr_nlms(validIdx)), std(results.snr_nlms(validIdx)));

fprintf('Average SNR Improvement (dB):\n');
fprintf('  Wiener:           %.2f ± %.2f\n', ...
        mean(results.snr_improvement_wiener(validIdx)), std(results.snr_improvement_wiener(validIdx)));
fprintf('  Improved Wiener:  %.2f ± %.2f\n', ...
        mean(results.snr_improvement_wiener_improved(validIdx)), std(results.snr_improvement_wiener_improved(validIdx)));
fprintf('  NLMS:             %.2f ± %.2f\n\n', ...
        mean(results.snr_improvement_nlms(validIdx)), std(results.snr_improvement_nlms(validIdx)));

%% Performance by Noise Type
fprintf('Performance by Noise Type:\n');
fprintf('----------------------------------\n');

uniqueNoises = unique(results.noiseTypes);
for i = 1:length(uniqueNoises)
    noiseIdx = strcmp(results.noiseTypes, uniqueNoises{i}) & validIdx;
    fprintf('\n%s:\n', uniqueNoises{i});
    fprintf('  Wiener improvement:          %.2f dB\n', mean(results.snr_improvement_wiener(noiseIdx)));
    fprintf('  Improved Wiener improvement: %.2f dB\n', mean(results.snr_improvement_wiener_improved(noiseIdx)));
    fprintf('  NLMS improvement:            %.2f dB\n', mean(results.snr_improvement_nlms(noiseIdx)));
end

%% Performance by SNR Level
fprintf('\n\nPerformance by SNR Level:\n');
fprintf('----------------------------------\n');

uniqueSNRs = unique(results.snrLevels);
for i = 1:length(uniqueSNRs)
    snrIdx = strcmp(results.snrLevels, uniqueSNRs{i}) & validIdx; % Use strcmp for strings
    fprintf('\n%s:\n', uniqueSNRs{i}); % Print as string
    fprintf('  Wiener improvement:          %.2f dB\n', mean(results.snr_improvement_wiener(snrIdx)));
    fprintf('  Improved Wiener improvement: %.2f dB\n', mean(results.snr_improvement_wiener_improved(snrIdx)));
    fprintf('  NLMS improvement:            %.2f dB\n', mean(results.snr_improvement_nlms(snrIdx)));
end

%% Visualization
fprintf('\n\nGenerating performance visualizations...\n');

% Figure 1: SNR Comparison Boxplot
figure('Name', 'SNR Performance Comparison', 'Position', [50 50 1200 600]);

% Figure 1: SNR Comparison
figure('Name', 'SNR Performance Comparison', 'Position', [50 50 1200 600]);

subplot(1,2,1);
% Create grouped bar chart instead of boxplot
meanSNR = [mean(results.snr_noisy(validIdx)), mean(results.snr_wiener(validIdx)), ...
           mean(results.snr_wiener_improved(validIdx)), mean(results.snr_nlms(validIdx))];
stdSNR = [std(results.snr_noisy(validIdx)), std(results.snr_wiener(validIdx)), ...
          std(results.snr_wiener_improved(validIdx)), std(results.snr_nlms(validIdx))];
bar(meanSNR);
hold on;
errorbar(1:4, meanSNR, stdSNR, 'k.', 'LineWidth', 1.5);
hold off;
set(gca, 'XTickLabel', {'Noisy', 'Wiener', 'Improved Wiener', 'NLMS'});
ylabel('SNR (dB)');
title('Mean SNR Across All Methods');
grid on;

subplot(1,2,2);
% Create grouped bar chart for improvements
meanImprovement = [mean(results.snr_improvement_wiener(validIdx)), ...
                   mean(results.snr_improvement_wiener_improved(validIdx)), ...
                   mean(results.snr_improvement_nlms(validIdx))];
stdImprovement = [std(results.snr_improvement_wiener(validIdx)), ...
                  std(results.snr_improvement_wiener_improved(validIdx)), ...
                  std(results.snr_improvement_nlms(validIdx))];
bar(meanImprovement);
hold on;
errorbar(1:3, meanImprovement, stdImprovement, 'k.', 'LineWidth', 1.5);
yline(0, 'r--', 'LineWidth', 1.5);
hold off;
set(gca, 'XTickLabel', {'Wiener', 'Improved Wiener', 'NLMS'});
ylabel('SNR Improvement (dB)');
title('Mean SNR Improvement');
grid on;

% Figure 2: Performance by Noise Type
figure('Name', 'Performance by Noise Type', 'Position', [100 100 1200 600]);

noiseTypeLabels = {};
wienerByNoise = [];
wienerImpByNoise = [];
nlmsByNoise = [];

for i = 1:length(uniqueNoises)
    noiseIdx = strcmp(results.noiseTypes, uniqueNoises{i}) & validIdx;
    noiseTypeLabels{i} = uniqueNoises{i};
    wienerByNoise(i) = mean(results.snr_improvement_wiener(noiseIdx));
    wienerImpByNoise(i) = mean(results.snr_improvement_wiener_improved(noiseIdx));
    nlmsByNoise(i) = mean(results.snr_improvement_nlms(noiseIdx));
end

x = 1:length(uniqueNoises);
bar(x, [wienerByNoise; wienerImpByNoise; nlmsByNoise]');
set(gca, 'XTickLabel', noiseTypeLabels);
xtickangle(45);
ylabel('Average SNR Improvement (dB)');
title('Performance by Noise Type');
legend('Wiener', 'Improved Wiener', 'NLMS', 'Location', 'best');
grid on;

% Figure 3: Performance by SNR Level
figure('Name', 'Performance by SNR Level', 'Position', [150 150 1000 600]);

wienerBySNR = [];
wienerImpBySNR = [];
nlmsBySNR = [];
snrLabels = {};

for i = 1:length(uniqueSNRs)
    snrIdx = strcmp(results.snrLevels, uniqueSNRs{i}) & validIdx; % Use strcmp for strings
    wienerBySNR(i) = mean(results.snr_improvement_wiener(snrIdx));
    wienerImpBySNR(i) = mean(results.snr_improvement_wiener_improved(snrIdx));
    nlmsBySNR(i) = mean(results.snr_improvement_nlms(snrIdx));
    snrLabels{i} = uniqueSNRs{i};
end

% Plot with string labels on x-axis
x = 1:length(uniqueSNRs);
plot(x, wienerBySNR, 'b-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Wiener');
hold on;
plot(x, wienerImpBySNR, 'g-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Improved Wiener');
plot(x, nlmsBySNR, 'm-d', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'NLMS');
hold off;
set(gca, 'XTick', x, 'XTickLabel', snrLabels);
xlabel('Input SNR Level');
ylabel('SNR Improvement (dB)');
title('Performance vs Input SNR Level');
legend('Location', 'best');
grid on;

fprintf('Visualizations complete!\n\n');

%% Save Results
fprintf('Saving batch processing results...\n');

outputDir = 'adaptive_filtering_results';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

save(fullfile(outputDir, 'batch_results.mat'), 'results');

% Create summary report
fid = fopen(fullfile(outputDir, 'summary_report.txt'), 'w');
fprintf(fid, '========================================\n');
fprintf(fid, 'Adaptive Filtering - Batch Results\n');
fprintf(fid, '========================================\n\n');
fprintf(fid, 'Number of samples processed: %d\n\n', numSamplesToProcess);
fprintf(fid, 'Average SNR Improvement (dB):\n');
fprintf(fid, '  Wiener Filter:          %.2f ± %.2f\n', ...
        mean(results.snr_improvement_wiener(validIdx)), std(results.snr_improvement_wiener(validIdx)));
fprintf(fid, '  Improved Wiener Filter: %.2f ± %.2f\n', ...
        mean(results.snr_improvement_wiener_improved(validIdx)), std(results.snr_improvement_wiener_improved(validIdx)));
fprintf(fid, '  NLMS Filter:            %.2f ± %.2f\n', ...
        mean(results.snr_improvement_nlms(validIdx)), std(results.snr_improvement_nlms(validIdx)));
fclose(fid);

fprintf('Results saved to: %s\n', outputDir);
fprintf('  - batch_results.mat\n');
fprintf('  - summary_report.txt\n\n');

fprintf('========================================\n');
fprintf('Batch Processing Complete!\n');
fprintf('========================================\n');