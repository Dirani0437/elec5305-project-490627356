% =========================================================================
% STAGE 1.2: NOIZEUS Dataset Preparation Script
% =========================================================================
% This script loads, organizes, and prepares the NOIZEUS dataset for
% speech enhancement experiments. It creates training and testing sets
% from clean and noisy audio files.
%
% Author: Alaa Aldirani
% Project: Real-Time Speech Enhancement
% =========================================================================

clear; close all; clc;

%% Configuration
% Define paths to your NOIZEUS dataset
cleanPath = 'C:\Users\diran\OneDrive - The University of Sydney (Students)\USYD\Year 3\Semester 2\Elec5305\Code\elec5305-project\noizeus_data\clean';
noisyPath = 'C:\Users\diran\OneDrive - The University of Sydney (Students)\USYD\Year 3\Semester 2\Elec5305\Code\elec5305-project\noizeus_data\noisy';

% Define noise types in the dataset
noiseTypes = {'airport', 'babble', 'car', 'exhibition', 'restaurant', ...
              'station', 'street', 'train'};

% Define SNR levels (excluding 15dB as mentioned)
snrLevels = {'0dB', '5dB', '10dB'};

% Train/Test split ratio (80% train, 20% test)
trainRatio = 0.8;

% Target sample rate (NOIZEUS is typically at 8000 Hz)
targetFs = 8000;

%% Step 1: Load Clean Audio Files
fprintf('========================================\n');
fprintf('NOIZEUS Dataset Preparation\n');
fprintf('========================================\n\n');

fprintf('Step 1: Loading clean audio files...\n');

% Get list of clean files
cleanFiles = dir(fullfile(cleanPath, '*.wav'));
numCleanFiles = length(cleanFiles);
fprintf('Found %d clean audio files\n', numCleanFiles);

% Load all clean signals
cleanData = struct('filename', {}, 'signal', {}, 'fs', {}, 'duration', {});

for i = 1:numCleanFiles
    filename = cleanFiles(i).name;
    filepath = fullfile(cleanPath, filename);
    
    % Read audio file
    [sig, fs] = audioread(filepath);
    
    % Store data
    cleanData(i).filename = filename;
    cleanData(i).signal = sig;
    cleanData(i).fs = fs;
    cleanData(i).duration = length(sig) / fs;
    
    if mod(i, 5) == 0
        fprintf('  Loaded %d/%d files\n', i, numCleanFiles);
    end
end

fprintf('Clean audio files loaded successfully!\n\n');

%% Step 2: Load Noisy Audio Files
fprintf('Step 2: Loading noisy audio files...\n');

% Initialize structure for noisy data
noisyData = struct('cleanFile', {}, 'noiseType', {}, 'snr', {}, ...
                   'signal', {}, 'fs', {}, 'filepath', {});

dataIdx = 1;

for noiseIdx = 1:length(noiseTypes)
    noiseType = noiseTypes{noiseIdx};
    % Base path for the noise type, e.g., .../noisy/airport
    noisySubPath = fullfile(noisyPath, noiseType);
    
    if ~exist(noisySubPath, 'dir')
        fprintf('  Warning: Noise type folder not found: %s\n', noisySubPath);
        continue;
    end
    
    for snrIdx = 1:length(snrLevels)
        snr = snrLevels{snrIdx}; % Gets the string, e.g., '0dB'
        
        % 1. Create path to intermediate SNR subfolder
        %    e.g., .../noisy/airport/airport_0dB
        snrFolderName = sprintf('%s_%s', noiseType, snr);
        intermediateFolderPath = fullfile(noisySubPath, snrFolderName);
        
        % 2. Check if this intermediate folder exists
        if ~exist(intermediateFolderPath, 'dir')
            fprintf('  Skipping missing SNR folder: %s\n', snrFolderName);
            continue;
        end

        % 3. Create the path to the FINAL subfolder (the extra level)
        %    e.g., .../noisy/airport/airport_0dB/0dB
        finalFolderPath = fullfile(intermediateFolderPath, snr);

        % 4. Check if this FINAL folder exists
        if ~exist(finalFolderPath, 'dir')
            fprintf('  Skipping missing final folder: %s\n', finalFolderPath);
            continue;
        end

        % 5. Get all .wav files from within that FINAL folder
        %    (Checking for both .wav and .WAV just in case)
        searchPattern_lower = '*.wav';
        searchPattern_upper = '*.WAV';
        
        noisyFiles_lower = dir(fullfile(finalFolderPath, searchPattern_lower));
        noisyFiles_upper = dir(fullfile(finalFolderPath, searchPattern_upper));
        
        noisyFiles = [noisyFiles_lower; noisyFiles_upper];
        
        % 6. Loop through all found .wav files
        for fileIdx = 1:length(noisyFiles)
            filename = noisyFiles(fileIdx).name;
            
            % 7. Build the FINAL filepath
            filepath = fullfile(finalFolderPath, filename);
            
            % Extract clean file reference (e.g., 'sp01' from 'sp01_airport_sn0.wav')
            tokens = regexp(filename, '^(sp\d+)_', 'tokens');
            if ~isempty(tokens)
                cleanFileBase = tokens{1}{1};
            else
                fprintf('  Warning: Could not parse filename: %s\n', filename);
                continue;
            end
            
            % Read noisy audio
            [sig, fs] = audioread(filepath);
            
            % Store data
            noisyData(dataIdx).cleanFile = [cleanFileBase '.wav'];
            noisyData(dataIdx).noiseType = noiseType;
            noisyData(dataIdx).snr = snr;
            noisyData(dataIdx).signal = sig;
            noisyData(dataIdx).fs = fs;
            noisyData(dataIdx).filepath = filepath;
            
            dataIdx = dataIdx + 1;
        end
    end
    
    fprintf('  Loaded noise type: %s\n', noiseType);
end

fprintf('Total noisy samples loaded: %d\n\n', length(noisyData));

%% Step 3: Create Training and Testing Sets
fprintf('Step 3: Creating training and testing splits...\n');

% Get unique clean file identifiers
uniqueCleanFiles = unique({noisyData.cleanFile});
numUniqueClean = length(uniqueCleanFiles);

% Shuffle and split
rng(42); % Set seed for reproducibility
shuffledIdx = randperm(numUniqueClean);
numTrain = round(trainRatio * numUniqueClean);

trainCleanFiles = uniqueCleanFiles(shuffledIdx(1:numTrain));
testCleanFiles = uniqueCleanFiles(shuffledIdx(numTrain+1:end));

% Assign data to train/test sets
trainData = struct('clean', {}, 'noisy', {}, 'noiseType', {}, 'snr', {});
testData = struct('clean', {}, 'noisy', {}, 'noiseType', {}, 'snr', {});

trainIdx = 1;
testIdx = 1;

for i = 1:length(noisyData)
    % Find corresponding clean signal
    cleanIdx = find(strcmp({cleanData.filename}, noisyData(i).cleanFile));
    
    if isempty(cleanIdx)
        continue;
    end
    
    % Check if this belongs to train or test set
    if any(strcmp(trainCleanFiles, noisyData(i).cleanFile))
        trainData(trainIdx).clean = cleanData(cleanIdx).signal;
        trainData(trainIdx).noisy = noisyData(i).signal;
        trainData(trainIdx).noiseType = noisyData(i).noiseType;
        trainData(trainIdx).snr = noisyData(i).snr;
        trainData(trainIdx).fs = noisyData(i).fs;
        trainData(trainIdx).cleanFile = noisyData(i).cleanFile;
        trainIdx = trainIdx + 1;
    else
        testData(testIdx).clean = cleanData(cleanIdx).signal;
        testData(testIdx).noisy = noisyData(i).signal;
        testData(testIdx).noiseType = noisyData(i).noiseType;
        testData(testIdx).snr = noisyData(i).snr;
        testData(testIdx).fs = noisyData(i).fs;
        testData(testIdx).cleanFile = noisyData(i).cleanFile;
        testIdx = testIdx + 1;
    end
end

fprintf('Training samples: %d\n', length(trainData));
fprintf('Testing samples: %d\n\n', length(testData));

%% Step 4: Verify Data Consistency
fprintf('Step 4: Verifying data consistency...\n');

% Check sample rates
allFs = [trainData.fs, testData.fs];
if length(unique(allFs)) > 1
    fprintf('  Warning: Multiple sample rates detected!\n');
else
    fprintf('  All files have sample rate: %d Hz\n', unique(allFs));
end

% Check signal lengths match between clean and noisy
mismatchCount = 0;
for i = 1:length(trainData)
    if length(trainData(i).clean) ~= length(trainData(i).noisy)
        mismatchCount = mismatchCount + 1;
    end
end

if mismatchCount > 0
    fprintf('  Warning: %d training samples have length mismatch!\n', mismatchCount);
else
    fprintf('  All training samples have matching lengths\n');
end

fprintf('Data preparation complete!\n\n');

%% Step 5: Save Prepared Dataset
fprintf('Step 5: Saving prepared dataset...\n');

% Create output directory if it doesn't exist
outputDir = 'prepared_data';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Save the data
save(fullfile(outputDir, 'noizeus_prepared.mat'), 'trainData', 'testData', ...
     'noiseTypes', 'snrLevels', 'trainCleanFiles', 'testCleanFiles', '-v7.3');

fprintf('Dataset saved to: %s\n', fullfile(outputDir, 'noizeus_prepared.mat'));

%% Step 6: Display Dataset Statistics
fprintf('\n========================================\n');
fprintf('Dataset Statistics\n');
fprintf('========================================\n');
fprintf('Total clean files: %d\n', numCleanFiles);
fprintf('Training clean files: %d\n', length(trainCleanFiles));
fprintf('Testing clean files: %d\n', length(testCleanFiles));
fprintf('Training samples: %d\n', length(trainData));
fprintf('Testing samples: %d\n', length(testData));
fprintf('Noise types: %d\n', length(noiseTypes));
fprintf('SNR levels: %d\n', length(snrLevels));
fprintf('Sample rate: %d Hz\n', targetFs);
fprintf('\nNoise types included:\n');
for i = 1:length(noiseTypes)
    fprintf('  - %s\n', noiseTypes{i});
end
fprintf('\nSNR levels included:\n');
for i = 1:length(snrLevels)
    fprintf('  - %s\n', snrLevels{i});
end
fprintf('========================================\n');

%% Step 7: Visualize Sample Data
fprintf('\nGenerating visualization of sample data...\n');

% Select a random training sample
sampleIdx = randi(length(trainData));
sample = trainData(sampleIdx);

figure('Name', 'Sample Data Visualization', 'Position', [100 100 1200 800]);

% Plot clean signal
subplot(3,2,1);
t = (0:length(sample.clean)-1) / sample.fs;
plot(t, sample.clean);
title(sprintf('Clean Signal: %s', sample.cleanFile), 'Interpreter', 'none');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

% Plot noisy signal
subplot(3,2,2);
plot(t, sample.noisy);
title(sprintf('Noisy Signal: %s @ %ddB SNR', sample.noiseType, sample.snr));
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

% Plot spectrograms
subplot(3,2,3);
spectrogram(sample.clean, hamming(256), 128, 512, sample.fs, 'yaxis');
title('Clean Spectrogram');
colorbar;

subplot(3,2,4);
spectrogram(sample.noisy, hamming(256), 128, 512, sample.fs, 'yaxis');
title('Noisy Spectrogram');
colorbar;

% Plot power spectral density
subplot(3,2,5);
[pxx_clean, f_clean] = pwelch(sample.clean, hamming(512), 256, 1024, sample.fs);
plot(f_clean, 10*log10(pxx_clean));
title('Clean PSD');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
grid on;

subplot(3,2,6);
[pxx_noisy, f_noisy] = pwelch(sample.noisy, hamming(512), 256, 1024, sample.fs);
plot(f_noisy, 10*log10(pxx_noisy));
title('Noisy PSD');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
grid on;

fprintf('Visualization complete!\n\n');
fprintf('Data preparation finished successfully!\n');
fprintf('You can now proceed to Stage 2: Adaptive Filtering Implementation\n');