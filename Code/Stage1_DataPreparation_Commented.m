% =========================================================================
% STAGE 1.2: NOIZEUS Dataset Preparation Script
% =========================================================================
% This script loads, organizes, and prepares the NOIZEUS dataset for
% speech enhancement experiments. It creates training and testing sets
% from clean and noisy audio files.
%
% The NOIZEUS database is a standard benchmark for speech enhancement 
% containing speech signals corrupted with various types of real-world
% noises at different SNR levels.
%
% Key Operations:
%   - Load clean reference speech files
%   - Load noisy versions with various noise types and SNR levels
%   - Split data into training (80%) and testing (20%) sets
%   - Verify data consistency and save prepared dataset
%
% Output:
%   - prepared_data/noizeus_prepared.mat containing train/test splits
%
% Author: Alaa Aldirani
% Project: Real-Time Speech Enhancement
% =========================================================================

clear; close all; clc;

%% Configuration
% =========================================================================
% Define paths to your NOIZEUS dataset
% UPDATE THESE PATHS to match your local file system structure
% =========================================================================
cleanPath = 'C:\Users\diran\OneDrive - The University of Sydney (Students)\USYD\Year 3\Semester 2\Elec5305\Code\elec5305-project\noizeus_data\clean';
noisyPath = 'C:\Users\diran\OneDrive - The University of Sydney (Students)\USYD\Year 3\Semester 2\Elec5305\Code\elec5305-project\noizeus_data\noisy';

% Define noise types in the dataset
% These correspond to different acoustic environments recorded in real-world settings
% Each noise type has distinct spectral and temporal characteristics:
%   - airport: Broadband noise with announcements and crowd chatter
%   - babble: Multiple overlapping speakers (cocktail party noise)
%   - car: Engine noise and road noise (relatively stationary)
%   - exhibition: Large venue ambient noise
%   - restaurant: Similar to babble but with additional environmental sounds
%   - station: Train/bus station announcements and crowd noise
%   - street: Traffic and pedestrian sounds
%   - train: Interior train noise (engine + track noise)
noiseTypes = {'airport', 'babble', 'car', 'exhibition', 'restaurant', ...
              'station', 'street', 'train'};

% Define SNR levels (excluding 15dB as mentioned in project requirements)
% SNR = Signal-to-Noise Ratio in decibels
% Lower SNR means more noise relative to speech:
%   - 0dB: Signal and noise have equal power (very challenging)
%   - 5dB: Signal is ~3x stronger than noise
%   - 10dB: Signal is ~10x stronger than noise
snrLevels = {'0dB', '5dB', '10dB'};

% Train/Test split ratio (80% train, 20% test)
% This split ensures enough data for training while reserving unseen data for testing
% Common practice in machine learning to evaluate generalization performance
trainRatio = 0.8;

% Target sample rate (NOIZEUS is typically at 8000 Hz)
% 8kHz is standard for telephone-bandwidth speech (narrowband)
% Sufficient for capturing speech information up to 4kHz (Nyquist frequency)
targetFs = 8000;

%% Step 1: Load Clean Audio Files
% =========================================================================
% Clean audio files serve as ground truth references for:
%   1. Creating training targets (what we want to recover)
%   2. Computing objective metrics (SNR, STOI, etc.)
%   3. Evaluating enhancement algorithm performance
% =========================================================================
fprintf('========================================\n');
fprintf('NOIZEUS Dataset Preparation\n');
fprintf('========================================\n\n');

fprintf('Step 1: Loading clean audio files...\n');

% Get list of clean files using wildcard pattern matching
% dir() returns file information including name, date, bytes, etc.
cleanFiles = dir(fullfile(cleanPath, '*.wav'));
numCleanFiles = length(cleanFiles);
fprintf('Found %d clean audio files\n', numCleanFiles);

% Load all clean signals into a structured array
% Using struct for organized storage of metadata with each signal
cleanData = struct('filename', {}, 'signal', {}, 'fs', {}, 'duration', {});

for i = 1:numCleanFiles
    filename = cleanFiles(i).name;
    filepath = fullfile(cleanPath, filename);
    
    % Read audio file using built-in MATLAB function
    % audioread returns: signal (column vector) and sample rate
    [sig, fs] = audioread(filepath);
    
    % Store data in structured format for easy access later
    cleanData(i).filename = filename;       % Original filename for reference
    cleanData(i).signal = sig;              % Actual audio samples
    cleanData(i).fs = fs;                   % Sample rate (should be 8000 Hz)
    cleanData(i).duration = length(sig) / fs; % Duration in seconds
    
    % Progress indicator every 5 files
    if mod(i, 5) == 0
        fprintf('  Loaded %d/%d files\n', i, numCleanFiles);
    end
end

fprintf('Clean audio files loaded successfully!\n\n');

%% Step 2: Load Noisy Audio Files
% =========================================================================
% The noisy audio files are organized in a hierarchical folder structure:
%   noisy/
%     ├── airport/
%     │   ├── airport_0dB/
%     │   │   └── 0dB/
%     │   │       └── sp01_airport_sn0.wav
%     │   ├── airport_5dB/
%     │   └── airport_10dB/
%     ├── babble/
%     ... (similar structure for each noise type)
%
% Each noisy file corresponds to a clean file with added noise at specific SNR
% =========================================================================
fprintf('Step 2: Loading noisy audio files...\n');

% Initialize structure for noisy data
% This will hold all combinations of: clean file × noise type × SNR level
noisyData = struct('cleanFile', {}, 'noiseType', {}, 'snr', {}, ...
                   'signal', {}, 'fs', {}, 'filepath', {});

dataIdx = 1; % Counter for total noisy samples

% Iterate through all noise types
for noiseIdx = 1:length(noiseTypes)
    noiseType = noiseTypes{noiseIdx};
    % Base path for the noise type, e.g., .../noisy/airport
    noisySubPath = fullfile(noisyPath, noiseType);
    
    % Check if this noise type folder exists
    if ~exist(noisySubPath, 'dir')
        fprintf('  Warning: Noise type folder not found: %s\n', noisySubPath);
        continue; % Skip to next noise type
    end
    
    % Iterate through all SNR levels for this noise type
    for snrIdx = 1:length(snrLevels)
        snr = snrLevels{snrIdx}; % Gets the string, e.g., '0dB'
        
        % 1. Create path to intermediate SNR subfolder
        %    e.g., .../noisy/airport/airport_0dB
        %    This folder naming convention combines noise type and SNR
        snrFolderName = sprintf('%s_%s', noiseType, snr);
        intermediateFolderPath = fullfile(noisySubPath, snrFolderName);
        
        % 2. Check if this intermediate folder exists
        if ~exist(intermediateFolderPath, 'dir')
            fprintf('  Skipping missing SNR folder: %s\n', snrFolderName);
            continue;
        end

        % 3. Create the path to the FINAL subfolder (the extra nesting level)
        %    e.g., .../noisy/airport/airport_0dB/0dB
        %    This is the actual location of the audio files
        finalFolderPath = fullfile(intermediateFolderPath, snr);

        % 4. Check if this FINAL folder exists
        if ~exist(finalFolderPath, 'dir')
            fprintf('  Skipping missing final folder: %s\n', finalFolderPath);
            continue;
        end

        % 5. Get all .wav files from within that FINAL folder
        %    Search for both lowercase and uppercase extensions for compatibility
        searchPattern_lower = '*.wav';
        searchPattern_upper = '*.WAV';
        
        noisyFiles_lower = dir(fullfile(finalFolderPath, searchPattern_lower));
        noisyFiles_upper = dir(fullfile(finalFolderPath, searchPattern_upper));
        
        % Combine results from both search patterns
        noisyFiles = [noisyFiles_lower; noisyFiles_upper];
        
        % 6. Loop through all found .wav files in this folder
        for fileIdx = 1:length(noisyFiles)
            filename = noisyFiles(fileIdx).name;
            
            % 7. Build the complete filepath to the noisy audio file
            filepath = fullfile(finalFolderPath, filename);
            
            % Extract clean file reference from noisy filename
            % Noisy files are named like: 'sp01_airport_sn0.wav'
            % where 'sp01' identifies the original clean speech file
            % Regular expression extracts the speaker ID (sp##)
            tokens = regexp(filename, '^(sp\d+)_', 'tokens');
            if ~isempty(tokens)
                cleanFileBase = tokens{1}{1}; % Extract 'sp01', 'sp02', etc.
            else
                fprintf('  Warning: Could not parse filename: %s\n', filename);
                continue; % Skip files that don't match expected naming pattern
            end
            
            % Read noisy audio file
            [sig, fs] = audioread(filepath);
            
            % Store data with all relevant metadata
            noisyData(dataIdx).cleanFile = [cleanFileBase '.wav']; % Link to clean version
            noisyData(dataIdx).noiseType = noiseType;              % Type of noise added
            noisyData(dataIdx).snr = snr;                          % SNR level
            noisyData(dataIdx).signal = sig;                       % Noisy audio samples
            noisyData(dataIdx).fs = fs;                            % Sample rate
            noisyData(dataIdx).filepath = filepath;                % Full path for reference
            
            dataIdx = dataIdx + 1;
        end
    end
    
    fprintf('  Loaded noise type: %s\n', noiseType);
end

fprintf('Total noisy samples loaded: %d\n\n', length(noisyData));

%% Step 3: Create Training and Testing Sets
% =========================================================================
% Split data by UNIQUE SPEAKERS to ensure:
%   1. No speaker appears in both train and test sets
%   2. Fair evaluation on unseen speakers (generalization test)
%   3. Avoiding data leakage between train and test
%
% This is crucial because if the same speaker appears in both sets,
% the model might learn speaker-specific features rather than general
% speech enhancement principles.
% =========================================================================
fprintf('Step 3: Creating training and testing splits...\n');

% Get unique clean file identifiers (each file = different utterance/speaker)
uniqueCleanFiles = unique({noisyData.cleanFile});
numUniqueClean = length(uniqueCleanFiles);

% Shuffle and split based on unique speakers
% Using fixed random seed (42) for reproducibility across runs
rng(42); % Set seed for reproducibility - always gives same "random" order
shuffledIdx = randperm(numUniqueClean); % Random permutation of indices
numTrain = round(trainRatio * numUniqueClean); % Number of speakers for training

% Assign speakers to train or test sets
trainCleanFiles = uniqueCleanFiles(shuffledIdx(1:numTrain));           % First 80%
testCleanFiles = uniqueCleanFiles(shuffledIdx(numTrain+1:end));       % Remaining 20%

% Initialize data structures for train/test splits
% Each entry will contain paired clean/noisy signals plus metadata
trainData = struct('clean', {}, 'noisy', {}, 'noiseType', {}, 'snr', {});
testData = struct('clean', {}, 'noisy', {}, 'noiseType', {}, 'snr', {});

trainIdx = 1;
testIdx = 1;

% Iterate through all noisy samples and assign to appropriate set
for i = 1:length(noisyData)
    % Find corresponding clean signal by matching filename
    cleanIdx = find(strcmp({cleanData.filename}, noisyData(i).cleanFile));
    
    if isempty(cleanIdx)
        continue; % Skip if no matching clean file found
    end
    
    % Check if this speaker belongs to train or test set
    if any(strcmp(trainCleanFiles, noisyData(i).cleanFile))
        % This speaker is in the training set
        trainData(trainIdx).clean = cleanData(cleanIdx).signal;
        trainData(trainIdx).noisy = noisyData(i).signal;
        trainData(trainIdx).noiseType = noisyData(i).noiseType;
        trainData(trainIdx).snr = noisyData(i).snr;
        trainData(trainIdx).fs = noisyData(i).fs;
        trainData(trainIdx).cleanFile = noisyData(i).cleanFile;
        trainIdx = trainIdx + 1;
    else
        % This speaker is in the testing set
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
% =========================================================================
% Important sanity checks to ensure data quality:
%   1. All files should have the same sample rate
%   2. Clean and noisy pairs should have matching lengths
%   3. No corrupted or missing data
%
% These checks prevent runtime errors in later processing stages
% =========================================================================
fprintf('Step 4: Verifying data consistency...\n');

% Check sample rates - all should be identical (8000 Hz for NOIZEUS)
allFs = [trainData.fs, testData.fs];
if length(unique(allFs)) > 1
    fprintf('  Warning: Multiple sample rates detected!\n');
    fprintf('  Found rates: %s Hz\n', num2str(unique(allFs)));
else
    fprintf('  All files have sample rate: %d Hz\n', unique(allFs));
end

% Check signal lengths match between clean and noisy pairs
% Length mismatch could cause errors in SNR computation and model training
mismatchCount = 0;
for i = 1:length(trainData)
    if length(trainData(i).clean) ~= length(trainData(i).noisy)
        mismatchCount = mismatchCount + 1;
    end
end

if mismatchCount > 0
    fprintf('  Warning: %d training samples have length mismatch!\n', mismatchCount);
    fprintf('  This may cause issues during processing.\n');
else
    fprintf('  All training samples have matching lengths\n');
end

fprintf('Data preparation complete!\n\n');

%% Step 5: Save Prepared Dataset
% =========================================================================
% Save the processed data to a .mat file for use in subsequent stages
% Using -v7.3 flag allows saving large files (>2GB) using HDF5 format
%
% The saved file contains:
%   - trainData: Training set with clean/noisy pairs
%   - testData: Test set with clean/noisy pairs  
%   - noiseTypes: List of noise types in dataset
%   - snrLevels: List of SNR levels
%   - trainCleanFiles: Which speakers are in training set
%   - testCleanFiles: Which speakers are in test set
% =========================================================================
fprintf('Step 5: Saving prepared dataset...\n');

% Create output directory if it doesn't exist
outputDir = 'prepared_data';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Save the data with all relevant variables
% -v7.3 format supports large arrays and compression
save(fullfile(outputDir, 'noizeus_prepared.mat'), 'trainData', 'testData', ...
     'noiseTypes', 'snrLevels', 'trainCleanFiles', 'testCleanFiles', '-v7.3');

fprintf('Dataset saved to: %s\n', fullfile(outputDir, 'noizeus_prepared.mat'));

%% Step 6: Display Dataset Statistics
% =========================================================================
% Summary statistics for understanding the dataset composition
% Useful for ensuring balanced representation across noise types and SNRs
% =========================================================================
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
% =========================================================================
% Visual inspection of sample data helps verify:
%   1. Data loading was successful
%   2. Clean/noisy signals are properly aligned
%   3. Noise is visibly present in spectrograms
%   4. Frequency content is as expected
%
% This creates a comprehensive view with:
%   - Time-domain waveforms
%   - Spectrograms (time-frequency representation)
%   - Power Spectral Density (frequency content)
% =========================================================================
fprintf('\nGenerating visualization of sample data...\n');

% Select a random training sample for visualization
sampleIdx = randi(length(trainData));
sample = trainData(sampleIdx);

figure('Name', 'Sample Data Visualization', 'Position', [100 100 1200 800]);

% Plot clean signal in time domain
% Time domain shows amplitude variations but doesn't reveal frequency info
subplot(3,2,1);
t = (0:length(sample.clean)-1) / sample.fs; % Create time vector in seconds
plot(t, sample.clean);
title(sprintf('Clean Signal: %s', sample.cleanFile), 'Interpreter', 'none');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

% Plot noisy signal in time domain
% Compare with clean to see how noise affects the waveform
subplot(3,2,2);
plot(t, sample.noisy);
title(sprintf('Noisy Signal: %s @ %ddB SNR', sample.noiseType, sample.snr));
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

% Plot spectrograms - time-frequency representations
% Spectrograms show how frequency content changes over time
% Bright regions indicate high energy; dark regions indicate low energy
% Speech shows formant structures (horizontal bands)
% Noise appears as uniform energy across frequencies

subplot(3,2,3);
% spectrogram(signal, window, overlap, fft_size, sample_rate, 'yaxis')
% Window: 256 samples (~32ms at 8kHz) for good time-frequency resolution
% Overlap: 128 samples (50%) for smooth temporal transitions
% FFT size: 512 points for frequency resolution
spectrogram(sample.clean, hamming(256), 128, 512, sample.fs, 'yaxis');
title('Clean Spectrogram');
colorbar; % Color scale shows power in dB

subplot(3,2,4);
spectrogram(sample.noisy, hamming(256), 128, 512, sample.fs, 'yaxis');
title('Noisy Spectrogram');
colorbar;

% Plot power spectral density (PSD) - average frequency content
% PSD shows which frequencies contain the most energy on average
% Useful for understanding noise characteristics (flat vs. colored noise)
subplot(3,2,5);
% pwelch: Welch's method for PSD estimation (reduces variance)
% hamming(512): 512-sample window with Hamming taper
% 256: 50% overlap between segments
% 1024: FFT size for frequency resolution
[pxx_clean, f_clean] = pwelch(sample.clean, hamming(512), 256, 1024, sample.fs);
plot(f_clean, 10*log10(pxx_clean)); % Convert to dB scale
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
