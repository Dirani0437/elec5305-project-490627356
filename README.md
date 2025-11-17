================================================================================
                    REAL-TIME SPEECH ENHANCEMENT PROJECT
                           ELEC5305 Course Project
                              Author: Alaa Aldirani
================================================================================

PROJECT OVERVIEW
--------------------------------------------------------------------------------
This project implements and compares multiple speech enhancement techniques for
noise reduction in speech signals. The system evaluates traditional Digital
Signal Processing (DSP) methods against a Convolutional Neural Network (CNN)
based machine learning approach.

Enhancement Methods Implemented:
- Standard Wiener Filtering
- Improved Wiener Filtering (with over-subtraction)
- Normalized Least Mean Squares (NLMS) Adaptive Filtering
- CNN-Based Speech Enhancement

================================================================================

DATASET
--------------------------------------------------------------------------------
Dataset: NOIZEUS Speech Corpus

The NOIZEUS database is a standard benchmark for speech enhancement algorithms.
It contains clean speech signals corrupted by different types of noise at 
various Signal-to-Noise Ratio (SNR) levels.

Dataset Characteristics:
- Sample Rate: 8000 Hz
- Noise Types: 8 types
  * airport
  * babble
  * car
  * exhibition
  * restaurant
  * station
  * street
  * train

- SNR Levels: 0dB, 5dB, 10dB (15dB excluded)

Data Organization:
- Training/Testing Split: 80% / 20%
- Clean speech files located in: noizeus_data/clean/
- Noisy speech files located in: noizeus_data/noisy/[noise_type]/[noise_type_SNR]/[SNR]/

The dataset is organized by speaker (sp01, sp02, etc.), with each clean 
utterance corrupted by all noise types at all SNR levels.

================================================================================

REQUIREMENTS
--------------------------------------------------------------------------------
Software:
- MATLAB R2019b or later (recommended)
- Deep Learning Toolbox (for CNN implementation)
- Signal Processing Toolbox

Hardware:
- Minimum 8GB RAM recommended
- GPU support optional but beneficial for CNN training

================================================================================

PROJECT STRUCTURE
--------------------------------------------------------------------------------

Main Processing Scripts:
-------------------------
1. Stage1_DataPreparation.m
   - Loads and organizes the NOIZEUS dataset
   - Creates training/testing splits (80/20)
   - Verifies data consistency
   - Saves prepared data to 'prepared_data/noizeus_prepared.mat'

2. Stage2_AdaptiveFiltering.m
   - Implements Voice Activity Detection (VAD)
   - STFT-based spectral processing
   - Standard Wiener filtering
   - Improved Wiener filtering with over-subtraction
   - NLMS adaptive filtering
   - Generates comprehensive visualizations

3. Stage3_MachineLearning.m
   - Designs CNN architecture for speech enhancement
   - Prepares training data (spectrogram pairs)
   - Computes Ideal Ratio Masks (IRM)
   - Trains CNN to predict enhancement masks
   - Performs inference on test samples

4. Stage4_ComparativeAnalysis.m
   - Comprehensive evaluation of all methods
   - Computes SNR and STOI metrics
   - Performance analysis by noise type and SNR level
   - Generates comparison visualizations

5. BatchProcessing.m
   - Processes multiple test samples (default: 50)
   - Computes average performance statistics
   - Analyzes performance across noise types and SNR levels
   - Saves results and summary reports

Supporting Files:
-----------------
- HelperFunctions.m: Utility functions for signal processing
- Project_Pipeline.docx: Overall project workflow documentation
- Project_description.pdf: Detailed project specifications
- Elec5305_Project_Proposal.pdf: Initial project proposal

Output Directories (created during execution):
----------------------------------------------
- prepared_data/: Contains processed dataset (.mat files)
- adaptive_filtering_results/: Wiener and NLMS filter results
- ml_results/: CNN training results and enhanced signals
- comparative_results/: Final comparison metrics and reports

================================================================================

HOW TO RUN THE CODE
--------------------------------------------------------------------------------

STEP 1: Dataset Setup
----------------------
1. Download the NOIZEUS dataset
2. Organize the data into the following structure:
   
   noizeus_data/
   ├── clean/
   │   ├── sp01.wav
   │   ├── sp02.wav
   │   └── ...
   └── noisy/
       ├── airport/
       │   ├── airport_0dB/
       │   │   └── 0dB/
       │   │       ├── sp01_airport_sn0.wav
       │   │       └── ...
       │   ├── airport_5dB/
       │   └── airport_10dB/
       ├── babble/
       └── ...

3. Update paths in Stage1_DataPreparation.m (lines 16-17):
   - cleanPath: Path to clean audio files
   - noisyPath: Path to noisy audio folder

STEP 2: Data Preparation (Stage 1)
-----------------------------------
Run: Stage1_DataPreparation.m

This script will:
- Load all clean and noisy audio files
- Create train/test splits
- Verify data consistency
- Save to 'prepared_data/noizeus_prepared.mat'

Expected Output:
- Dataset statistics printed to console
- Visualization of sample data
- Prepared dataset saved (~50-100MB)

STEP 3: Adaptive Filtering (Stage 2)
-------------------------------------
Run: Stage2_AdaptiveFiltering.m

This script will:
- Load prepared dataset
- Apply VAD to identify speech/noise regions
- Implement standard and improved Wiener filters
- Implement NLMS adaptive filter
- Generate comparison visualizations

Expected Output:
- VAD statistics
- SNR improvement metrics
- Multiple visualization figures
- Enhanced audio signals

STEP 4: Machine Learning (Stage 3)
-----------------------------------
Run: Stage3_MachineLearning.m

This script will:
- Prepare spectrograms for CNN training
- Train CNN model (may take 15-30 minutes)
- Evaluate on test samples
- Save trained model

Expected Output:
- Training progress information
- Model performance metrics
- Enhanced audio using CNN
- Trained model saved in 'ml_results/'

STEP 5: Comparative Analysis (Stage 4)
---------------------------------------
Run: Stage4_ComparativeAnalysis.m

This script will:
- Evaluate all methods on multiple test samples
- Compute SNR and STOI metrics
- Analyze by noise type and SNR level
- Generate comprehensive comparison plots

Expected Output:
- Detailed performance tables
- evaluation_report.txt with key findings
- Multiple comparison figures

STEP 6: Batch Processing (Optional)
------------------------------------
Run: BatchProcessing.m

Configure:
- numSamplesToProcess: Number of samples (default: 50, set to -1 for all)

This script will:
- Process specified number of test samples
- Compute average performance statistics
- Generate performance visualizations
- Save summary report

================================================================================

SIGNAL PROCESSING PARAMETERS
--------------------------------------------------------------------------------
Short-Time Fourier Transform (STFT):
- Window Length: 32 ms (256 samples at 8kHz)
- Hop Size: 16 ms (128 samples, 50% overlap)
- FFT Size: 256 (next power of 2)
- Window Function: Hamming (periodic)

Voice Activity Detection (VAD):
- Method: Energy-based threshold
- Threshold: Mean energy - 5 dB
- Smoothing: Median filter (length 5)

Wiener Filter Parameters:
- Standard: Basic spectral subtraction
- Improved: Over-subtraction factor α = 2.0, spectral floor β = 0.01

NLMS Filter Parameters:
- Filter Order: 32
- Step Size (μ): 0.1
- Regularization (δ): 0.01

CNN Architecture:
- Input: Log-magnitude spectrograms
- Target: Ideal Ratio Masks (IRM)
- Network: Convolutional layers with batch normalization
- Output: Enhanced magnitude spectrogram

================================================================================

EVALUATION METRICS
--------------------------------------------------------------------------------
1. Signal-to-Noise Ratio (SNR):
   - Measures noise reduction effectiveness
   - Higher values indicate better enhancement
   - Computed as: 10*log10(signal_power / noise_power)

2. Short-Time Objective Intelligibility (STOI):
   - Measures speech intelligibility preservation
   - Range: 0 to 1 (higher is better)
   - Correlates well with human listening tests

3. SNR Improvement:
   - Difference between enhanced and noisy SNR
   - Positive values indicate improvement

================================================================================

RESULTS SUMMARY
--------------------------------------------------------------------------------
Based on evaluation of 50 test samples:

Average SNR Improvement (dB):
- Standard Wiener:   2.09 dB
- Improved Wiener:   2.96 dB
- NLMS:             -3.84 dB (performance degradation)
- CNN:               5.54 dB (best performance)

Average STOI Score:
- Noisy:             0.506
- Standard Wiener:   0.517 (+0.011)
- Improved Wiener:   0.515 (+0.009)
- NLMS:              0.322 (-0.184)
- CNN:               0.570 (+0.064) (best performance)

Key Findings:
- CNN-based method achieves highest SNR improvement and STOI scores
- Improved Wiener filter outperforms standard Wiener by ~0.9 dB
- NLMS filter shows negative performance (not suitable for this task)
- Most challenging noise types: airport and babble

================================================================================

OUTPUT FILES
--------------------------------------------------------------------------------
After running all stages, the following files will be generated:

Data Files:
- prepared_data/noizeus_prepared.mat: Processed dataset
- adaptive_filtering_results/batch_results.mat: Batch processing results
- ml_results/trained_cnn.mat: Trained CNN model

Reports:
- summary_report.txt: Overall performance summary
- evaluation_report.txt: Comprehensive evaluation results

Visualizations:
- time_domain_comparison.png: Waveform comparisons
- psd_comparison.png: Power spectral density analysis
- wiener_enhancement_visualization.png: Wiener filter results
- standard_wiener_enhancement_visualization.png: Standard Wiener results
- wiener_gain_comparison.png: Wiener gain function analysis
- spectrogram_comparison.png: Time-frequency representations
- cnn_enhancement_visualization.png: CNN enhancement results
- performance_by_noise_type.png: Performance across noise types
- overall_performance_comparison.png: Summary comparison chart
- performance_by_snr_level.png: Performance vs input SNR

================================================================================

TROUBLESHOOTING
--------------------------------------------------------------------------------
Common Issues:

1. "File not found" errors:
   - Verify dataset paths in Stage1_DataPreparation.m
   - Ensure NOIZEUS data is properly organized
   - Check folder structure matches expected format

2. Out of memory errors:
   - Reduce numTrainSamples in Stage3_MachineLearning.m
   - Reduce numSamplesToProcess in BatchProcessing.m
   - Close other MATLAB figures: close all;

3. Slow CNN training:
   - Reduce number of training samples
   - Decrease number of epochs
   - Use GPU if available (requires Parallel Computing Toolbox)

4. SNR calculation failures:
   - Usually due to signal length mismatches
   - Check that clean and noisy signals have same length
   - NaN values will be excluded from statistics

5. Missing toolboxes:
   - Verify Signal Processing Toolbox is installed
   - Verify Deep Learning Toolbox for Stage 3
   - Check MATLAB license for required features

================================================================================

FUTURE IMPROVEMENTS
--------------------------------------------------------------------------------
1. Implement recurrent neural networks (LSTM/GRU) for temporal modeling
2. Add PESQ (Perceptual Evaluation of Speech Quality) metric
3. Implement real-time processing capability
4. Test with additional noise types and SNR levels
5. Explore transfer learning with pre-trained models
6. Implement spectral masking with phase reconstruction

================================================================================

REFERENCES
--------------------------------------------------------------------------------
1. NOIZEUS Speech Corpus: https://ecs.utdallas.edu/loizou/speech/noizeus/
2. Loizou, P. C. (2013). Speech Enhancement: Theory and Practice, 2nd Ed.
3. STOI Metric: Taal et al., "A Short-Time Objective Intelligibility Measure 
   for Time-Frequency Weighted Noisy Speech," ICASSP 2010.
4. MATLAB Signal Processing Toolbox Documentation
5. MATLAB Deep Learning Toolbox Documentation

================================================================================

CONTACT
--------------------------------------------------------------------------------
Author: Alaa Aldirani
Course: ELEC5305 - Biomedical Signal Processing
Institution: The University of Sydney

================================================================================

LICENSE
--------------------------------------------------------------------------------
This project is for educational and research purposes as part of coursework.
The NOIZEUS dataset is property of UTDallas and should be used according to 
their terms.

================================================================================
