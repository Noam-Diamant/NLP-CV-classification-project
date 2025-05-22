# NLP & CV Classification Project: Alcoholic Beverage Text Detection

## 1. Overview

This project simulates a real-world classification problem: given a set of images, the goal is to build a classifier that detects images containing superimposed text related to alcoholic beverages. The pipeline involves data exploration, analysis, and the construction of a robust classification system that achieves strong performance metrics.

### Task Objectives
- Explore the dataset and understand the inputs/outputs.
- Design and explain a classification pipeline architecture.
- Suggest and use appropriate metrics for performance assessment.
- Implement the pipeline using the provided data.
- Report results and share valuable findings.

### Dataset
- A set of images containing text, split into two folders: `Alcohol` and `Non-alcohol`, based on whether the text mentions alcohol.

## 2. How to Run

### Prerequisites
- Python 3.11
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Running the Classifier
1. Place your images in a folder (e.g., `Images/`).
2. Run the main script to classify images:
   ```bash
   python run.py Images
   ```
3. The script will output a `results.csv` file with two columns: `file_name` and `prediction` (either `alcohol` or `nonalcohol`).

### Example Output
```
file_name,prediction
image1.jpg,alcohol
image2.jpg,nonalcohol
```

## 3. Main Analysis

### Data Exploration and Problem Identification
- The dataset consists of images with superimposed text, divided into `Alcohol` and `Non-alcohol` categories.
- Example: An image with the text "Don't miss limited edition nightlife bourbon event" is labeled as `alcohol`, while "energy drink" is labeled as `nonalcohol`.
- The initial data exploration revealed several challenging issues that needed to be addressed:
  - Inverted/flipped images with upside-down or mirrored text
  - Poor text-background contrast where text color was too similar to background
  - Text visibility issues with blurred or unclear text
  - Word splitting/merging errors (e.g., "wat er" instead of "water")
  - Character recognition errors with numbers appearing instead of letters
  - Irrelevant text content that could confuse classification

### Image Processing Solutions
To address image-level challenges, a comprehensive preprocessing pipeline was developed:
- Image orientation detection: For each image, both original and flipped versions were processed, with selection based on OCR confidence scores
- Color space separation: Images were converted to RGB and YUV color spaces to improve text visibility in cases of poor contrast
- Multi-variant processing: Each image was processed through multiple combinations (original/flipped × RGB/YUV color spaces) with the highest confidence result selected

### Text Processing and Cleaning
Extracted text underwent several cleaning steps:

- Case normalization: All text converted to lowercase
- Character mapping: Numbers mapped to likely letter equivalents (e.g., '1' → 'i' or 'l')
- Punctuation and whitespace cleanup: Removed irrelevant characters and normalized spacing
- Noise reduction: Filtered out non-relevant text content

### Classification Approach
A hybrid classification system was implemented:

- Keyword-based classifier: Used curated lists of alcohol-related and non-alcohol-related terms with fuzzy string matching
- Neural network classifier: Fine-tuned DistilBERT model for text classification
- Hybrid logic: Combined both approaches, using keyword matching for clear cases and falling back to the neural network for ambiguous cases

### Performance and Results
The final pipeline showed significant improvements across all metrics:

- OCR quality: Substantial reduction in images returning "no_text" results
- Average confidence scores: Marked improvement in OCR confidence across all images
- Classification metrics: Strong performance on precision, recall, F1-score, and accuracy
- Robustness: Effective handling of various image quality issues and text extraction challenges

The majority of development time was invested in image processing and text extraction optimization, which proved critical for achieving good classification results.

#### Notes
- The pipeline is CPU-only and does not require a GPU.
