# NLP & CV Classification Project: Alcoholic Beverage Text Detection

## 1. Overview

This project simulates a real-world classification problem: given a set of images, the goal is to build a classifier that detects images containing superimposed text related to alcoholic beverages. The pipeline involves data exploration, analysis, and the construction of a robust classification system that achieves strong performance metrics. The classifier's performance may be evaluated on an unseen test set.

### Background
DoubleVerify's Universal Content Intelligence is an online classification engine that powers expansive content categorization, analyzing visual, audio, text, and other content types across all channels, devices, and formats. This project focuses on building a component for text extraction (OCR) and text-based classification for alcohol references in images.

### Task Objectives
- Explore the dataset and understand the inputs/outputs.
- Design and explain a classification pipeline architecture.
- Suggest and use appropriate metrics for performance assessment.
- Implement the pipeline using the provided data.
- Report results and share valuable findings.

### Dataset
- A set of images containing text, split into two folders: `Alcohol` and `Non-alcohol`, based on whether the text mentions alcohol.

### OCR Tools
- The project uses off-the-shelf Python OCR libraries (EasyOCR and PaddleOCR) for text extraction from images. Both are easy to set up and run on CPU.

### Evaluation Criteria
- **Robustness:** How well does the pipeline handle noisy, distorted, or adversarial text?
- **Explainability:** Documentation of approach and parameter choices.
- **Performance:** Final classifier is measured on unseen data.
- **Creativity & Initiative:** Use of custom text-cleaning, advanced post-processing, or domain knowledge.

## 2. Main Analysis (Summary of Report)

### Data Exploration
- The dataset consists of images with superimposed text, divided into `Alcohol` and `Non-alcohol` categories.
- Example: An image with the text "Don't miss limited edition nightlife bourbon event" is labeled as `alcohol`, while "energy drink" is labeled as `nonalcohol`.

### Methodology
- **Text Extraction:** Used EasyOCR and PaddleOCR to extract text from each image.
- **Text Cleaning:** Applied normalization, lowercasing, digit/whitespace removal, and punctuation reduction to improve OCR output quality.
- **Keyword Matching:** Built a list of alcohol-related keywords (e.g., vodka, rum, whiskey, bourbon, wine, beer, etc.) and non-alcohol keywords (e.g., chocolate, hot, coffee, tea, etc.).
- **Classification Logic:**
  - If the extracted text contains any alcohol-related keyword and not a non-alcohol keyword, classify as `alcohol`.
  - If the text contains only non-alcohol keywords, classify as `nonalcohol`.
  - If ambiguous, fallback to a default or majority class.
- **Metrics:**
  - Used accuracy, precision, recall, and F1-score to evaluate performance on both training and validation sets.
  - Analyzed failure cases (e.g., images with poor OCR results, ambiguous text, or brand names not in the keyword list).

### Key Findings
- OCR quality is critical: images with low contrast or distorted text often yield poor results.
- Keyword-based classification is effective but limited by the comprehensiveness of the keyword list.
- Some images are ambiguous or contain slang/brand names, which can lead to misclassification.
- With more time, improvements could include: advanced NLP for context, expanding the keyword list, or using a small neural network for text classification.

## 3. How to Run

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

### Notes
- The pipeline is CPU-only and does not require a GPU.
- You can use either EasyOCR or PaddleOCR for text extraction (see `requirements.txt` for installation).
- For best results, ensure images are clear and text is legible.

## 4. Additional Information
- The code is modular and well-documented.
- For further improvements, consider integrating more advanced NLP models or expanding the keyword lists.
- For any questions or clarifications, please refer to the comments in the code or contact the project maintainer. 
