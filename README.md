# Comparison of Handcrafted and CNN-based Methods in the Classification of Skin Lesions

## Project Goals

The aim of this project is to compare the lesion classification performance of dermatoscopic images on the ISIC dataset using both handcrafted feature extraction methods and convolutional neural network (CNN) based deep learning. First, the accuracy obtained with the SVM classifier by extracting GLCM, color, and shape-based features on a balanced dataset is calculated, along with F1-score and other metrics using the `sklearn` library. Then, fine-tuning is performed with pre-trained CNN models such as ResNet50 using a transfer learning approach and evaluated with the same metrics and 5-fold cross-validation. Finally, the obtained performance values are subjected to paired t-test, Wilcoxon signed-rank, and McNemar’s tests to determine which method is more effective.

## Dataset and Preprocessing

The project was implemented in Python using Google Colab. Required libraries (`os`, `pandas`, `sklearn`, `torchvision`, `tqdm`) were imported in the `79479_75700_Project.ipynb` notebook, which was connected to Google Drive via `mount`. Images (`.jpg`) and metadata files (`.csv`) were loaded and filtered to ensure consistency. Rare classes were removed, and remaining labels were encoded numerically.

Training data was split into 80% training and 20% validation sets. The original class distribution was imbalanced, so classes with too many samples were downsampled randomly, and those with too few samples were upsampled via binary sampling to create a balanced dataset. Sample images from each class were visualized during preprocessing.

Data preprocessing included defining a PyTorch `Dataset` class and using `DataLoader` to batch samples. Images were transformed sequentially to the required dimensions and normalized before being fed into models.

## Methods

### Handcrafted Features + SVM

1. Lesion masks were obtained by applying Gaussian and median blurs, Otsu thresholding, and morphological operations.
2. For each masked region, features such as mean, standard deviation, largest region area, perimeter, and Hu moments were computed for each RGB channel.
3. Contrast and homogeneity features were extracted from the gray-level co-occurrence matrix (GLCM).
4. All features were concatenated into a single feature vector and scaled with `StandardScaler`.
5. An SVM classifier with RBF kernel and balanced class weights was trained on the feature matrix.
6. Validation predictions yielded an overall accuracy of 0.87, with precision of at least 0.80 per class. The `unknown` class showed lower recall (0.62) and F1-score (0.70) compared to other classes.

### CNN-based Classification

1. A custom `SimpleCNN` architecture was compared against pre-trained `ResNet50` and `vit_b_16` models, with ResNet50 showing the best baseline performance.
2. The ResNet50 model was fine-tuned on the balanced dataset using cross-entropy loss and the Adam optimizer (learning rate=3e-5, weight decay=1e-4) for 25 epochs with early stopping.
3. 5-fold cross-validation was conducted to ensure robustness across data splits.
4. The fine-tuned CNN achieved approximately 95% accuracy and overall F1-scores of 95–96%, with near-perfect AUC values. Melanoma, lentigo NOS, and seborrheic keratosis were classified with 100% F1-score, while minor misclassifications occurred in the nevus and unknown classes.

## Statistical Tests & Results

All tests used a significance level of 0.05:

- **Cross-Validation Comparison**: Mean and standard deviation across folds confirmed that the CNN outperforms the handcrafted+SVM approach on all metrics.
- **Paired t-Test**: Significant differences (p < 0.05) were found in accuracy and F1-score, favoring the CNN model.
- **Wilcoxon Signed-Rank Test**: Results corroborated the paired t-test findings.
- **McNemar’s Test**: Analysis of classification disagreements showed that the CNN corrected more misclassifications than the handcrafted method, indicating superior performance.

## Conclusion

Statistical analyses confirm that the CNN-based method significantly outperforms the handcrafted feature extraction + SVM approach across multiple metrics. The CNN’s capacity to learn complex visual representations provides a clear advantage in challenging tasks such as skin lesion classification, validating the preference for deep learning methods in medical image analysis.
