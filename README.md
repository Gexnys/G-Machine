# "G-Machine: Extreme ViT Confusion Matrix, Vision Transformer Machine Learning"

# 1. Data Preparation
• Downloads the MNIST dataset, which contains handwritten digit images.

• Normalizes the pixel values to have zero mean and unit variance.

• Splits the data into 80% training and 20% testing sets.

• Converts the data into PyTorch tensors and batches it for efficient processing.

# 2. Model Setup
• Builds a Vision Transformer (ViT) model that divides each image into small patches and treats them as a sequence.

• Uses transformer encoder blocks to learn global relationships between these patches via self-attention mechanisms.

• Ends with fully connected layers for classifying the digit classes.

# 3. Training the Model
• Moves the model to GPU (if available) or CPU.

• Uses the AdamW optimizer and CrossEntropyLoss to train the model.

• Applies a learning rate scheduler (CosineAnnealingWarmRestarts) to adjust the learning rate over time.

• For each epoch (one full pass over the training data), the model:

• Makes predictions on batches of images.

• Calculates the loss (error) between predictions and true labels.

• Updates model weights to minimize the loss.

• Prints the loss and accuracy at the end of each epoch.

# 4. Evaluating the Model
• After training, the model predicts labels for the unseen test dataset.

• Calculates performance metrics like accuracy and weighted F1 score.

• Generates a confusion matrix plot showing which digits the model predicts correctly or confuses.

# 5. What changed in the G-Machine update

• Data Augmentation was added (RandomRotation, RandomAffine, RandomErasing) → stronger generalization.

• A lighter & deeper ViT (embedding size = 192, depth = 12, heads = 6) was used → balance between speed and accuracy.

• Stochastic Depth (DropPath) → reduced Transformer overfitting.

• Label Smoothing → CrossEntropyLoss(label_smoothing=0.1) was applied to prevent overfitting.

• Gradient Clipping was added → to prevent exploding gradients.

• Early Stopping → training stops if no improvement is observed for 5 epochs.

• Test-Time Augmentation (TTA) → multiple predictions under small rotations were averaged during inference.

• TensorBoard logger was added → to visualize loss and accuracy.

#Summary
 • G-Machine builds and trains an advanced Vision Transformer to recognize handwritten digits from MNIST. It handles everything from data loading and preprocessing, through training, to evaluation automatically •

# This code was written by Gexnys.
• Email address: developergokhan@proton.me


