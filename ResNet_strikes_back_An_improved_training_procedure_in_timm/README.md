# Result Comparison

Paper Link 

[ResNet Strikes Back: An Improved Training Procedure in timm]https://arxiv.org/pdf/2110.00476)

## Objective
The goal is to compare the results obtained from training a pre-trained ResNet50 model on the MNIST dataset using standard procedures, with the results mentioned in the paper titled "ResNet Strikes Back: An Improved Training Procedure in timm." This comparison highlights key insights into the performance of deep learning models, particularly in terms of accuracy, loss, and training complexity across datasets of varying difficulty levels.

### Current Experiment Results (MNIST)
In the current experiment, the ResNet50 model was trained for 5 epochs on the MNIST dataset using basic training procedures. The following results were obtained:

- **Training Loss (Epoch 5):** 0.5382
- **Training Accuracy (Epoch 5):** 99.87%
- **Validation Loss (Epoch 5):** 0.0011
- **Validation Accuracy (Epoch 5):** 98.87%

### Summary of the Paper
The paper focuses on improvements in training procedures for deep models such as ResNet50, with specific attention on Mixup, CutMix, Cosine Learning Rate Schedulers, and advanced augmentations. These techniques were tested on ImageNet and compared to traditional training methods. The paper highlights the following:

- Improved Regularization Techniques: Enhanced Mixup and CutMix strategies improved generalization on larger, more complex datasets.
- Learning Rate Schedulers: Using cosine annealing schedules resulted in better model convergence.
- Batch Size and Training Duration: The paper experimented with large batch sizes and extended training (600 epochs for the most comprehensive procedure).

### Dataset Differences
#### ImageNet (Used in the Paper):

- Dataset Size: 1.2 million images across 1,000 classes.
- Complexity: High-resolution, colored images with diverse categories.
- Training Time: Long training times (days to weeks) on high-performance GPUs.
- ResNet50 Performance: Top-1 error of 23.56%, Top-5 error of 3.57%.

### MNIST (Used in the Current Experiment):

- Dataset Size: 60,000 training images, 10,000 test images across 10 classes (grayscale, 28x28).
- Complexity: Simpler dataset, focusing on digit recognition.
- Training Time: Training is completed in a few minutes with minimal GPU resources.
- ResNet50 Performance: Training accuracy of 99.87% and validation accuracy of 98.87% after 5 epochs.

### Model Architecture and Transfer Learning
Paper Results (ResNet50 on ImageNet):

- The ResNet50 model was trained on ImageNet using advanced augmentations (Mixup, CutMix) and cosine learning rate scheduling.
- The final output layer corresponds to 1,000 classes for ImageNet classification.
- The model achieved 23.56% Top-1 error and 3.57% Top-5 error on ImageNet after extensive training.

#### Current Experiment (ResNet50 on MNIST):

- The ResNet50 model was trained on MNIST, with the final layer modified to output 10 classes.
- Standard augmentations like RandomResizedCrop and HorizontalFlip were used, without Mixup or CutMix.
- Achieved 98.87% validation accuracy after only 5 epochs, demonstrating the model’s strong performance on simpler datasets like MNIST.

### Training Epochs
#### Paper Results (ImageNet):

- Training took place over 100–600 epochs, utilizing advanced learning rate scheduling and regularization methods to improve accuracy.
- This extensive training is typical for ImageNet due to the complexity of the dataset and the high number of classes.

#### Current Experiment (MNIST):

Training was conducted for 5 epochs, and the model achieved near-perfect accuracy on MNIST.
The simplicity of MNIST allows models to converge quickly, even without extensive training.

### Results Comparison
#### Accuracy
#### Current Experiment (MNIST):

- Training Accuracy (Epoch 5): 99.87%
- Validation Accuracy (Epoch 5): 98.87%
#### Paper Results (ImageNet):

- Top-1 Accuracy: 76.44% (Top-1 error of 23.56%)

- Top-5 Accuracy: 96.43% (Top-5 error of 3.57%)

- Key Difference: The accuracy for MNIST is much higher due to the simplicity of the dataset compared to ImageNet’s complexity, where achieving over 75% Top-1 accuracy is impressive.

#### Loss
#### Current Experiment (MNIST):

- Training Loss (Epoch 5): 0.5382
- Validation Loss (Epoch 5): 0.0011
#### Paper Results (ImageNet):

- Training Loss: Higher throughout training due to the complexity of ImageNet.
Loss was significantly reduced through the use of techniques like Mixup and CutMix, but still remained higher than on MNIST due to the dataset’s size and complexity.
Training Time
#### Current Experiment (MNIST):

- Training was completed in minutes due to the small dataset and low resolution of the images (28x28 pixels).
#### Paper Results (ImageNet):

- Training took days or even weeks due to the size of the dataset and the need for 100+ epochs to achieve competitive performance on ImageNet

### Insights from Comparison
#### Dataset Complexity
- **MNIST:** The simplicity of MNIST allows models to converge quickly, achieving near-perfect accuracy in just a few epochs.
- **ImageNet:** ImageNet requires significant computational resources, time, and advanced techniques like Mixup and CutMix to train models effectively.
#### Transfer Learning Effectiveness
- **Paper:** Transfer learning techniques applied on ImageNet lead to significant improvements in convergence and generalization on complex datasets like CIFAR-10.
- **Current Experiment:** Transfer learning was not employed, but ResNet50 showed high accuracy on MNIST, likely due to the model's capacity to handle more complex tasks than MNIST.
#### Model Architecture Impact
- **ResNet50:** The use of skip connections in ResNet50 helps the model perform well on both simpler datasets (MNIST) and more complex ones (ImageNet). The architecture is highly adaptable and efficient in both scenarios.

#### Conclusion
The comparison between the two experiments demonstrates how different datasets and training procedures impact model performance. The MNIST experiment, with 98.87% validation accuracy after 5 epochs, highlights the ease of achieving high performance on simpler datasets. In contrast, the paper’s approach on ImageNet required significantly more epochs, advanced regularization, and data augmentation techniques to achieve competitive accuracy, proving the adaptability and strength of the ResNet50 architecture across varying complexities.
