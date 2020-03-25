# Insider-Threat-Detection

Insider threats are a cause of serious concern to organizations because of the significant damage that can be inflicted by malicious insiders. In this paper, we propose an approach for insider threat classification which is motivated by the effectiveness of pre-trained deep convolutional neural networks (DCNNs) for image classification. In the proposed approach, we extract features from usage patterns of insiders, and represent these features as images. Hence, images are used to represent the resource access patterns of the employees within an organization. After construction of images, we use pre-trained DCNNs for anomaly detection, with the aim to identify malicious insiders. The proposed approach is evaluated using the MobileNetV2, VGG19 and ResNet50 pre-trained models, and a benchmark dataset. 

## Modules
The project consists of the following modules:
  * Feature Extraction
  * Image Representation
  * Classification
  
### Feature Extraction
The feature extraction process reads the log files which consists of logon/logoff information, the files handled by the user, the external devices used by the user, email communications sent/received by the user, details of the browsing history. The logs contain raw information in the form of rows and columns. These logs are used to extract useful features from the data. We have used the popular CERT CMU insider threat data set. 

### Image Representation
The features extracted from the log files are represented as grayscale images. These images are provided as input to the learning model to predict malicious and non-malicious users. The log files from the CMU dataset are pre-processed to obtain the feature vector of each user for each day. This is implelemted using Matlab.

### Classification
Deep learning models are applied to images to perform anomaly detection. Any deviation from the normal user behavior is considered as an anomaly. The anomaly detection approach focuses on mining anomalous patterns from the extracted features. We use the Deep Convolutional Neural Network (DCNN) for this purpose. Transfer learning is commonly used with problems on predictive modeling where the input is in the form of images. We used the MobileNetV2, VGG19 and ResNet50 to demonstrate the proposed approach.
