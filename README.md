# Machine Learning (ML) Course

This repository is a comprehensive course on **machine learning**, covering essential topics and their implementations from **open classroom** and additionally examples. The course is designed to provide both theoretical understanding and practical skills in various machine learning algorithms and techniques. The content follows the **Stanford** course **CS229**, taught by *Andrew Ng*. We augment this course with implementations in Python.

## Outlines
----------- 

## Regression:
- **Linear Regression**: modeling  relationship between a dependent variable and one independent variable by fitting a linear equation to the observed data.
- **Cost Function**:  evaluating the performance of a ML model by quantifying the difference between the predicted and actual values, guiding the optimization process.
- **Optimizations**: Techniques used to adjust the parameters of model to minimize the cost function and improve model performance, including gradient descent and other optimization algorithms.
- **Mutliple Linear Regression**: An extension of linear regression that models the relationship between a dependent variable and multiple independent variables by fitting a linear equation to the observed data.
## Classification:
- **Logistic Regression**: A classification algorithm that models the probability of a binary outcome based on one or more predictor variables using a logistic function.
- **Support Vector Classifier**: modeling the relationship between input features and continuous target values by finding the optimal hyperplane that minimizes prediction errors.
- **Feature Descriptor**: Methods for detecting and extracting features from images, such as LBP, SIFT, ORB, HOG and others.
- **Feature Based Learning (Annotation)**: Methods for training various ML algorithm to learn patterns through features.
## Artificial Neural Networks (ANN):
#### Introducing Building Blocks of Artificial Neural Networks:
- **Input Layers**: The initial layer that receives the input data.
- **Hidden Layers**: Intermediate layers that process inputs through weighted connections.
- **Output Layers**: The final layer that produces the network's predictions or outputs.
- **Perceptron**: A single-layer neural network unit used for binary classification.
- **Activation Functions**: Functions that introduce non-linearity, allowing the network to learn complex patterns.
- Forward Pass: The process of calculating the output of a neural network by passing input data through each layer of the network.
- Backward Pass (Back Propagation): The method of adjusting the weights of a neural network by propagating the error gradient backward through the network using the chain rule.
- Shallow Networks: Neural networks with few layers, suitable for simpler problems but prone to underfitting.
- Deep Neural Networks: Neural networks with many layers, capable of modeling complex patterns but at risk of overfitting.
## Implementation:
Each lecture includes the following implementations:
- **Manual Implementation**: Concepts are implemented using numpy without relying on built-in machine learning libraries, offering a clear understanding of the underlying mechanics.
- **scikit-learn Implementation**: Demonstrates basic machine learning models using scikit-learn's built-in functionalities for ease of use and efficiency.
- **Keras Implementation**: Utilizes the high-level Keras API, which provides user-friendly implementations of specialized machine learning and deep learning models with both sequential and functional approaches.
- **PyTorch Implementation**: Employs the PyTorch framework, known for its flexibility and research-oriented design, to implement neural networks with full control over their architecture and training process.
  
## Getting Started

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/qazimsajjad/Machine-Learning-Course.git
    ```
2. **Navigate to the Directory**:
    ```sh
    cd Machine-Learning-Course
    ```
3. **Install Dependencies**:
    ```sh
    python => 3.9
    numpy
    matplotlib
    pillow
    opencv
    sk-learn
    torch
    keras
    tensorflow
    
    ```

## Usage

Open the Jupyter Notebooks provided in the repository to explore different **Machine Learning** techniques. Each notebook contains detailed explanations, code implementations, and example images to help you understand the concepts.

## Contributor:

**Kaleem Ullah**
Research Assitant **Digital Image Processing (DIP) Lab** Department of Computer Scinece Islamia College University, Peshawar, Pakistan.
Remote Research Assistant **Visual Analytics Lab (VIS2KNOW)** Department of Applied AI Sungkyunkwan University, Seoul, South Korea.
