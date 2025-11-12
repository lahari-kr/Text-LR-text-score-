# Text-LR
# Neural Network Model Comparison Dashboard  
### Baseline vs Transformer Model on Synthetic Book Dataset  

This project demonstrates the **comparison of a Baseline Dense Neural Network** and a **Transformer-based model** on a **synthetic book dataset**.  
It includes two main components:  
- **Model Training and Evaluation (source.py)**  
- **Interactive Visualization Dashboard (source1.py)**  


##  Project Overview

This experiment evaluates how well a transformer architecture performs against a dense feedforward neural network on a text classification task.  
The dataset (`synthetic_book_dataset.csv`) contains book entries with features such as:
- **Title**
- **Genre**
- **Description**

The goal is to **predict the book description** category based on title and genre.


##  Machine Learning Models Used

###  Baseline Model 
A fully connected neural network trained on encoded title and genre data.
```python
Dense(128, activation='relu')
Dropout(0.3)
Dense(256, activation='relu')


### Transformer Model

A lightweight Transformer-based architecture leveraging multi-head attention and layer normalization for contextual understanding.

Dense Embedding (256 units)
MultiHeadAttention (8 heads)
LayerNormalization
FeedForward Layer (512 units)
Dropout (0.4)
Output Layer (Softmax)

Dropout(0.3)
Dense(output_dim, activation='softmax')
