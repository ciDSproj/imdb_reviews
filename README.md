# IMDB Sentiment Classification with LSTM
A neural network model that classifies movie reviews from the IMDB dataset as **positive** or **negative** using an LSTM architecture. This project demonstrates deep learning model development, evaluation, and custom text inference.

## Overview  
This project builds a binary sentiment classifier using the IMDB movie reviews dataset. The model uses an **Embedding layer**, an **LSTM layer with dropout**, and **Dense layers** to learn patterns in text sequences. After training, the model achieved **~86% accuracy** on unseen test data. The project also includes a custom function to predict sentiment for any user‑provided review.

## Resources Used
- Python 3.7
- Matplotlib for data visualization
- TensorFlow and Keras for building the deep neural network

## Dataset 
The project uses the **IMDB Movie Reviews** dataset provided by `tensorflow.keras.datasets`. 
- 25,000 preprocessed reviews labeled as positive or negative
- Reviews are pre‑encoded as integer sequences  
- Vocabulary size limited to **10,000 most frequent words**

## Key Features  
- LSTM‑based neural network for sequence modeling  
- Text preprocessing and tokenization using IMDB’s built‑in word index  
- Sequence padding for uniform input length  
- Training with validation tracking  
- Test evaluation and accuracy reporting  
- Custom review prediction function for real‑time inference

## Model Architecture  
Embedding (input_dim=10000, output_dim=64)
LSTM (units=64, dropout=0.2, recurrent_dropout=0.2)
Dense (units=32, activation='relu')
Dense (units=1, activation='sigmoid')

This architecture is designed to capture sequential patterns in text while maintaining generalization through dropout.

## Training  
- **Loss:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Epochs:** 3  
- **Batch Size:** 256  
- **Validation Split:** 20%  

Training accuracy improves steadily, reaching ~88% on the final epoch.

## Results  
- **Test Accuracy:** 0.8581 
- Example prediction:  
  - *Predicted sentiment:* positive  
  - *Probability:* 0.9831
  - *True label:* positive

## Training history plots
- **Accuracy:** Training accuracy rises from ~0.69 to ~0.88, while validation accuracy increases from ~0.80 to ~0.86. Both 
    curves improve steadily and stay close to each other, indicating good generalization.  
- **Loss:** Training loss decreases from ~0.55 to ~0.30, and validation loss decreases from ~0.42 to ~0.35. The gap between 
    the curves remains small, suggesting the model is not overfitting.

Together, these curves show that the model is learning efficiently and maintaining stable performance on unseen data.

## Interpretation  
Across three epochs, the model demonstrates:
- Rapid learning  
- Stable validation performance  
- No signs of overfitting  
- Strong alignment between accuracy and loss curves  

These results support the final test accuracy of **86%**, confirming that the LSTM architecture is effective for binary sentiment classification on the IMDB dataset.

## Custom Review Prediction  
The project includes a helper function that:  
1. Tokenizes custom text using IMDB’s word index  
2. Encodes unknown words as a special token  
3. Pads the sequence to the required length  
4. Returns the predicted sentiment and probability  

Example:
review = "I loved the movie, the story was very moving."
predict_review(review)
Output: Prediction: positive (0.6324)





  
