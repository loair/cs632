This code is about a classic deep learning problem - cats and dogs, which is to train a model to classify pictures as a dog or cat correctly. The image size is 32 * 32, and the dataset includes 10000 instances. The datasets is divided randomly as two parts, training data, and viladation data. Training data includes 8000 instances, while vildation data including 2000. 

The method used here is convolutional neural network(CNN), and the summary for the model is shown below: 
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_7 (Conv2D)            (None, 31, 31, 32)        416       
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 30, 30, 32)        4128      
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 15, 15, 32)        0         
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 14, 14, 64)        8256      
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 13, 13, 64)        16448     
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 6, 6, 64)          0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 6, 6, 64)          0         
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 5, 5, 128)         32896     
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 4, 4, 128)         65664     
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 2, 2, 128)         0         
_________________________________________________________________
dropout_5 (Dropout)          (None, 2, 2, 128)         0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 512)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 128)               65664     
_________________________________________________________________
dense_5 (Dense)              (None, 128)               16512     
_________________________________________________________________
dropout_6 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_6 (Dense)              (None, 1)                 129       
_________________________________________________________________
activation_2 (Activation)    (None, 1)                 0         
=================================================================
Total params: 210,113
Trainable params: 210,113
Non-trainable params: 0
_________________________________________________________________

The epohcs is set equal to 40, and the best result of viladation accuracy achieved here is around 75%. The prediction accuracy is 
