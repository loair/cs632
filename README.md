# cs632
Part 1
1.	Yes, it is important. Feature scaling can be achieved by two ways. One is min-max scaling and another is standardization.
Min-max scaling can be done by subtracting the min value and dividing by the max minus the min.
Standardization can be done by subtracting the mean value and then dividing by the variance.

2.	Numeric data can be computed while categorical feature cannot.
We can convert text labels to numbers, or we can use one-hot encoding method.

3.	Our brain is highly prone to overfitting, and testing data can help us to avoid this.

4.	Supervised means you are given labeled training examples.

5.	 I want to include height and radius of its stem as additional features. Cause these two features are numeric and easy to achieve and process.

Part 2
1. Strenths: a. Bag of Words is easy to understand and collect
             b. invariance to scale and orientation
             c. promising to adopt existing algorithms
   Weakness: a. size of vocabulary
             b. effeciency of generating words
             c. feature selection and reduction
             d. accounting for partial information
             
2. Money is most predictive and a is most, and am is least.
   It is because a lot Spams are related with promotion advertisments or financial fraud, so money is correlated with it a lot. The word "am" used a lot in regular emails, so this feature is not very significance.
 
3. Yes. There are two reasons I think causing the problem. 
   1. The parameter K are responsible for misclassifying since it may be too small or too big.
   2. The featur selection - the choice of Bag of Words is related with it.
