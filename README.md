# Decision-Tree-Implementation-in-Python
Coded decision tree in python to generate a model with accuracy 91% on the test dataset


### Algorithm Explanation:
- Take Data input from CSV file
- Decision tree is built as below-
    - Find which attribute has the maximum information gain by finding the
    entropy for tuple.
    - Now find the best split value of that particular attribute and save it in the
    TreeNode.
    - Now as per the split separate the dataset into two parts - Left and Right and
    then recursively find the attribute with maximum gain with repeating the
    steps above.
    - At the end of the recursion we will have the model built with leaf nodes
    representing the class variables - ‘B’, ‘G’, ‘M’, ‘N’
- After the tree is built, we save the tree using pickle package in python
- To test the model on test dataset, we find confusion matrix and the accuracy of
the model as below-
    - Key concept is that we run our model using the classify() method which will
    traverse the model as per the each instance of the test dataset and finds out
    the predicted class.
    - Accuracy is calculated using the number of instances
    

### How to run the program:
To train the decision tree example:
``` python train.py train_data.csv ```

To test the decision tree example:
``` python test.py test_data.csv ```

We dont need to provide the pickle file name in the arguments as it is being saved as 'model.pkl' in the code
