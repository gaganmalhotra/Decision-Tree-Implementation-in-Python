import numpy as np
from itertools import groupby
import math
import collections
from copy import deepcopy
import pickle

class TreeNode:
    def __init__(self,split,col_index):
        self.col_id= col_index
        self.split_value= split
        self.parent=None
        self.left= None
        self.right= None
        
class Tree():
    
    def __init__(self):
        self.treemodel = None
  
    def train(self,trainData):
        #Attributes/Last Column is class
        self.createTree(trainData)
        
    
    def createTree(self,trainData):
        #create the tree
        self.treemodel=build_tree(trainData,[])
        saveTree(self.treemodel)
        
    def accuracy_confusion_matrix(self,testData):
        #prints the tree confusion matrix along with the accuracy
        build_confusion_matrix(self.treemodel,testData)
     
#returns the best split on the data instance along 
#with the splitted dataset and column index     
def getBestSplit(data):
    #set the max information gain
    maxInfoGain = -float('inf')
    
    #convert to array
    dataArray = np.asarray(data)    

    #to extract rows and columns
    dimension = np.shape(dataArray)
    
    #iterate through the matrix
    for col in range(dimension[1]-1):
       dataArray = sorted(dataArray, key=lambda x: x[col])
       for row in range(dimension[0]-1):
           val1=dataArray[row][col]
           val2=dataArray[row+1][col]
           expectedSplit = (float(val1)+float(val2))/2.0
           infoGain,l,r= calcInfoGain(data,col,expectedSplit)
           if(infoGain>maxInfoGain):
                maxInfoGain=infoGain
                best= (col,expectedSplit,l,r)
    return best      
     
#This method is used to calculate the gain and returns 
#the left and right data as per the split
def calcInfoGain(data,col,split):
    totalLen = len(data)
    infoGain = entropy(data)
    
    left_data, right_data =getDataSplit(data,split,col)
    
    infoGain = infoGain- (len(left_data)/totalLen * entropy(left_data))
    infoGain = infoGain- (len(right_data)/totalLen * entropy(right_data))
   
    return infoGain,left_data,right_data
    

def getDataSplit(data, split, col):
    l_data=[]
    r_data=[]
 
    for val in data:
        if(val[col]<split):
            l_data.append(val)
        else:
            r_data.append(val)
    
    return l_data,r_data

#calculates the entropy of the data set provided    
def entropy(data):
    totalLen = len(data)
    entropy = 0
    group_by_class= groupby(data, lambda x:x[5])
    for key,group in group_by_class:
        grp_len = len(list(group))
        entropy+= -(grp_len/totalLen)*math.log((grp_len/totalLen),2)
    return entropy   
           
#this method builds the decision tree recursively until the leaf nodes are reached           
def build_tree(data,parent_data):
    #code to find out if the class variable is all one value
    count=0;
    group_by_class= groupby(data, lambda x:x[5])
   
   #finds out if all the instances have the same class or not
    for key,group in group_by_class:
        count=count+1;
    
    #if same class for all instances then return the leaf node class value
    if(count==1):
        return data[0][5];
        
    elif(len(data)==0):
        #this counts all the column class variable row values and finds most common in it
        return collections.Counter(np.asarray(data[:,5])).most_common(1)[0][0]
    
    else:
        bestsplit= getBestSplit(data)
        node = TreeNode(bestsplit[1],bestsplit[0])
        node.left= build_tree(bestsplit[2],data)
        node.right= build_tree(bestsplit[3],data)
        return node
 
#this method is used to classify the test set with the model created 
def classify(tree, row):
    if type(tree)==str:
        return tree
    if row[tree.col_id]<=tree.split_value:
        return classify(tree.left, row)
    else:
        return classify(tree.right, row)

#this method saves the decision tree model using pickle package    
def saveTree(tree):
    decisionTree= deepcopy(tree)
    pickle.dump(decisionTree,open('model.pkl','wb'))

#this method creates a confusion matrix and finds accuracy for test dataset    
def build_confusion_matrix(tree, data):
    confusion_mat = [[0 for row in range(4)]for col in range(4)]
    
    total_len=len(data)
    num_correct_instances=0;
    num_incorrect_instances = 0;
    
    #map required to build the confusion matrix
    map={'B':0,'G':1,'M':2, 'N':3}    
    
    for row in data:
        actual_class = row[5]
        predicted_class=classify(tree, row)
        if(actual_class==predicted_class):
            num_correct_instances=num_correct_instances+1
            confusion_mat[map.get(actual_class)][map.get(actual_class)]=confusion_mat[map.get(actual_class)][map.get(actual_class)]+1
        else:
            num_incorrect_instances=num_incorrect_instances+1
            confusion_mat[map.get(actual_class)][map.get(predicted_class)]=confusion_mat[map.get(actual_class)][map.get(predicted_class)]+1
    
    print("Accuracy of the model:",(num_correct_instances/total_len)*100,"%")
    print("Correct instances",num_correct_instances)
    print("Incorrect instances",num_incorrect_instances)
    
    
    print_map={0:'B',1:'G',2:'M', 3:'N'}   
    print("Confusion Matrix:")
    
    print("    B  G  M  N")
    
    ind=0;
    #printing matrix
    for row in confusion_mat:
        print(print_map.get(ind),"", row)        
        ind+=1
        