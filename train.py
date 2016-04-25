from Tree_Model import Tree
import sys

#this method is used to read the csv file provided in the argument
def read(data):
    result=[]
    for sample in data.split('\n')[1:]:
        samplesplit=sample.split(',')    
        list1= [float(val) for val in samplesplit[:-1]]
        list1.append(samplesplit[-1])
        result.append(list1)
    return result

raw_data=open(sys.argv[1]).read();
data=read(raw_data)
model =  Tree()
model.train(data)
