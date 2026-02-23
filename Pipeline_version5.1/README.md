Folder -> RedditWork 5.1 

Labeling_dataset.ipynb -> handles all formatting the dataset and adding or removing  

manual_label_batch1_updated.csv Data shape: 188, prev was 177, added new 11 label = 1 to the dataset to balance the label 0 and label 1 ratio 
# dataset Info
test_data_with_labels_BINARY.csv (167 rows) 

fewshot_examples.csv (10 old one)(i can’t find the updated 20 samples i ran with); have to create one again  

manual_label_batch2.csv - > new labels 1, 11 in total 

manual_label_batch1_updated.csv -> 188 ; Batch 1: 167 rows Batch 2: 10 rows Filtered Batch 2: 11 rows; the new dataset 

Create a new few-shot example csv  

fewshot_examples.csv (10 rows) + filtered_label_1_batch2.csv (11 rows)-> fewshot_examples_updated.csv 

#In-Context Learning (ICL)

Original dataset shape: (188, 4) Columns: ['id', 'text', 'similarity_score', 'label']  

Few-shot examples: 20 Test samples: 168 

Label distribution: label 0     139 , Label 1  49 

Few-shot example taken label 1 = 20  

#Bert Topic
->   ran on 188 data points 


LABEL DISTRIBUTION IN TRAINING DATA: 
================================================== 
label 
0    139 
1     49 
Name: count, dtype: int64 
Class 0: 139 samples 
Class 1: 49 samples 
 
Number of unique labels: 2 
 

CLASS WEIGHTS (to handle imbalance): 
================================================== 
Class 0 weight: 0.6763 
Class 1 weight: 1.9184 
(Higher weight = model penalized more for getting it wrong)


PREDICTION RESULTS: 
================================================== 
Predicted class 0: 108 samples 
Predicted class 1: 80 samples 
 
Confusion Matrix: 
[[100  39] 
[  8  41]] 


#Classification Trigrams

Ran with new dataset: manual_label_batch1_updated.csv 

Test set 20% = 36 samples and training set 80% = 141 samples 

N-gram type: unigrams_to_trigrams  

Using: unigrams + bigrams + trigrams  

Feature matrix shape (train): (150, 1000) 

 Feature matrix shape (test): (38, 1000) Number of features: 1000 

Results Summary:  

Model: SVM (linear kernel)  

Features: unigrams_to_trigrams  

Accuracy: 89.47%  

F1 Score (Weighted): 0.8976  

F1 Score (Macro): 0.8721 
