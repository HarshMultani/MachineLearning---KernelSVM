# Kernel - SVM
install.packages('caTools')
library('caTools')

# Import the dataset
dataset <- read.csv('Social_Network_Ads.csv')
dataset = dataset[3:5]


# Encoding the target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0,1))


# Splitting the dataset into training and test set
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# Feature Scaling
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])


# Fitting Kernel SVM to the training set
install.packages('e1071')
library(e1071)
model = svm(formula = Purchased ~ ., data = training_set, type = 'C-classification', kernel = 'radial')



# Predicting the test set results
Y_pred = predict(model, newdata = test_set[-3])



# Making the confusion matrix
cm = table(test_set[ ,3], Y_pred)


