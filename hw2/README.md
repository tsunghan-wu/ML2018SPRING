# ML homework-2 code

In this task, we will train on income-prediction. There are three mothods in this repo:

## Three Models

1. generative model 
	- use naive bayes
	- with data standardization
	- feature: drop --> all ? + fnlwgt + native country

2. logistic regression
	- handcraft logistic regression (regularization + adagrad are usage)
	- with standardization
	- feature: drop --> fnlwgt / add --> [age, male, female] with power 2, 3

3. logistic regression
	- use sklearn package (L1 regularization)
	- with standardization
	- feature: add --> [age, capital gain, capital loss, hour per week] with power (2 -> 100)

## How to get the answer easily

```
## generative done
bash hw2_generative.sh $train.csv $test.csv $train_X $train_Y $test_X $output_path

## logistic
bash hw2_logistic.sh $train.csv $test.csv $train_X $train_Y $test_X $output_path

## best
bash hw2_best.sh $train.csv $test.csv $train_X $train_Y $test_X $output_path
```

Then, you will get the answer.


