![Regression CS Header](https://raw.githubusercontent.com/boogiedev/regression-case-study/master/regressionedaheader.png)

<p align="center">
  <img src="https://img.shields.io/badge/Maintained%3F-In Progress-green?style=flat-square"></img>
</p>

### Contributors
|  [Feli Gentle](https://github.com/oro13)  |
 [Tyler Woods](https://github.com/tylerjwoods)  |
 [Wesley Nguyen](https://github.com/boogiedev)  |
   
---

## Sections:
 |  **[Introduction](#introduction)**  |
 **[Plan of Attack](#plan-of-attack)**  |
 **[Analysis & Modeling](#analysis-and-modeling)**  |
 **[Performance](#performance)**  |
 **[Takeaways](#takeaways)**  |

---

## Introduction

**Main Goal:** <br>
Predict the sale price of a particular piece of equipment at auction based on it's usage, equipment type, and configuration. <br>

*Note: This data is sourced from auction results postings and includes information on usage and equipment configurations.*

**Evaluation:**
The evaluation of our model will be based on Root Mean Squared Log Error.
Which is computed as follows:

![Root Mean Squared Logarithmic Error](images/rmsle.png)

where *p<sub>i</sub>* are the predicted values (predicted auction sale prices) 
and *a<sub>i</sub>* are the actual values (the actual auction sale prices).


---

## Plan of Attack

<img align="right" src="https://image.flaticon.com/icons/svg/81/81203.svg" width="100" height="100">
Initally going into this case study, we decided to tackle the tasks of cleaning the data, and getting a baseline model together as a group. We did this to ensure that everyone was on the same level of understanding before we delved into partitioned tasks.

<br>
<br>
<br>

**Cleaning Data:**
With the intent of using Linear Regression in mind, we processed and cleaned some of the data in order for this to be possible. Using a custom function, we dropped rows that had a high percentage of null values.

<details>
  <summary>
    Columns with more than 50% Missing Values
  </summary>
<img src="https://raw.githubusercontent.com/boogiedev/regression-case-study/master/images/cleaningdata1.png"></img>

</details>

<br>


---

## Analysis and Modeling

**Baseline Model: Linear Regression**

```python
# Split up Data Between Features (X) and SalePrice, i.e. the Target Values (y))
X = clean_df.drop(columns=['SalePrice'])
y = clean_df['SalePrice']

summary_model(X, y)
```
<details>
  <summary>
    OLS Summary 
  </summary>
<img src="https://raw.githubusercontent.com/boogiedev/regression-case-study/master/images/olssummary.png"></img>
</details>

<br>

```python
# Split up Data Between Features (X) and SalePrice, i.e. the Target Values (y))
X = clean_df.drop(columns=['SalePrice'])
y = clean_df['SalePrice']

y.hist(bins=100)
plt.show()
```

<details>
  <summary>
    Sales Price Histogram
  </summary>  
  <img src="https://raw.githubusercontent.com/boogiedev/regression-case-study/master/images/prelogTargetHist.png"></img>
</details>

<br>

```python
# Split up Data Between Features (X) and SalePrice, i.e. the Target Values (y))
X = clean_df.drop(columns=['SalePrice'])
# Log the Target
y = np.log(clean_df['SalePrice'])

y.hist(bins=100)
plt.show()
```
<details>
  <summary>
    Sales Price Histogram (Log)
  </summary>  
  <img src="https://raw.githubusercontent.com/boogiedev/regression-case-study/master/images/postlogTargetHist.png"></img>
</details>

<br>

<details>
  <summary>
    Cross Validation Errors (Pre-Log)
  </summary>  
<p>
 
  
```python
n_folds = 10
kf = KFold(n_splits=n_folds, shuffle=True)
test_cv_errors, train_cv_errors = np.empty(n_folds), np.empty(n_folds)
X_array = np.array(X)
y_array = np.array(y)

for idx, (train, test) in enumerate(kf.split(X)):
    model = LinearRegression()
    model.fit(X_array[train], y_array[train])
    y_hat = model.predict(X_array[test])
    y_train = model.predict(X_array[train])
    
    train_cv_errors[idx] = rmse(y_array[train], y_train)
    test_cv_errors[idx] = rmse(y_array[test], y_hat)

train_cv_errors, test_cv_errors
```  


</p>

</details>


```python
(array([0.40606488, 0.40622868, 0.40621928, 0.40665517, 0.40652397,
        0.40657838, 0.40608442, 0.40651964, 0.40617827, 0.40609071]),
 array([0.40858626, 0.40711805, 0.40721177, 0.40327332, 0.40446531,
        0.403964  , 0.40841977, 0.40449285, 0.40757546, 0.40838406]))
```



---

## Performance

---

## Takeaways 

---
