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
 **[Exploration](#exploration)**  |
 **[Analysis & Modeling](#analysis-&-modeling)**  |
 **[Performance](#performance)**  |
 **[Takeaways](#takeaways)**  |

---

## Introduction

**Main Goal:** <br>
Predict the sale price of a particular piece of equipment at auction based on it's usage, equipment type, and configuration. <br>

*Note: This data is sourced from auction results postings and includes information on usage and equipment configurations.*

---

## Plan of Attack

Initally going into this case study, we decided to tackle the tasks of cleaning the data, and getting a baseline model together as a group. We did this to ensure that everyone was on the same level of understanding before we delved into partitioned tasks. 


**Cleaning Data:**
With the intent of using Linear Regression in mind, we processed and cleaned some of the data in order for this to be possible. Using a custom function, we dropped rows that had a high percentage of null values.

<details>
  <summary>
    Columns with more than 50% Missing Values
  </summary>
<img src="https://raw.githubusercontent.com/boogiedev/regression-case-study/master/images/cleaningdata1.png"></img>
</details>


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

---

## Exploration

---

## Analysis & Modeling

---

## Performance

---

## Takeaways 

---
