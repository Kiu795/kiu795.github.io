# MLCourse1 Lab Summary

***

## 1. Lab04 Gradient Descent

### 1.1 Tools

```python
import math, copy
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients
```

### 1.2 Problem Statement

课程中出现过的两个数据点，一栋1000平方英尺的房子以30万美元售出，一栋2000平方英尺的房子以50万美元售出。

| 面积(1000 sqft) | 价格(1000s of dollars) |
| --------------- | ---------------------- |
| 1               | 300                    |
| 2               | 500                    |

```python
# Load our data set
x_train = np.array([1.0, 2.0])   #features
y_train = np.array([300.0, 500.0])   #target value
```



### 1.3 `Compute_Cost`

```python
#Function to calculate the cost
def compute_cost(x, y, w, b):
   
    m = x.shape[0] 
    cost = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost

    return total_cost
```

