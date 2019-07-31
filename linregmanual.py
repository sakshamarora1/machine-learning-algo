from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

def create_dataset(n, variance, step=2, correlation=False):
    val = 1
    ys = []
    for _ in range(n):
        y = val + random.randrange(-variance,variance)
        ys.append(y)
        if correlation == 'pos':
            val+=step
        elif correlation == 'neg':
            val-=step

    xs = [ i+5 for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_slope_intc(xs,ys):
    m = ( ( mean(xs)*mean(ys) - mean(xs*ys) ) /
            ( mean(xs)**2 - mean((xs)**2) )     )
    b = mean(ys) - m * mean(xs)
    return m,b 

def squared_error(ys_orig,ys_line):
    return sum( (ys_line - ys_orig)**2 )

def coeff_determination(ys_orig,ys_line):
    y_mean_line = [ mean(ys_orig) for y in ys_orig ]
    squared_error_regr = squared_error(ys_orig,ys_line)
    squared_error_y_mean = squared_error(ys_orig,y_mean_line)
    return 1 - ( squared_error_regr / squared_error_y_mean)

xs,ys = create_dataset(40,20,2,correlation='pos')
m,b = best_slope_intc(xs,ys)

regression_line = [ (m*x) + b for x in xs ]

predict_x = 50
predict_y = m*predict_x +b

r_sq = coeff_determination(ys,regression_line)
print(r_sq)
print('y = '+str(m)+'x + '+str(b))
print('The predicted y-coordinate is :- ',predict_y)
plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y,c='k')
plt.plot(xs,regression_line,c='red')
plt.show()