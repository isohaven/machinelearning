import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('Solarize_Light2')

mean = lambda nums : sum(nums) / len(nums)


def main():
    draw_regression_line([10.908, 12.344, 13.87, 5.706], [0.348, 0.38, 0.46, 0.16])


def draw_regression_line(xs, ys):
    if len(xs) != len(ys):
        raise ValueError('data sets x and y must be of equal length')
    xs = np.array(xs, dtype= np.float64)
    ys = np.array(ys, dtype= np.float64)
    m, b = regression(xs, ys)
    regression_line = [ (m*x) + b for x in xs]
    r_squared = coeff_of_determination(ys, regression_line)

    points = plt.scatter(xs, ys, label = '')

    line = plt.plot(xs, regression_line, label = 'R² = {}'.format(r_squared))
    print('y = {}x + {}'.format(m, b))
    print('R² = {}'.format(r_squared))

    plt.title('y = {}x + {}'.format(m, b))
    plt.xlabel('')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def regression(xs, ys):
    m = ( (mean(xs)*mean(ys) - mean(xs*ys))  /
        (mean(xs)**2 - mean(xs**2)) )
    b = mean(ys) - (m*mean(xs))
    return m, b

def squared_error(y_points, regression_line):
    return sum( (regression_line - y_points)**2)
def coeff_of_determination(y_points, regression_line):
    y_mean_line = [mean(y_points) for y in regression_line]
    squared_error_regr = squared_error(y_points, regression_line)
    squared_error_y_mean = squared_error(y_points, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)
if __name__ == '__main__':
    main()

