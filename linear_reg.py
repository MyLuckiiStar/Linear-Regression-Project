import numpy as np
from matplotlib import pyplot as plt

#amazon closing stock prices from April 27 to June 22 
x_values = np.array([1,7,14,21,28,35,42,49,56])
y_values = np.array([2286.04,2379.61,2409.78,2436.88,2442.37,2483.00,2545.02,2675.01,2692.87])

#finding the regression line
def best_fit_line(x_values, y_values):
    m = (((x_values.mean() * y_values.mean()) - (x_values * y_values).mean()) / ((x_values.mean()) ** 2 - (x_values ** 2).mean()))
    b = y_values.mean() - m * x_values.mean()

    return m, b
m, b = best_fit_line(x_values, y_values)
print(f"regression line: y = {round(m, 2)}x + {round(b, 2)}")

#predicting a coordinate
x_prediction = float(input("What x value do you want to find the coordinates of: "))
y_prediction = (m * x_prediction) + b 
print(f"predicted coordinate: ({round(x_prediction, 2)}, {round(y_prediction, 2)})")

#determining r^2 value
regression_line = [(m * x) + b for x in x_values]

def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))
def r_squared_value(ys_orig, ys_line):
    squared_error_reg = squared_error(ys_orig, ys_line)
    y_mean_line = [ys_orig.mean() for y in ys_orig]
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_reg / squared_error_y_mean)

r_squared = r_squared_value(y_values, regression_line)
print(f"r^2 value: {round(r_squared, 2)} \nHint: The closer the value is to 1 the better fit of your regression line.")

#plotting the regression model
plt.title("Linear Regression of Amazon Closing Stock Prices (Apr 27 - June 22)")
plt.scatter(x_values, y_values, color = 'pink', label = 'Data')
plt.scatter(x_prediction, y_prediction, color = 'blue', label = 'Predicted')
plt.plot(x_values, regression_line, color = 'green', label = 'Regression Line')
plt.legend()
plt.savefig("linregamazon.png")