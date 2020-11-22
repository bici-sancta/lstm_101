
def rmse(y, y_hat):

    root_mean_squared_error = ((y_hat - y) ** 2).mean() ** .5
    return root_mean_squared_error
