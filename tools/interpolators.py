def fast_interpolation(x, y, x_new):
    """
    Fast interpolation function for 1D data.

    Parameters:
    x : The x-coordinates of the data points.
    y : The y-coordinates of the data points.
    x_new : The new x-coordinate at which to interpolate.

    Returns:
    The interpolated y-coordinate corresponding to x_new.
    """
    if x_new <= x[0]:
        return y[0]
    elif x_new >= x[-1]:
        return y[-1]
    
    # Binary search to find the correct interval
    low = 0
    high = len(x) - 1
    
    while low <= high:
        mid = (low + high) // 2
        
        if x[mid] < x_new:
            low = mid + 1
        elif x[mid] > x_new:
            high = mid - 1
        else:
            return y[mid]
    
    # Linear interpolation
    x0, x1 = x[high], x[low]
    y0, y1 = y[high], y[low]
    
    return y0 + (y1 - y0) * ((x_new - x0) / (x1 - x0))