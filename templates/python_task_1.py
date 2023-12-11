import pandas as pd


def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Write your logic here
    df=df.pivot(index='id_1',columns='id_2',values='car')
    df.fillna(0,inplace=True)

    return df


def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Write your logic here
    car_types=df['car'].unique().tolist()
    car_type_counts={}
    for car_type in car_types:
        car_type_counts[car_type] = df['car'].eq(car_type).sum()

    return car_type_counts


def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Write your logic here
    bus_mean = df['bus'].mean()
    bus_gt_twice_mean = df['bus'].gt(2 * bus_mean)

    return df[bus_gt_twice_mean].index.tolist()



def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Write your logic here
    truck_mean = df['truck'].mean()
    truck_gt_mean = df['truck'].gt(7 * truck_mean)

    return df[truck_gt_mean].index.tolist()


def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Write your logic here

    matrix['bus'] = matrix['bus'].apply(lambda x: x * 2 if x > bus_threshold else x)
    matrix['truck'] = matrix['truck'].apply(lambda x: x * 3 if x > truck_threshold else x)

    return matrix


def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here

    time_window = pd.Timedelta(days=7)
    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()

    # Create a DatetimeIndex for 7 days with 1 hour resolution
    date_range = pd.date_range(start=start_time, end=end_time, freq='1H')

    # Count the occurrence of each unique timestamp in the dataframe
    timestamp_count = df['timestamp'].value_counts().sort_index()

    # Create a boolean series to indicate whether the timestamps for each unique timestamp cover a full 24-hour and 7 days period
    is_complete = pd.Series(data=False, index=date_range)
    is_complete[timestamp_count.index] = timestamp_count >= 24

    return is_complete
