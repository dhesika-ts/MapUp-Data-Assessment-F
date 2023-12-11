import pandas as pd
import numpy as np

def calculate_distance_matrix(df):
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """

    # Find unique start and end locations
    start_locations = df['id_start'].unique()
    end_locations = df['id_end'].unique()

    # Initialize a new distance matrix with dimensions len(start_locations) x len(end_locations)
    distance_matrix = pd.DataFrame(
        index=start_locations,
        columns=end_locations,
        data=np.nan
    )

    # Iterate over the dataframe and populate the distance matrix
    for index, row in df.iterrows():
        start_location = row['id_start']
        end_location = row['id_end']
        distance = row['distance']

        distance_matrix.loc[start_location, end_location] = distance

    return distance_matrix

# Load the dataset into a pandas DataFrame
df = pd.read_csv('dataset3.csv')

# Calculate the distance matrix
distance_matrix = calculate_distance_matrix(df)

# Display the distance matrix
print(distance_matrix)


def unroll_distance_matrix(df):
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """

    # Create an empty DataFrame to store the unrolled data
    unrolled_df = pd.DataFrame(columns=['id_start', 'id_end', 'distance'])

    # Find all the unique id_start and id_end values
    unique_ids = df.index.union(df.columns)

    # Iterate over all the possible combinations of id_start and id_end
    for id_start in unique_ids:
        for id_end in unique_ids:
            # Skip the case where id_start is equal to id_end
            if id_start == id_end:
                continue

            # Look up the corresponding distance value in the distance matrix
            distance = df.loc[id_start, id_end]

            # Append the current combination of id_start, id_end, and distance to the unrolled DataFrame
            unrolled_df = unrolled_df.append({'id_start': id_start, 'id_end': id_end, 'distance': distance}, ignore_index=True)

    return unrolled_df

# Calculate the distance matrix
distance_matrix = calculate_distance_matrix(df)

# Unroll the distance matrix
unrolled_df = unroll_distance_matrix(distance_matrix)

# Display the unrolled DataFrame
print(unrolled_df)


def find_ids_within_ten_percentage_threshold(df, reference_id):
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Calculate the average distance for the reference value
    reference_average_distance = df.loc[reference_id, 'distance'].mean()

    # Define a range that is 10% wider than the reference value's average distance
    lower_bound = reference_average_distance * 0.9
    upper_bound = reference_average_distance * 1.1

    # Find all the values from the id_start column that fall within the specified range
    valid_ids = df.loc[df['distance'].between(lower_bound, upper_bound), 'id_start'].unique()

    # Return a sorted list of the values from the id_start column that meet the criteria
    return sorted(valid_ids)

# Test the function
valid_ids = find_ids_within_ten_percentage_threshold(df, 100)

# Display the valid IDs
print(valid_ids)


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Define the rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Calculate the toll rate for each vehicle type by multiplying the distance
        # by the appropriate rate coefficient
        for vehicle_type, rate_coefficient in rate_coefficients.items():
            df.loc[index, vehicle_type] = row['distance'] * rate_coefficient

    return df

# Test the function
calculate_toll_rate(df)


import datetime


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Define the discount factors for each time interval within a day
    discount_factors = {
        'weekday_early': 0.8,
        'weekday_late': 1.2,
        'weekday_night': 0.8,
        'weekend': 0.7
    }

    # Add columns to the DataFrame to store the start_day, start_time, end_day, and end_time
    df['start_day'] = pd.Categorical(df['start_time'].dt.weekday_name, categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    df['start_time'] = df['start_time'].dt.time
    df['end_day'] = pd.Categorical(df['end_time'].dt.weekday_name, categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    df['end_time'] = df['end_time'].dt.time

    # Define a function to calculate the toll rate based on the time interval
    def calculate_toll_rate(row):
        start_time = row['start_time']
        end_time = row['end_time']
        if row['start_day'] == row['end_day']:
            if start_time < datetime.time(10, 0, 0) and end_time <= datetime.time(10, 0, 0):
                return discount_factors['weekday_early']
            elif start_time < datetime.time(18, 0, 0) and end_time <= datetime.time(18, 0, 0):
                return discount_factors['weekday_late']
            elif start_time < datetime.time(23, 59, 59) and end_time <= datetime.time(23, 59, 59):
                return discount_factors['weekday_night']
        if row['start_day'] in ['Saturday', 'Sunday'] or row['end_day'] in ['Saturday', 'Sunday']:
            return discount_factors['weekend']
        return 1.0

    # Iterate over each row in the DataFrame and calculate the toll rate for each vehicle type
    for column in ['moto', 'car', 'rv', 'bus', 'truck']:
        df[column] = df.apply(calculate_toll_rate, axis=1) * df[column]

    return df

# Test the function
calculate_time_based_toll_rates(df)