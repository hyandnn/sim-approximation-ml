"""
Filename: module_data_preprocessing.py
Author: Haoling Yang
Date Created: September 15, 2024

Description:
    The DataPreprocessor class is responsible for handling data preprocessing tasks,
    including loading data from JSON files, extracting relevant input and output
    parameters, cleaning missing values, and scaling the data using various scalers
    (e.g., MinMaxScaler, StandardScaler). It saves the processed data for use in
    machine learning tasks.

Usage:
    Preprocessing can be run by instantiating the DataPreprocessor class with the
    desired parameters and calling the `run` method.
"""

import json
import ast
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging


class DataPreprocessor:
    def __init__(self, file_path, input_params, output_params, scaler_type='MinMaxScaler'):
        """
        Initialize the DataPreprocessor with file path, input/output parameters, and scaler type.

        Args:
            file_path (str): Path to the JSON data file.
            input_params (list): List of input parameters to extract.
            output_params (list): List of output parameters to extract.
            scaler_type (str): Type of scaler to use ('StandardScaler' or 'MinMaxScaler').
        """
        self.file_path = file_path
        self.input_params = input_params
        self.output_params = output_params
        self.scaler_type = scaler_type
        self.data = self.load_json()  # Load data from the JSON file

        # Initialize logging to record processing steps and issues
        logging.basicConfig(filename='data_preprocessing.log', level=logging.INFO)
        logging.info("Initialized DataPreprocessor with file_path: %s, scaler_type: %s",
                     self.file_path, self.scaler_type)

    def load_json(self):
        """
        Load data from a JSON file and return it.

        Returns:
            dict: Data loaded from the JSON file.

        Raises:
            Exception: If the JSON file cannot be loaded.
        """
        try:
            with open(self.file_path, 'r') as file:
                data = json.load(file)
                logging.info("Successfully loaded JSON data.")
                return data
        except Exception as e:
            logging.error("Failed to load JSON file: %s", str(e))
            raise

    def extract_data(self):
        """
        Extract input and output data based on the specified parameters from the loaded JSON data.

        Returns:
            tuple: Two pandas DataFrames, one for input data and one for output data.
        """
        # Initialize dictionaries for storing input and output data
        input_data = {param: [] for param in self.input_params}
        output_data = {param: [] for param in self.output_params}
        ids = self.data[self.input_params[0]].keys()  # Get all ID keys from the data

        # Iterate over each ID and extract corresponding input and output values
        for id_key in ids:
            valid = True  # Flag to check if the data for this ID is valid
            temp_input_data = {}
            temp_output_data = {}

            # Extract input data for the given ID
            for param in self.input_params:
                value = self.data[param].get(id_key)
                if value is not None:
                    temp_input_data[param] = value
                else:
                    valid = False
                    logging.warning("Missing input parameter '%s' for ID: %s", param, id_key)
                    break

            # If input data is valid, extract output data for the given ID
            if valid:
                for param in self.output_params:
                    if param == 'Throughput':
                        # Extract throughput value and handle special format case
                        throughput_value = self.data[param].get(id_key)
                        if throughput_value is not None:
                            try:
                                temp_output_data[param] = ast.literal_eval(throughput_value)['Sink']
                            except (ValueError, SyntaxError):
                                valid = False
                                logging.error("Invalid format for 'Throughput' at ID: %s", id_key)
                                break
                        else:
                            valid = False
                            logging.warning("Missing 'Throughput' parameter for ID: %s", id_key)
                            break
                    else:
                        # Extract other output data
                        value = self.data[param].get(id_key)
                        if value is not None:
                            temp_output_data[param] = value
                        else:
                            valid = False
                            logging.warning("Missing output parameter '%s' for ID: %s", param, id_key)
                            break

            # If data is valid, append it to the input and output dictionaries
            if valid:
                for param in self.input_params:
                    input_data[param].append(temp_input_data[param])
                for param in self.output_params:
                    output_data[param].append(temp_output_data[param])

        logging.info("Extracted data with %d valid records.", len(input_data[self.input_params[0]]))
        return pd.DataFrame(input_data), pd.DataFrame(output_data)

    def clean_data(self, input_df, output_df):
        """
        Combine input and output data and clean it by removing any rows with missing values.

        Args:
            input_df (DataFrame): DataFrame containing input data.
            output_df (DataFrame): DataFrame containing output data.

        Returns:
            DataFrame: Cleaned combined data.
        """
        # Concatenate input and output data to form a single DataFrame
        combined_data = pd.concat([input_df, output_df], axis=1)
        logging.info("Combined data shape before cleaning: %s", combined_data.shape)

        # Drop rows with missing values
        cleaned_data = combined_data.dropna()
        logging.info("Cleaned data shape after dropping NA: %s", cleaned_data.shape)
        return cleaned_data

    # If you have a big dataset, you can use this new method of clean_data
    def clean_data_new(self, input_df, output_df):
        """
        Combine input and output data, clean it by removing missing values,
        retaining the most frequent output for identical inputs, and removing any duplicates.

        Args:
            input_df (DataFrame): DataFrame containing input data.
            output_df (DataFrame): DataFrame containing output data.

        Returns:
            DataFrame: Fully cleaned combined data, with missing values removed,
                      most frequent output retained for identical inputs, and duplicates removed.
        """
        # Concatenate input and output data to form a single DataFrame
        combined_data = pd.concat([input_df, output_df], axis=1)
        logging.info("Combined data shape before cleaning: %s", combined_data.shape)

        # Step 1: Drop rows with missing values
        cleaned_data = combined_data.dropna()
        logging.info("Cleaned data shape after dropping NA: %s", cleaned_data.shape)

        # Step 2: Retain the most frequent output for identical inputs
        # Group the data by the input columns
        grouped = cleaned_data.groupby(self.input_params)

        # Create a list to store rows to keep
        rows_to_keep = []

        for input_vals, group in grouped:
            # Check if there are multiple outputs for the same inputs
            if group[self.output_params].drop_duplicates().shape[0] > 1:
                # Count the occurrences of each unique output group
                output_group_counts = group[self.output_params].value_counts()

                # Get the most frequent output group
                most_frequent_output = output_group_counts.idxmax()

                # Retain the first occurrence of the most frequent output
                rows_to_keep.append(group[group[self.output_params].eq(most_frequent_output).all(axis=1)].iloc[0])
            else:
                # If there is only one output, keep it
                rows_to_keep.append(group.iloc[0])

        # Recreate the cleaned dataframe with only the rows to keep
        cleaned_data = pd.DataFrame(rows_to_keep)
        logging.info("Cleaned data shape after retaining most frequent outputs: %s", cleaned_data.shape)

        # Step 3: Remove any remaining duplicates
        cleaned_data_dedup = cleaned_data.drop_duplicates()
        logging.info("Cleaned data shape after dropping duplicates: %s", cleaned_data_dedup.shape)

        return cleaned_data_dedup

    def process_data(self, combined_data):
        """
        Scale the input and output data using the specified scaler type (StandardScaler or MinMaxScaler).

        Args:
            combined_data (DataFrame): The cleaned combined input and output data.

        Returns:
            tuple: Two numpy arrays containing the scaled input (X) and output (y) data.
        """
        # Initialize the appropriate scaler based on the scaler type
        if self.scaler_type == 'StandardScaler':
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
        elif self.scaler_type == 'MinMaxScaler':
            self.scaler_X = MinMaxScaler()
            self.scaler_y = MinMaxScaler()

        logging.info("Starting data scaling with %s.", self.scaler_type)

        # Scale the input and output data
        X_processed = self.scaler_X.fit_transform(combined_data[self.input_params])
        y_processed = self.scaler_y.fit_transform(combined_data[self.output_params])
        logging.info("Data scaling completed.")

        return X_processed, y_processed

    def save_data(self, X_processed, y_processed, usage, index):
        """
        Save the processed data to .npz files based on the usage type (train, test, or split).

        Args:
            X_processed (numpy array): Scaled input data.
            y_processed (numpy array): Scaled output data.
            usage (str): Usage type ('train', 'test', or 'split').
            index (int): File index for saving multiple datasets.
        """
        # Save processed data to files based on usage type
        if usage == 'train':
            np.savez(f'train_data_{index}.npz', X_train=X_processed, y_train=y_processed)
        elif usage == 'test':
            np.savez(f'test_data_{index}.npz', X_test=X_processed, y_test=y_processed)
        elif usage == 'split':
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.1,
                                                                random_state=42)
            np.savez(f'train_data_{index}.npz', X_train=X_train, y_train=y_train)
            np.savez(f'test_data_{index}.npz', X_test=X_test, y_test=y_test)
        logging.info(f"{usage.capitalize()} data {index} successfully saved")

    def save_scalers(self):
        """
        Save the fitted scalers to pickle files for later use in model training and evaluation.
        """
        with open('scaler_X.pkl', 'wb') as f:
            pickle.dump(self.scaler_X, f)
        with open('scaler_y.pkl', 'wb') as f:
            pickle.dump(self.scaler_y, f)
        logging.info("Scalers successfully saved")

    def run(self, usage, index):
        """
        Main method to execute the full data preprocessing pipeline: extraction, cleaning, processing, and saving.

        Args:
            usage (str): Usage type ('train', 'test', or 'split').
            index (int): File index for saving multiple datasets.
        """
        try:
            # Extract data from the JSON file
            input_df, output_df = self.extract_data()

            # Clean the data by removing any rows with missing values
            combined_data_cleaned = self.clean_data(input_df, output_df)

            # Scale the data
            X_processed, y_processed = self.process_data(combined_data_cleaned)

            # Save the processed data
            self.save_data(X_processed, y_processed, usage, index)

            # Save scalers only for the first dataset
            if index == 1:
                self.save_scalers()

        except Exception as e:
            logging.error(f"Data Preprocessing Failed: {str(e)}")
            raise
