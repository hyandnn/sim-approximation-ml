"""
Filename: main.py
Author: Haoling Yang
Date Created: September 15, 2024

Description:
    This file contains the implementation of a GUI for managing the entire
    AutoML workflow, including data preprocessing, model training, evaluation,
    prediction, and post-processing. The GUI allows users to select JSON files
    for processing, configure settings for preprocessing and model training,
    and visualize model results through images and metrics.

    The GUI is built using the `tkinter` library and provides an easy-to-use
    interface for handling machine learning tasks without needing to directly
    interact with the codebase.

Usage:
    To run the GUI, simply execute this file. The user will be prompted to
    interact with the interface to complete various tasks.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
from module_data_preprocessing import DataPreprocessor
from module_automl import AutoMLRegressor
from module_data_postprocessing import PostProcessor
import pandas as pd
import numpy as np
import pickle
from keras.models import load_model


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.grid(padx=20, pady=20)
        self.file_usages = []
        self.train_file_indices = []
        self.test_file_indices = []
        self.create_widgets()
        self.automl_regressor = None  # Holds an instance of AutoMLRegressor
        self.selected_model = None
        self.loaded_model = None
        self.image_label = tk.Label(self)  # Label to display the image
        self.image_label.grid(row=0, column=5, rowspan=20, padx=20, pady=20, sticky="n")

    def create_widgets(self):
        button_width = 20
        button_height = 1
        padx = 5
        pady = 5

        self.select_file_button = tk.Button(self, text="Select JSON Files", command=self.select_files,
                                            width=button_width, height=button_height)
        self.select_file_button.grid(row=0, column=0, padx=padx, pady=pady)

        self.preprocess_button = tk.Button(self, text="Data Preprocessing", command=self.show_preprocessing_options,
                                           width=button_width, height=button_height)
        self.preprocess_button.grid(row=1, column=0, padx=padx, pady=pady)

        self.train_button = tk.Button(self, text="Model Training", command=self.run_training, width=button_width,
                                      height=button_height)
        self.train_button.grid(row=2, column=0, padx=padx, pady=pady)

        self.evaluate_button = tk.Button(self, text="Model Evaluation", command=self.run_evaluation, width=button_width,
                                         height=button_height)
        self.evaluate_button.grid(row=3, column=0, padx=padx, pady=pady)

        self.predict_button = tk.Button(self, text="Predict", command=self.setup_prediction, width=button_width,
                                        height=button_height)
        self.predict_button.grid(row=5, column=0, padx=padx, pady=pady)

        self.postprocess_button = tk.Button(self, text="Post Processing", command=self.run_postprocessing,
                                            width=button_width, height=button_height)
        self.postprocess_button.grid(row=4, column=0, padx=padx, pady=pady)

        self.display_image_button = tk.Button(self, text="Display Image", command=self.select_and_display_image,
                                              width=button_width, height=button_height)
        self.display_image_button.grid(row=6, column=0, padx=padx, pady=pady)

        self.quit = tk.Button(self, text="QUIT", fg="red", command=self.master.destroy, width=button_width,
                              height=button_height)
        self.quit.grid(row=7, column=0, padx=padx, pady=pady)

        self.create_input_fields()

    def create_input_fields(self):
        self.input_entries = {}
        self.output_entries = {}
        self.input_params = ['AmountServer', 'Coolingdefect', 'InterarrivalTime', 'defekteModulanzahl']
        self.output_params = [
            'AverageServerUtilisation', 'AverageFlowTime', 'OEE', 'TotalAverageQueueLength',
            'ProcessingTimeAverage', 'WaitingTimeAverage', 'MovingTimeAverage', 'FailedTimeAverage',
            'BlockedTimeAverage', 'Throughput'
        ]

        entry_width = 10
        padx = 5
        pady = 5

        # Creating input fields for input parameters
        for i, param in enumerate(self.input_params):
            label = tk.Label(self, text=param)
            label.grid(row=i, column=1, padx=padx, pady=pady, sticky="e")
            entry = tk.Entry(self, width=entry_width)
            entry.grid(row=i, column=2, padx=padx, pady=pady)
            self.input_entries[param] = entry

        # Creating input fields for output parameters
        for i, param in enumerate(self.output_params):
            label = tk.Label(self, text=param)
            label.grid(row= i, column=3, padx=padx, pady=pady, sticky="e")
            entry = tk.Entry(self, state='readonly', width=entry_width)
            entry.grid(row= i, column=4, padx=padx, pady=pady)
            self.output_entries[param] = entry

    def select_files(self):
        self.file_usages = []
        self.train_file_indices = []
        self.test_file_indices = []
        index = 1

        while True:
            file_path = filedialog.askopenfilename(title="Select JSON File", filetypes=[("JSON files", "*.json")])
            if file_path:
                # Use ttk.Combobox to select the usage of the file
                usage_window = tk.Toplevel(self)
                usage_window.title("Select File Usage")
                usage_label = tk.Label(usage_window, text=f"Select the usage for this file:")
                usage_label.grid(row=0, column=0, padx=5, pady=5)

                usage_var = tk.StringVar(value="train")  # Default value is 'train'
                usage_menu = ttk.Combobox(usage_window, textvariable=usage_var)
                usage_menu['values'] = ("train", "test", "split")
                usage_menu.grid(row=1, column=0, padx=5, pady=5)

                def confirm_selection():
                    usage = usage_var.get()
                    self.file_usages.append((file_path, usage, index))
                    if usage == 'train' or usage == 'split':
                        self.train_file_indices.append(index)
                    if usage == 'test' or usage == 'split':
                        self.test_file_indices.append(index)
                    usage_window.destroy()

                confirm_button = tk.Button(usage_window, text="Confirm", command=confirm_selection)
                confirm_button.grid(row=2, column=0, padx=5, pady=5)

                usage_window.wait_window()  # Wait for the child window to close before continuing
                index += 1  # Increment the index after confirming the usage

            else:
                break

            continue_selection = messagebox.askyesno("Continue", "Do you want to select another file?")
            if not continue_selection:
                break

        if self.file_usages:
            messagebox.showinfo("Files Selected", "Selected files and their usages have been recorded.")
        else:
            messagebox.showwarning("No File Selected", "No files were selected.")

    def show_preprocessing_options(self):
        # Create a new window for selecting Scaler
        options_window = tk.Toplevel(self)
        options_window.title("Preprocessing Options")

        # Scaler selection
        scaler_label = tk.Label(options_window, text="Choose Scaler:")
        scaler_label.grid(row=0, column=0, padx=5, pady=5)

        scaler_var = tk.StringVar(value="MinMaxScaler")  # Default value is MinMaxScaler
        scaler_menu = ttk.Combobox(options_window, textvariable=scaler_var)
        scaler_menu['values'] = ("StandardScaler", "MinMaxScaler")
        scaler_menu.grid(row=1, column=0, padx=5, pady=5)

        def confirm_selection():
            scaler_type = scaler_var.get()
            self.run_preprocessing(scaler_type)
            options_window.destroy()

        confirm_button = tk.Button(options_window, text="Confirm", command=confirm_selection)
        confirm_button.grid(row=2, column=0, padx=5, pady=5)

    def run_preprocessing(self, scaler_type):
        if not self.file_usages:
            messagebox.showwarning("No Files Selected", "Please select JSON files first.")
            return
        try:
            input_params = ['AmountServer', 'Coolingdefect', 'InterarrivalTime', 'defekteModulanzahl']
            output_params = [
                'AverageServerUtilisation', 'AverageFlowTime', 'OEE', 'TotalAverageQueueLength',
                'ProcessingTimeAverage', 'WaitingTimeAverage', 'MovingTimeAverage', 'FailedTimeAverage',
                'BlockedTimeAverage', 'Throughput'
            ]

            for file_path, usage, index in self.file_usages:
                preprocessor = DataPreprocessor(file_path, input_params, output_params, scaler_type=scaler_type)
                preprocessor.run(usage, index)

            messagebox.showinfo("Success", "Data Preprocessing Completed Successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Data Preprocessing Failed: {str(e)}")

    def run_training(self):
        try:
            # Gather all training data
            X_train_list = []
            y_train_list = []

            for index in self.train_file_indices:
                data = np.load(f'train_data_{index}.npz')
                X_train_list.append(data['X_train'])
                y_train_list.append(data['y_train'])

            # Concatenate all training data
            X_train = np.concatenate(X_train_list, axis=0)
            y_train = np.concatenate(y_train_list, axis=0)

            with open('scaler_X.pkl', 'rb') as f:
                scaler_X = pickle.load(f)
            with open('scaler_y.pkl', 'rb') as f:
                scaler_y = pickle.load(f)

            self.automl_regressor = AutoMLRegressor(train_data_path=None, scaler_x_path=None, scaler_y_path=None)
            self.automl_regressor.X_train = X_train
            self.automl_regressor.y_train = y_train
            self.automl_regressor.scaler_X = scaler_X
            self.automl_regressor.scaler_y = scaler_y

            self.automl_regressor.run_train_only()  # Run training only
            messagebox.showinfo("Success", "Model Training Completed Successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Model Training Failed: {str(e)}")

    def run_evaluation(self):
        try:
            # Gather all test data
            X_test_list = []
            y_test_list = []

            for index in self.test_file_indices:
                data = np.load(f'test_data_{index}.npz')
                X_test_list.append(data['X_test'])
                y_test_list.append(data['y_test'])

            # Concatenate all test data
            X_test = np.concatenate(X_test_list, axis=0)
            y_test = np.concatenate(y_test_list, axis=0)

            if not self.automl_regressor:
                # If self.automl_regressor is empty, create a new instance without training
                with open('scaler_X.pkl', 'rb') as f:
                    scaler_X = pickle.load(f)
                with open('scaler_y.pkl', 'rb') as f:
                    scaler_y = pickle.load(f)
                self.automl_regressor = AutoMLRegressor(scaler_x_path=None, scaler_y_path=None)
                self.automl_regressor.scaler_X = scaler_X
                self.automl_regressor.scaler_y = scaler_y

            self.automl_regressor.X_test = X_test
            self.automl_regressor.y_test = y_test

            # Perform evaluation directly
            self.automl_regressor.run_evaluation_only(X_test, y_test)  # Pass concatenated data for evaluation
            messagebox.showinfo("Success", "Model Evaluation Completed Successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Model Evaluation Failed: {str(e)}")

    def run_postprocessing(self):
        try:
            post_processor = PostProcessor()
            post_processor.run()
            messagebox.showinfo("Success", "Post-processing Completed Successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Post-processing Failed: {str(e)}")

    def select_and_display_image(self):
        image_path = filedialog.askopenfilename(title="Select Image File",
                                                filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if image_path:
            self.display_image(image_path)
        else:
            messagebox.showwarning("No File Selected", "Please select an image file.")

    def display_image(self, image_path):
        try:
            image = Image.open(image_path)
            # Resize the image
            image = image.resize((600, 400), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo  # Keep a reference to the image
            messagebox.showinfo("Success", "Image Displayed Successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Displaying Image Failed: {str(e)}")

    def setup_prediction(self):
        # Create a new window for selecting the model
        model_window = tk.Toplevel(self)
        model_window.title("Select Model")

        # Model selection
        model_label = tk.Label(model_window, text="Choose Model:")
        model_label.grid(row=0, column=0, padx=5, pady=5)

        model_var = tk.StringVar(value="random")  # Default value is 'random'
        model_menu = ttk.Combobox(model_window, textvariable=model_var)
        model_menu['values'] = ['random', 'hyperband', 'greedy', 'bayesian']
        model_menu.grid(row=1, column=0, padx=5, pady=5)

        def confirm_selection():
            self.selected_model = model_var.get()
            messagebox.showinfo("Model Selected", f"Selected model: {self.selected_model}")
            model_window.destroy()

            # Ensure AutoMLRegressor is properly initialized
            if not self.automl_regressor:
                with open('scaler_X.pkl', 'rb') as f:
                    scaler_X = pickle.load(f)
                with open('scaler_y.pkl', 'rb') as f:
                    scaler_y = pickle.load(f)
                self.automl_regressor = AutoMLRegressor(scaler_x_path=None, scaler_y_path=None)
                self.automl_regressor.scaler_X = scaler_X
                self.automl_regressor.scaler_y = scaler_y


            # Create input and output fields for prediction
            self.create_input_fields()

            # Add a button for running prediction
            self.run_predict_button = tk.Button(self, text="Run Prediction", command=self.run_prediction, width=20,
                                                height=2)
            self.run_predict_button.grid(row=len(self.input_params) + len(self.output_params), column=0, padx=5, pady=5)

        confirm_button = tk.Button(model_window, text="Confirm", command=confirm_selection)
        confirm_button.grid(row=2, column=0, padx=5, pady=5)


    def run_prediction(self):
        try:
            input_data = []
            for param, entry in self.input_entries.items():
                value = float(entry.get())
                input_data.append(value)

            # Create a pandas DataFrame using the column names
            input_data_df = pd.DataFrame([input_data], columns=self.input_params)

            # Transform using DataFrame to keep feature names consistent
            input_data_normalized = self.automl_regressor.scaler_X.transform(input_data_df)

            # Use AutoMLRegressor's predict method to make predictions
            predictions_inverse = self.automl_regressor.predict(self.selected_model, input_data_normalized)

            # Inverts the prediction back to the original scale
            for i, param in enumerate(self.output_params):
                self.output_entries[param].config(state='normal')
                self.output_entries[param].delete(0, tk.END)
                self.output_entries[param].insert(0, f"{predictions_inverse[0][i]:.4f}")
                self.output_entries[param].config(state='readonly')

            messagebox.showinfo("Success", "Prediction Completed Successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction Failed: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Automated Machine Learning Workflow")
    app = Application(master=root)
    app.mainloop()


