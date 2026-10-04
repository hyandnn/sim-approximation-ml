"""
Filename: module_data_postprocessing.py
Author: Haoling Yang
Date Created: September 15, 2024

Description:
    The PostProcessor class handles the post-processing of model training results,
    including loading MAE data and training times for different tuner types, and
    generating visualizations such as MAE plots and comparisons of MAE vs. training time.
    It also allows for the comparison of results across multiple tuner types.

Usage:
    To process and visualize the results, instantiate the PostProcessor class and
    call the `run` method, which handles all steps from loading trial data to
    generating plots.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt


class PostProcessor:
    def __init__(self, tuner_types=None, colors=None, titles=None):
        """
        Initialize the PostProcessor with optional tuner types, colors for plotting, and titles.

        Args:
            tuner_types (list): List of tuner types used in the AutoML process.
            colors (list): List of colors to use for plotting each tuner type.
            titles (list): List of titles to use for each tuner type plot.
        """
        self.tuner_types = tuner_types if tuner_types else ['random', 'hyperband', 'greedy', 'bayesian']
        self.colors = colors if colors else ['red', 'green', 'orange', 'blue']
        self.titles = titles if titles else ['Random Search', 'Hyperband', 'Greedy', 'Bayesian optimization']

        # Dictionaries to store the trial MAE, training times, and the best MAE for each tuner type
        self.trial_mae = {tuner_type: [] for tuner_type in self.tuner_types}
        self.training_times = {tuner_type: [] for tuner_type in self.tuner_types}
        self.best_mae = {tuner_type: None for tuner_type in self.tuner_types}

    def load_trial_data(self, tuner_type):
        """
        Load trial data for a specific tuner type by reading JSON files.

        Args:
            tuner_type (str): The type of tuner for which to load trial data.
        """
        project_dir = f'project_{tuner_type}'
        if os.path.exists(project_dir):
            # Read training time data from JSON file
            training_time_path = os.path.join(project_dir, 'training_time.json')
            if os.path.exists(training_time_path):
                with open(training_time_path, 'r') as f:
                    training_time_data = json.load(f)
                    self.training_times[tuner_type].append(training_time_data['training_time'])

            # Iterate through the trials and load their MAE values
            for trial_dir in sorted(os.listdir(project_dir)):
                trial_path = os.path.join(project_dir, trial_dir, 'trial.json')
                if os.path.exists(trial_path):
                    try:
                        with open(trial_path, 'r') as f:
                            trial_data = json.load(f)
                            val_loss_observations = trial_data.get('metrics', {}).get('metrics', {}).get('val_loss',
                                                                                                         {}).get(
                                'observations', [])
                            if val_loss_observations:
                                val_loss_value = val_loss_observations[0].get('value', [])
                                if val_loss_value:
                                    self.trial_mae[tuner_type].append(val_loss_value[0])
                                    # Update the best MAE if a lower value is found
                                    if self.best_mae[tuner_type] is None or val_loss_value[0] < self.best_mae[
                                        tuner_type]:
                                        self.best_mae[tuner_type] = val_loss_value[0]
                    except json.JSONDecodeError:
                        pass

    def process_all_trials(self):
        """
        Process all trial data for each tuner type by calling load_trial_data.
        Raises a ValueError if no valid MAE data is found.
        """
        for tuner_type in self.tuner_types:
            self.load_trial_data(tuner_type)

        # Raise an error if no valid MAE data is available for any tuner type
        if all(len(mae_list) == 0 for mae_list in self.trial_mae.values()):
            raise ValueError("No valid MAE data found in any of the trial folders.")

    def plot_mae(self):
        """
        Plot the Mean Absolute Error (MAE) for each tuner type across trials.
        Saves the plot as 'mae_plot_diff.png'.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for ax, tuner_type, title, color in zip(axes.flatten(), self.tuner_types, self.titles, self.colors):
            mae_list = self.trial_mae[tuner_type]
            if mae_list:
                ax.plot(range(len(mae_list)), mae_list, color=color)
                ax.set_title(title)
                ax.set_xlabel('Trials')
                ax.set_ylabel('MAE')
                ax.set_ylim(0, max(mae_list) + 0.1)
            else:
                ax.set_title(f"{title} (No data)")
                ax.set_xlabel('Trials')
                ax.set_ylabel('MAE')
                ax.set_ylim(0, 1)  # Set default y-axis range
        plt.tight_layout()
        plt.savefig('mae_plot_diff.png', dpi=300)
        # plt.show()  # Optionally display the plot

    def plot_mae_time_comparison(self):
        """
        Plot the comparison of the best MAE and average training time for each tuner type.
        Saves the plot as 'mae_time_comparison.png'.
        """
        mae_values = [self.best_mae[tuner_type] for tuner_type in self.tuner_types]
        time_values = [np.mean(self.training_times[tuner_type]) for tuner_type in self.tuner_types]

        fig, ax1 = plt.subplots()

        bar_width = 0.35
        index = np.arange(len(self.tuner_types))

        # Plot MAE values as bars
        bar1 = ax1.bar(index - bar_width / 2, mae_values, bar_width, label='MAE', color='b', align='center')
        ax1.set_xlabel('Tuner Type')
        ax1.set_ylabel('MAE', color='b')
        ax1.set_xticks(index)
        ax1.set_xticklabels(self.tuner_types)
        ax1.set_ylim(0, max(mae_values) * 1.2)  # Enlarge y-axis range for MAE

        for i, v in enumerate(mae_values):
            ax1.text(i - bar_width / 2 - 0.02, v + 0.005, f'{v:.4f}', ha='center', color='b')

        # Plot training time values as bars on a second y-axis
        ax2 = ax1.twinx()
        bar2 = ax2.bar(index + bar_width / 2, time_values, bar_width, label='Time', color='skyblue', align='center')
        ax2.set_ylabel('Search time for trials [s]', color='skyblue')
        ax2.set_ylim(0, max(time_values) * 1.2)  # Enlarge y-axis range for time

        for i, v in enumerate(time_values):
            ax2.text(i + bar_width / 2, v + 0.005, f'{v:.1f}', ha='center', color='skyblue')

        # Place legends within the plot
        ax1.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=1)
        ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.9), ncol=1)

        plt.title('MAE and Training Time Comparison')
        plt.tight_layout()
        plt.savefig('mae_time_comparison.png', dpi=300)

    def run(self):
        """
        Execute the entire post-processing workflow:
        - Process trial data for each tuner type.
        - Plot MAE results.
        - Plot MAE vs. training time comparison.
        """
        self.process_all_trials()
        self.plot_mae()
        self.plot_mae_time_comparison()


if __name__ == "__main__":
    post_processor = PostProcessor()
    post_processor.run()

