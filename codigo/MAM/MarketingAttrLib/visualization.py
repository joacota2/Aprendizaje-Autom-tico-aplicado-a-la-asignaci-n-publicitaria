import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Optional: Set a style
plt.style.use('seaborn-darkgrid')  # This style has gridlines and a modern look

import matplotlib.pyplot as plt
import seaborn as sns

class AttributionVisualizer:
    def __init__(self, attribution_results):
        """
        Initializes the AttributionVisualizer with the results from AttributionModel.
        
        :param attribution_results: DataFrame containing the attribution results.
        """
        self.attribution_results = attribution_results

    def plot_bar_chart(self):
        """
        Plots bar charts for each attribution method results.
        """
        # Determine how many columns/attributions methods are present
        num_columns = len(self.attribution_results.columns)
        
        if num_columns == 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            column_name = self.attribution_results.columns[0]
            sns.barplot(x=self.attribution_results.index, y=self.attribution_results[column_name], ax=ax)
            ax.set_title(f'{column_name.replace("_", " ").capitalize()} Attribution')
            ax.set_ylabel('Conversions')
            ax.set_xlabel('Channels')
        else:
            fig, axes = plt.subplots(nrows=num_columns, figsize=(10, 6 * num_columns))
            for ax, column in zip(axes.flat, self.attribution_results.columns):
                sns.barplot(x=self.attribution_results.index, y=self.attribution_results[column], ax=ax)
                ax.set_title(f'{column.replace("_", " ").capitalize()} Attribution')
                ax.set_ylabel('Conversions')
                ax.set_xlabel('Channels')

        plt.xticks(rotation=45)  # Optional: rotate labels for better readability
        plt.tight_layout()
        plt.show()
    

    def plot_pie_chart(self):
        """
        Plots pie charts for each attribution method results dynamically based on DataFrame columns.
        """
        num_columns = len(self.attribution_results.columns)
        
        if num_columns == 1:
            fig, ax = plt.subplots()
            column_name = self.attribution_results.columns[0]
            ax.pie(self.attribution_results[column_name], labels=self.attribution_results.index, autopct='%1.1f%%', startangle=140)
            ax.set_title(f'{column_name.replace("_", " ").capitalize()} Attribution')
        else:
            fig, axes = plt.subplots(nrows=1, ncols=num_columns, figsize=(15, 6))
            for ax, column in zip(axes, self.attribution_results.columns):
                ax.pie(self.attribution_results[column], labels=self.attribution_results.index, autopct='%1.1f%%', startangle=140)
                ax.set_title(f'{column.replace("_", " ").capitalize()} Attribution')

        plt.tight_layout()
        plt.show()

    def plot_comparison(self, scale_data=True):
        """
        Creates a comparison bar chart for all attribution methods, with optional scaling.
        
        :param scale_data: Boolean, whether to scale data to a common range (default: True)
        """
        data = self.attribution_results.copy()
        
        if scale_data:
            # Scale the data to the range 0-1
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data)
            data = pd.DataFrame(data_scaled, columns=data.columns, index=data.index)

        # Melt the data for seaborn's barplot
        melted_data = data.reset_index().melt(id_vars='index', var_name='Method', value_name='Conversions')
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=melted_data, x='index', y='Conversions', hue='Method')
        plt.title('Comparison of Attribution Methods (Scaled)' if scale_data else 'Comparison of Attribution Methods')
        plt.ylabel('Conversions (Scaled)' if scale_data else 'Conversions')
        plt.xlabel('Channels')
        plt.xticks(rotation=45)
        plt.legend(title='Attribution Method')
        plt.tight_layout()
        plt.show()
