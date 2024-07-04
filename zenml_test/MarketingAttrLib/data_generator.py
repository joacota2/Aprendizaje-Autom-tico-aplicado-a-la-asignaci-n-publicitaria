import pandas as pd
import numpy as np


class MamDatasets:
    # costos 0,15 
    def __init__(self, data=None, start_date="2022-01-01", interaction_probs=[0.3, 0.6, 0.1], conversion_rates={'click': 0.2, 'impression': 0.02, 'conversion': 1.0}, cost_params=(5.50, 1.0), conversion_value_params=(50, 10)):
        self.data = data if data is not None else pd.DataFrame()
        self.start_date = start_date
        self.interaction_probs = interaction_probs
        self.conversion_rates = conversion_rates
        self.cost_params = cost_params
        self.conversion_value_params = conversion_value_params

    def generate_data(self, num_channels=5, num_entries=1000, num_cookies=100, columns_to_generate=None):
        if self.data.empty:
            num_entries_to_generate = num_entries
        else:
            num_entries_to_generate = len(self.data)
        
        columns = {
            'time': lambda n: pd.date_range(start=self.start_date, periods=n, freq="H").to_numpy(),
            'cookie': lambda n: [f"cookie_{i+1}" for i in np.random.randint(1, num_cookies + 1, size=n)],
            'interaction': lambda n: np.random.choice(['click', 'impression', 'conversion'], n, p=self.interaction_probs),
            'conversion': lambda n: [np.random.binomial(1, self.conversion_rates[x]) for x in self.data.get('interaction', np.random.choice(['click', 'impression', 'conversion'], n, p=self.interaction_probs))],
            'cost': lambda n: np.random.normal(self.cost_params[0], self.cost_params[1], size=n),
            'conversion_value': lambda n: [np.random.normal(self.conversion_value_params[0], self.conversion_value_params[1]) if x else 0 for x in self.data.get('conversion', [np.random.binomial(1, self.conversion_rates[x]) for x in self.data.get('interaction', np.random.choice(['click', 'impression', 'conversion'], n, p=self.interaction_probs))])],
            'channel': lambda n: [f"Channel {i+1}" for i in np.random.randint(1, num_channels + 1, size=n)]
        }

        required_columns = set(columns.keys())
        existing_columns = set(self.data.columns)
        missing_columns = required_columns - existing_columns if columns_to_generate is None else set(columns_to_generate) & (required_columns - existing_columns)

        for col in missing_columns:
            self.data[col] = columns[col](num_entries_to_generate)

        # If data was previously loaded with timestamps and the time column is missing, shuffle timestamps
        if 'time' in missing_columns and not self.data.empty:
            np.random.shuffle(self.data['time'].values)

        return self.data

    def dataset_summary(self):
        if self.data.empty:
            return "Data is not generated yet. Please generate data first."

        # Calculate interaction counts and ensure it's formatted as a DataFrame
        interaction_counts = self.data['interaction'].value_counts().rename_axis('Interaction Type').reset_index(name='Count')

        # Convert interaction_counts into a dictionary with interaction types as keys and counts as values
        interaction_dict = interaction_counts.set_index('Interaction Type')['Count'].to_dict()

        # Create a single summary dictionary including interactions
        summary = {
            "Total Entries": len(self.data),
            "Unique Cookies": self.data['cookie'].nunique(),
            "Average Cost": self.data['cost'].mean(),
            "Average Conversion Value": self.data['conversion_value'][self.data['conversion'] == 1].mean(),
            "Total Conversion Value": self.data['conversion_value'].sum(),
            "Total Cost": self.data['cost'].sum(),
            "Conversion Rate": self.data['conversion'].mean()
        }

        # Update the summary dictionary with interaction counts
        summary.update(interaction_dict)

        # Create a DataFrame from the updated summary dictionary
        summary_df = pd.DataFrame([summary])

        return summary_df

    def interactions_per_channel(self, interaction_type='conversion'):
        if self.data.empty:
            return "Data is not generated yet. Please generate data first."

        filtered_data = self.data[self.data['interaction'] == interaction_type]
        interactions_df = filtered_data.groupby('channel').size().reset_index(name=f'{interaction_type} Count')
        return interactions_df



# Example usage:
# mam_dataset = MamDatasets()
# data = mam_dataset.generate_data()
# print(data)

# summary = mam_dataset.dataset_summary()
# print(summary)

# interactions = mam_dataset.interactions_per_channel(interaction_type='conversion')
# print(interactions)
# print(mam_dataset.data.head())