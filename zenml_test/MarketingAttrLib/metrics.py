import pandas as pd

class AttributionMetrics:
    def __init__(self, interactions_data, attribution_data):
        '''
        Initialize the AttributionMetrics instance.
        Args:
            interactions_data (DataFrame): A DataFrame containing each interaction with columns like 'cookie', 'time', 'interaction', 'conversion', 'conversion_value', 'channel', 'cost'.
            attribution_data (DataFrame): A DataFrame containing attribution results. This can be either a single column or multiple columns.
        '''
        self.interactions_data = interactions_data
        self.attribution_data = attribution_data
        self.metrics = pd.DataFrame()

    def calculate_metrics(self):

        # REVISAR INTERACCIONES Y CONVERSIONES POR CANAL
        
        self.interactions_data['time'] = pd.to_datetime(self.interactions_data['time'])

        # Total conversions by channel
        conversions_by_channel = self.interactions_data[self.interactions_data['conversion'] == 1].groupby('channel')['conversion'].count()

        # Total interactions by channel
        total_interactions = self.interactions_data.groupby('channel').size()

        # Conversion Rate by Channel
        self.metrics['Conversion Rate by Channel'] = conversions_by_channel / total_interactions

        # Number of Touchpoints Before Conversion
        conversion_paths = self.interactions_data[self.interactions_data['conversion'] == 1].groupby('cookie').count()
        self.metrics['Avg. Touchpoints Before Conversion'] = conversion_paths['interaction'].mean()

        # Time Until Conversion
        conversion_times = self.interactions_data.groupby('cookie').apply(lambda x: (x[x['conversion'] == 1]['time'].min() - x['time'].min()).total_seconds() / 3600)
        self.metrics['Avg. Time Until Conversion (hours)'] = conversion_times.mean()

        # Join attribution results
        self.metrics = self.metrics.join(self.attribution_data, how='outer')

        # Cost and conversion value analysis per channel
        # Revisar ROI, no puede dar negativo

        self.metrics['Total Cost'] = self.interactions_data.groupby('channel')['cost'].sum()
        self.metrics['Total Conversion Value'] = self.interactions_data[self.interactions_data['conversion'] == 1].groupby('channel')['conversion_value'].sum()
        self.metrics['ROI'] = (self.metrics['Total Conversion Value'] - self.metrics['Total Cost']) / self.metrics['Total Cost']

        # Replace infinite values and NaNs from calculations
        self.metrics.replace([float('inf'), float('-inf'), pd.NA], 0, inplace=True)

    def get_metrics(self):
        '''
        Returns the DataFrame containing all calculated metrics.
        '''
        return self.metrics

    def display_metrics(self):
        '''
        Displays the metrics in a readable format.
        '''
        print("Attribution Metrics Summary:")
        print(self.metrics)



