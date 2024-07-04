import pandas as pd
from itertools import chain, combinations
from math import factorial
from collections import defaultdict



class AttributionModel:
    def __init__(self, marketing_data):
        self.marketing_data = marketing_data
        self.attribution_results = {}
        
    def first_touch(self):
        sorted_data = self.marketing_data.sort_values(by=['cookie', 'time'])
        first_touches = sorted_data.drop_duplicates(subset='cookie', keep='first')
        self.attribution_results['first_touch'] = first_touches[first_touches['conversion'] == 1]['channel'].value_counts()

    def last_touch(self):
        sorted_data = self.marketing_data.sort_values(by=['cookie', 'time'])
        last_touches = sorted_data.drop_duplicates(subset='cookie', keep='last')
        self.attribution_results['last_touch'] = last_touches[last_touches['conversion'] == 1]['channel'].value_counts()

    def linear(self):
        converted_interactions = self.marketing_data[self.marketing_data['conversion'] == 1]
        unique_conversions = converted_interactions.drop_duplicates(subset=['cookie', 'channel'])
        self.attribution_results['linear'] = unique_conversions.groupby('channel').size()

    def time_decay(self, decay_factor=0.5):
        self.marketing_data = self.marketing_data.sort_values(by=['cookie', 'time'])
        decay_scores = self.marketing_data.groupby('cookie').cumcount(ascending=False) + 1
        decay_scores = decay_factor ** decay_scores
        self.marketing_data['time_decay'] = decay_scores
        self.attribution_results['time_decay'] = self.marketing_data[self.marketing_data['conversion'] == 1].groupby('channel')['time_decay'].sum()

    def position_based(self, first_touch_weight=0.4, last_touch_weight=0.4):
        def apply_weights(group):
            if len(group) == 1:
                return [1.0]
            return [first_touch_weight] + [((1 - first_touch_weight - last_touch_weight) / max(1, len(group) - 2))] * (len(group) - 2) + [last_touch_weight]
        weights = self.marketing_data.groupby('cookie').apply(lambda x: pd.Series(apply_weights(x)))
        self.marketing_data['position_weight'] = weights.values.flatten()
        self.attribution_results['position_based'] = self.marketing_data[self.marketing_data['conversion'] == 1].groupby('channel')['position_weight'].sum()



    def calculate_shapley(self, sample_size=None):
        # Convertir los datos a un formato más adecuado para cálculos de atribución
        if sample_size:
            data_sample = self.marketing_data.sample(n=sample_size)
        else:
            data_sample = self.marketing_data

        # Agrupar los datos por 'cookie' y listar los 'channels' involucrados en cada conversión
        grouped = data_sample[data_sample['conversion'] == 1].groupby('cookie')['channel'].apply(list)

        # Todos los canales únicos
        unique_channels = data_sample['channel'].unique()

        # Función para calcular la contribución de cada subconjunto
        def v(S):
            if not S:
                return 0
            mask = grouped.apply(lambda x: set(S).issubset(x))
            return grouped[mask].count()

        # Calcular los valores de Shapley
        shapley_values = defaultdict(int)
        for channel in unique_channels:
            for subset in chain.from_iterable(combinations(unique_channels, r) for r in range(len(unique_channels)+1)):
                if channel in subset:
                    subset_without_channel = list(subset)
                    subset_without_channel.remove(channel)
                    weight = (factorial(len(subset_without_channel)) * factorial(len(unique_channels) - len(subset)) / factorial(len(unique_channels)))
                    marginal_contribution = v(subset) - v(subset_without_channel)
                    shapley_values[channel] += weight * marginal_contribution

        # Guardar los resultados
        self.attribution_results['shapley'] = shapley_values

    def run_all(self, test_mode=False):
        self.first_touch()
        self.last_touch()
        self.linear()
        self.time_decay()
        self.position_based()
        if test_mode:
            self.calculate_shapley(sample_size=100)  # Ejemplo: usar solo 100 muestras para prueba
        else:
            self.calculate_shapley()
        return pd.DataFrame(self.attribution_results).fillna(0)

       

    def get_results(self):
        return pd.DataFrame(self.attribution_results).fillna(0)


