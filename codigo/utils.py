import numpy as np
import pandas as pd

def create_mmm_dataframe(media_data, extra_features, target, costs):
    # Determinar la forma de los datos de entrada
    data_size, n_media_channels, geos = media_data.shape
    n_extra_features = extra_features.shape[1]
    
    # Extender costs para que tenga la misma forma que media_data
    costs_extended = np.tile(costs, (data_size, geos, 1)).transpose((0, 2, 1))

    # Aplanar los datos para que cada combinación de tiempo, canal y geografía tenga una fila
    media_data_flat = media_data.reshape(data_size * n_media_channels * geos)
    costs_flat = costs_extended.reshape(data_size * n_media_channels * geos)
    
    # Crear un índice para todas las combinaciones posibles
    index = pd.MultiIndex.from_product([range(data_size), range(n_media_channels), range(geos)],
                                       names=['Time', 'Media_Channel', 'Geo'])
    
    # DataFrames para media y costos
    media_df = pd.DataFrame(media_data_flat, index=index, columns=['Media_Value'])
    costs_df = pd.DataFrame(costs_flat, index=index, columns=['Cost'])
    
    # Repetir target y extra_features para cada canal y geografía
    target_repeated = np.repeat(target.flatten(), n_media_channels)
    extra_features_repeated = np.repeat(extra_features.reshape(data_size * geos, n_extra_features), n_media_channels, axis=0)
    
    # Incorporar target y extra_features en DataFrames
    target_df = pd.DataFrame(target_repeated, index=index, columns=['Target'])
    extra_features_df = pd.DataFrame(extra_features_repeated, index=index, columns=[f'Extra_Feature_{i}' for i in range(n_extra_features)])
    
    # Unir todos los DataFrames
    final_df = pd.concat([media_df, costs_df, extra_features_df, target_df], axis=1)
    
    return final_df


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class MarketingDataExplorer:
    def __init__(self, dataframe):
        self.df = dataframe

    def show_head(self, n=5):
        '''
        Muestra las primeras n filas del DataFrame.
        '''
        return self.df.head(n)

    def show_info(self):
        '''
        Muestra información general del DataFrame, incluyendo tipo de datos y valores faltantes.
        '''
        return self.df.info()

    def describe_data(self):
        '''
        Devuelve estadísticas descriptivas que resumen la tendencia central, dispersión y forma de la distribución del dataset.
        '''
        return self.df.describe()

    def plot_missing_values(self):
        '''
        Grafica la cantidad de valores faltantes por columna.
        '''
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        missing.sort_values(inplace=True)
        missing.plot.bar()
        plt.title('Número de valores faltantes por característica')
        plt.show()

    def plot_distributions(self, columns=None):
        '''
        Plots histograms for multiple columns in the dataframe.
        Args:
            columns (list): List of column names to plot distributions. If None, plots all numeric columns.
        '''
        if columns is None:
            columns = self.df.select_dtypes(include=['number']).columns

        for column in columns:
            plt.figure(figsize=(10, 4))
            sns.histplot(self.df[column], kde=True)
            plt.title(f'Distribución de {column}')
            plt.show()

    def plot_correlation_matrix(self):
        '''
        Grafica la matriz de correlación de las características numéricas.
        '''
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title('Matriz de Correlación')
        plt.show()

    def compare_categories(self, column):
        '''
        Compara el número de observaciones en cada categoría para una columna específica.
        '''
        count = self.df[column].value_counts()
        plt.figure(figsize=(10, 5))
        sns.barplot(x=count.index, y=count.values)
        plt.title(f'Comparación de categorías en {column}')
        plt.xticks(rotation=45)
        plt.show()

    def plot_missing_values(self):
        '''
        Grafica la cantidad de valores faltantes por columna.
        '''
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            print("No missing values to plot.")
        else:
            missing.sort_values(inplace=True)
            missing.plot.bar()
            plt.title('Número de valores faltantes por característica')
            plt.show()

    def all(self):
        '''
        Executes all analysis methods on the DataFrame.
        '''
        self.show_head()
        self.show_info()
        self.describe_data()
        self.plot_missing_values()
        self.plot_distributions()
        self.plot_correlation_matrix()
        # Add the column name manually for `compare_categories` if it is needed.
        # self.compare_categories('some_column_name')


# Ejemplo de uso
# df = pd.read_csv('tu_archivo.csv')
# explorer = MarketingDataExplorer(df)
# explorer.show_head()
# explorer.show_info()
# explorer.plot_missing_values()
# explorer.plot_distributions(['columna1', 'columna2'])
# explorer.plot_correlation_matrix()
# explorer.compare_categories('columna_categorica')
