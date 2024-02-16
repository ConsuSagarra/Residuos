
from numpy import asarray
from pandas import DataFrame
from matplotlib import pyplot
import matplotlib.pyplot as plt
import pandas as pd


#Cálculo del residuo 
def calculate_residuals(y_test:DataFrame, y_pred:DataFrame) -> DataFrame:
    
    """
    Parámetros:
    y_test (pd.DataFrame): Primer DataFrame.
    y_pred (pd.DataFrame): Segundo DataFrame.
    Devuelve:
    residual: DataFrame.
    """
    #Para asegurarse que los índices están alineados
    y_test = y_test.sort_index()
    y_pred = y_pred.sort_index()

    residual = y_pred - y_test
    return residual


# Gráfico de la función de df1 y df2
def viz_df1_df2(df1:DataFrame, df2:DataFrame, titulo:str, etiqueta_x:str, etiqueta_y:str):
    
    """
    Parámetros:
    df1 (pd.DataFrame): Primer DataFrame.
    df2 (pd.DataFrame): Segundo DataFrame.
    titulo (str): Título del gráfico.
    etiqueta_x (str): Etiqueta para el eje x.
    etiqueta_y (str): Etiqueta para el eje y.
    """
    # Asegurarse de que los índices están alineados
    df1 = df1.sort_index()
    df2 = df2.sort_index()

    # Graficar los DataFrames
    plt.figure(figsize=(16,  4))
    plt.plot(df1.index, df1[etiqueta_y], color ='black', label='df1')
    plt.plot(df2.index, df2[etiqueta_y], color ='orange', label='df2')
    plt.title(titulo)
    plt.xlabel(etiqueta_x)
    plt.ylabel(etiqueta_y)
    plt.legend()
    plt.grid()
    plt.show()

    return

# Gráfico de la función de df1, df2 y el residuo
def viz_df1_df2_residual(df1:DataFrame, df2:DataFrame, residuo:DataFrame, titulo:str, etiqueta_x:str, etiqueta_y:str):
    
    """
    Parámetros:
    df1 (pd.DataFrame): Primer DataFrame, dato.
    df2 (pd.DataFrame): Segundo DataFrame, modelo.
    residuo (pd.DataFrame): Tercer DataFrame, residuo = df1-df2. 
    titulo (str): Título del gráfico.
    etiqueta_x (str): Etiqueta para el eje x.
    etiqueta_y (str): Etiqueta para el eje y.
    """
    # Asegurarse de que los índices están alineados
    df1 = df1.sort_index()
    df2 = df2.sort_index()
    residuo = residuo.sort_index()

    # Graficar los DataFrames
    plt.figure(figsize=(16,  4))
    plt.plot(df1.index, df1[etiqueta_y], color ='black', label='df1')
    plt.plot(df2.index, df2[etiqueta_y], color ='orange', label='df2')
    plt.plot(residuo.index, residuo[etiqueta_y], color ='green', label='Residual')
    plt.title(titulo)
    plt.xlabel(etiqueta_x)
    plt.ylabel(etiqueta_y)
    plt.legend()
    plt.grid()
    plt.show()

    return

# Gráfico de la función de df1, df2, residual y anomalía conocida
def viz_df1_df2_residual_outk(df1:DataFrame, df2:DataFrame, residuo:DataFrame, known_outliers:DataFrame, titulo:str, etiqueta_x:str, etiqueta_y:str):
    
    """
    Parámetros:
    df1 (pd.DataFrame): Primer DataFrame, dato.
    df2 (pd.DataFrame): Segundo DataFrame, modelo.
    residuo (pd.DataFrame): Tercer DataFrame, residuo = df1-df2. 
    known_outliers (pd.DataFrame): Cuarto DataFrame, anomalías conocidas
    titulo (str): Título del gráfico.
    etiqueta_x (str): Etiqueta para el eje x.
    etiqueta_y (str): Etiqueta para el eje y.
    """
    # Asegurarse de que los índices están alineados
    df1 = df1.sort_index()
    df2 = df2.sort_index()
    residuo = residuo.sort_index()

    # Graficar los DataFrames
    plt.figure(figsize=(16,  4))
    

    plt.plot(df1.index, df1[etiqueta_y], color ='black', label='df1')
    plt.plot(df2.index, df2[etiqueta_y], color ='orange', label='df2')
    plt.plot(residuo.index, residuo[etiqueta_y], color ='green', label='Residual')
    plt.scatter(known_outliers.index, known_outliers[etiqueta_y], color='blue', label = 'Anomaly-known')
    plt.title(titulo)
    plt.xlabel(etiqueta_x)
    plt.ylabel(etiqueta_y)
    plt.legend()
    plt.grid()
    plt.show()

    return


# Función para generar una función binaria en el cual 1 es un dato oultiers
def binarizar_sobre_cota(df:DataFrame, cota:float)->DataFrame:
    
    """
    Binariza los valores de un DataFrame basándose en una cota.
    Parámetros:
    df (pd.DataFrame): DataFrame original.
    cota (float): Valor de referencia para la binarización.
    Devuelve:
    df_binarizado: DataFrame con valores binarizados (0 ó 1)
    """
    # Aplicar la función lambda a cada elemento del DataFrame
    df_binarizado = df.applymap(lambda x:  0 if x > cota else  1)
    
    return df_binarizado



# Gráfico de la función de df1, df2, residual, anomalía conocida y anomalías detectadas a partir de una cota en el residuo
def viz_df1_df2_residual_outk_out(df1, df2, residuo, known_outliers, outliers, titulo, etiqueta_x, etiqueta_y):
    
    """
    Parámetros:
    df1 (pd.DataFrame): Primer DataFrame, dato.
    df2 (pd.DataFrame): Segundo DataFrame, modelo.
    residuo (pd.DataFrame): Tercer DataFrame, residuo = df1-df2. 
    known_outliers (pd.DataFrame): Cuarto DataFrame, anomalías conocidas.
    outliers (pd.DataFrame): Quinto DaraFrame, outliers obtenidos a partir del calculo del residuo y la cota.
    titulo (str): Título del gráfico.
    etiqueta_x (str): Etiqueta para el eje x.
    etiqueta_y (str): Etiqueta para el eje y.
    """
    # Asegurarse de que los índices están alineados
    df1 = df1.sort_index()
    df2 = df2.sort_index()
    residuo = residuo.sort_index()

    # Graficar los DataFrames
    plt.figure(figsize=(16,  4))
    
    # Acomodamos el dato de outliers 
    
    a = residuo.loc[outliers[etiqueta_y] == 1, [etiqueta_y]]

    plt.plot(df1.index, df1[etiqueta_y], color ='black', label='df1')
    plt.plot(df2.index, df2[etiqueta_y], color ='orange', label='df2')
    plt.plot(residuo.index, residuo[etiqueta_y], color ='green', label='Residual')
    plt.scatter(a.index, a[etiqueta_y], color='red', label = 'Outliers')
    plt.scatter(known_outliers.index, known_outliers[etiqueta_y], color='blue', label = 'Anomaly-known')
    plt.title(titulo)
    plt.xlabel(etiqueta_x)
    plt.ylabel(etiqueta_y)
    plt.legend()
    plt.grid()
    plt.show()
    
    return 