
import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

#creamos sinteticos realistas

np.random.seed(42)
fechas = pd.date_range('2023-01-01', end= '2024-12-31', freq='D')
n_productos = ['Lapto', 'Mouse', 'Teclado', 'Monitor', 'Auriculares']
Regiones = ['Norte', 'Sur', 'Este','Oeste', 'Centro']

#Generamos el DataSet

data = []
for fecha in fechas:
    for _ in range (np.random.poisson(10)):  #10 ventas promedio por dia
        data.append({
            'Fechas': fecha,
            'Producto': np.random.choice(n_productos),
            'region' : np.random.choice(Regiones),
            'Ventas' : np.random.randint(1,6),
            'Precio unitario' : np.random.uniform(50, 1500),
            'Vendedor' : f'Vendedor {np.random.randint(1, 21)}',
        })

df = pd.DataFrame(data)

#print(df)

df ['Venta_total'] = df['Ventas'] * df['Precio unitario']
#print(df)
#print('shape del Dataset: ', df.shape)
#print(df.head(10))
#print('\información General')
#print(df.info())
#print('\nDescripción estadística')
#print(df.describe())



#Configuración de la pagina
st.set_page_config(page_title='Dashboard de ventas', page_icon=':bar_chart:', layout='wide')
st.title('Dashboard de analisis de ventas')
st.markdown('---')

#Siderbar para filtros
st.sidebar.header('Filtros')
productos_selecionado = st.sidebar.multiselect(
    'Selecciona productos:',
    options=df['Producto'].unique(),
    default=df['Producto'].unique(),
)

Regiones_selecionado = st.sidebar.multiselect(
    'Selecciona Región:',
    options=df['region'].unique(),
    default=df['region'].unique(),
)

#Filtrar los datos basado en la selección
df_filtrado = df[
    (df['Producto'].isin(productos_selecionado)) &
    (df['region'].isin(Regiones_selecionado))
]


#1. Ventas por Mes
#def graficar_ventas(df):
#df_monthly = df.groupby(df['Fechas'].dt.to_period('M'))['Venta_total'].sum().reset_index()
#df_monthly['Fecha'] = df_monthly['Fechas'].astype(str)
  #print(df_monthly)
  
  #Ventas por mes(con filtros)
df_monthly = df_filtrado.groupby(df_filtrado['Fechas'].dt.to_period('M'))['Venta_total'].sum().reset_index()
df_monthly['Fecha'] = df_monthly['Fechas'].astype(str)

#____________________

fig_monthly = px.line(df_monthly, x='Fecha', y='Venta_total',
                        title= 'Tendencias de Ventas Mensuales',
                        labels={'Venta_total': 'Ventas ($)', 'Fecha' : 'Mes'})
fig_monthly.update_traces(line=dict(width=4, color= 'royalblue'))
  #fig_monthly.show()


#app.graficar_ventas(df)

#2. Top productos
#def graficar_top_productos(df):
df_productos = df_filtrado.groupby('Producto')['Venta_total'].sum().sort_values(ascending=True)
fig_productos = px.bar(x=df_productos.values, y=df_productos.index,
                         orientation='h' , title='Ventas por Producto',
                         labels={'x': 'Ventas Totales ($)', 'y' : 'Producto'})
fig_productos.update_traces(marker_color='royalblue')
  #fig_productos.show()

#graficar_top_productos(df)

  #3. Análisis Geográfico
#def graficar_analisis_geografico(df):
df_regiones = df_filtrado.groupby('region')['Venta_total'].sum().reset_index()
fig_regiones = px.pie(df_regiones, values='Venta_total', names='region',
                        title='Distribución de Ventas por Región',
                        labels={'Venta_total' : 'Ventas Totales ($)'})
fig_regiones.update_traces(textposition='inside', textinfo='percent+label')
  #fig_regiones.show()

#import importlib
#import app
#importlib.reload(app)
#graficar_analisis_geografico(df)

#4. Correlación entre variables
#def graficar_correlacion(df):
df_corr = df_filtrado[['Ventas', 'Precio unitario', 'Venta_total']].corr()
fig_heatmap = px.imshow(df_corr, text_auto=True, aspect= 'auto',
                          title='Correlación entre variables',
                          labels=dict(x='Variables', y='Variables', color='Correlación'))
fig_heatmap.update_layout(coloraxis_colorbar_title_text='Correlación')
  #fig_heatmap.show()

#graficar_correlacion(df)

#5. distribución Ventas
#def graficar_distribucion_ventas(df):
fig_dist = px.histogram(df_filtrado, x='Venta_total', nbins=50,
                          title='Distribución de Individuales')
fig_dist.update_layout(bargap=0.2)
  #fig_dist.show()

#graficar_distribucion_ventas(df)



#Colocar filtros

#Métricas Principales
col1, col2, col3, col4 = st.columns(4)
with col1:
  st.metric('Ventas Totales', f"${df_filtrado['Venta_total'].sum():,.0f}")
with col2:
  st.metric('Promedio por Venta', f"${df_filtrado['Venta_total'].mean():,.0f}")
with col3:
  st.metric('Número de Ventas', f"{len(df_filtrado):,}")
with col4:
  crecimiento =((df_filtrado[df_filtrado['Fechas'] >= '2024-01-01']['Venta_total'].sum() /
                 df_filtrado[df_filtrado['Fechas'] < '2024-01-01']['Venta_total'].sum()) - 1) * 100
  st.metric('Crecimiento de Ventas 2024', f"{crecimiento:.1f}%")

  #Layout con dos columnas
col1, col2 = st.columns(2)
with col1:
  #st.subheader('Ventas Mensuales')
  st.plotly_chart(fig_monthly, use_container_width=True)
  st.markdown('---')
  st.markdown('✅ **Análisis**: La tendencia mensual de ventas permite identificar los periodos de mayor y menor actividad comercial. Esta información es clave para planificar campañas, promociones o inventarios estratégicamente.')
  st.plotly_chart(fig_productos, use_container_width=True)
  st.markdown('---')
  st.markdown('✅ **Análisis**: El análisis por producto muestra claramente cuáles son los artículos más rentables. Los productos con mayor volumen de ventas pueden representar oportunidades de expansión o especialización comercial.')

with col2:
  st.plotly_chart(fig_regiones, use_container_width=True)
  st.markdown('---')
  st.markdown('✅ **Análisis**: La distribución geográfica revela qué regiones concentran mayor facturación. Esto ayuda a focalizar esfuerzos logísticos, comerciales y de atención al cliente en zonas clave.')
  st.plotly_chart(fig_heatmap, use_container_width=True)
  st.markdown('---')
  st.markdown('**Análisis**: La matriz de correlación indica que existe una relación fuerte entre el total de venta y la cantidad, como es lógico. Esto valida la estructura del modelo de ventas y permite detectar posibles patrones de compra.')
  
  #Gráfico completo en la parte inferior
st.plotly_chart(fig_dist, use_container_width=True)
st.markdown('---')
st.markdown("✅ **Análisis**: La distribución de ventas individuales muestra cómo se comporta el valor de cada transacción. Si la mayoría de las ventas están concentradas en un rango bajo o medio, puede considerarse una estrategia de diversificación de precios.")

