

# Aprendizaje Automático aplicado a la asignación publicitaria

## Descripción
Este repositorio contiene el código y los recursos asociados a mi tesis sobre el uso de Aprendizaje Automático para la asignación publicitaria, explorando modelos de atribución y mezcla de medios. Este trabajo incluye el desarrollo, la evaluación y puesta en produccion de varios modelos.

## Estructura del repositorio

- `/codigo`: Contiene todos los Jupyter Notebooks utilizados para la observación inicial de datasets, análisis exploratorio de datos (EDA), y los scripts de Python para la generación y visualización de datos. Subdirectorios:
  - `/MAM`: Jupyter Notebooks que demuestran la aplicación de la biblioteca de atribución de marketing desarrollada, `MarketingAttrLib`.
  - `/MMM`: Scripts relacionados con la implementacion de los modelos de LighweightMMM para mezcla de medios.
- `/data`: Incluye todos los datasets utilizados en los análisis iniciales.
- `/imagenes`: Almacena imágenes generadas externamente y logos de herramientas de proveedores externos.
- `/mlflow`: Pruebas de implementación de MLflow para producción (no incluido en el escrito).
- `/zenml_test`: Código implementado para la puesta en producción de los modelos utilizando ZenML.

## Instalación

Primero, clona este repositorio en tu máquina local usando:

```bash
git clone URL_DEL_REPOSITORIO
```

Asegúrate de tener Python instalado y luego instala las dependencias necesarias con:

```bash
pip install -r requirements.txt
```

## Uso



### ZenML
Para ejecutar el servidor de ZenML y desplegar los modelos, sigue estos pasos en la terminal:

1. Navega al directorio `zenml_test` donde se encuentra el código de producción.
2. Inicia el servidor ZenML usando:

```bash
zenml up
```

3. Para ejecutar los pipelines de ZenML, utiliza:

```bash
zenml pipeline run NOMBRE_DEL_PIPELINE
```




