# Proyecto de Clasificación de SMS Spam

Este repositorio contiene un proyecto educativo para la clasificación de mensajes SMS como "spam" o "no spam" utilizando técnicas de machine learning. El objetivo principal de este proyecto es demostrar cómo se puede implementar un clasificador básico para analizar mensajes de texto, proporcionando una experiencia de aprendizaje práctica en el uso de modelos de regresión logística.

## Estructura del Proyecto

El proyecto contiene los siguientes archivos principales:

- `spam.py`: Contiene el código principal del proyecto, incluyendo la carga del modelo, la preparación de los datos y la interfaz de usuario para la clasificación de los mensajes.
- `utilities.py`: Proporciona funciones útiles como la limpieza de los datos, combinación y deduplicación de bases de datos, y la creación de botones para la interfaz gráfica.
- `local_data/spam.csv` y `local_data/spam2.csv`: Archivos CSV que contienen los datos de mensajes de spam y no spam utilizados para entrenar el modelo.

## Requisitos Previos

Para ejecutar este proyecto, necesitarás tener instalado:

- Python 3.7+
- Las bibliotecas requeridas especificadas en el código:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `tkinter`

Puedes instalar las dependencias con el siguiente comando:

```sh
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Instrucciones de Uso

1. **Clonar el Repositorio**

   Clona este repositorio en tu máquina local:

   ```sh
   git clone https://github.com/Anthar17/IA-UADY.git
   ```

2. **Ejecutar el Clasificador de SMS Spam**

   Para ejecutar la aplicación principal que clasifica los mensajes, ejecuta el archivo `spam.py`:

   ```sh
   python spam.py
   ```

   Se abrirá una ventana donde podrás ingresar un mensaje SMS y determinar si es spam o no.

3. **Funcionalidades**

   La aplicación incluye varias funcionalidades para la visualización de datos y el análisis del modelo:

   - Clasificar mensajes ingresados manualmente.
   - Mostrar la curva ROC para evaluar el rendimiento del modelo.
   - Mostrar la matriz de confusión de los resultados.
   - Mostrar la importancia de las palabras más relevantes según el modelo.
   - Ver estadísticas generales del dataset.

## Estructura del Código

### `spam.py`

Este archivo contiene el flujo principal del proyecto, que incluye:

- **Carga y limpieza de datos**: Lectura de los archivos CSV, eliminación de duplicados y valores nulos.
- **Entrenamiento del modelo**: Utiliza un clasificador de regresión logística para entrenar un modelo que clasifica mensajes en spam o no spam.
- **Interfaz Gráfica**: Una interfaz gráfica (GUI) construida con Tkinter para interactuar con el modelo, mostrar resultados y visualizar el rendimiento del sistema.

### `utilities.py`

Este archivo incluye funciones auxiliares para tareas comunes, como:

- **Limpieza de datos**: Eliminar valores nulos y caracteres no deseados.
- **Lectura y combinación de bases de datos**: Leer y combinar datos de diferentes archivos CSV, asegurando que no haya duplicados.
- **Creación de elementos GUI**: Generar botones y ventanas para la interfaz de usuario.

## Datos

Los archivos de datos `spam.csv` y `spam2.csv` contienen ejemplos de mensajes de texto clasificados como "ham" (no spam) o "spam". Estos datos se usan para entrenar el modelo de machine learning.

- **`spam.csv`**: Contiene datos etiquetados como spam y no spam, con columnas para la clase (`class`) y el mensaje (`sms`).
- **`spam2.csv`**: Un archivo adicional de mensajes que se combina con `spam.csv` para entrenar un modelo más robusto.


