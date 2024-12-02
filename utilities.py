""""Este módulo contiene funciones auxiliares que son utilizadas en la aplicación para la clasificación de mensajes de spam.
Estas funciones permiten limpiar, procesar y gestionar datos de mensajes, así como crear elementos de interfaz gráfica 
(Tkinter) para mejorar la experiencia del usuario. 

Funciones principales:
- **clean_null(data)**: Limpia un DataFrame eliminando valores nulos e inconsistentes.
- **read_and_convert(base_path)**: Lee un archivo CSV de spam y lo convierte a un formato binario (0 para "ham" y 1 para "spam").
- **combine_deduplicate(bases)**: Combina varias bases de datos, limpia caracteres innecesarios y elimina duplicados.
- **save_and_load(combined_data, output_path)**: Guarda los datos combinados en un archivo y los carga posteriormente si existe.
- **create()**: Realiza todo el proceso de carga, limpieza, combinación y deduplicación de las bases de datos.
- **msj_start(msj)**: Muestra un mensaje emergente de carga por un tiempo determinado.
- **create_button(frame, text, command, pady=5)**: Crea un botón en la interfaz gráfica con las propiedades especificadas.

Este archivo se utiliza como soporte para realizar las tareas repetitivas de limpieza y manejo de datos, así como para 
gestionar algunos aspectos de la interfaz gráfica.

"""
import os
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

def read_clean_convert(base_path):
    """
    Lee un archivo CSV, convierte las etiquetas de clase a formato numérico,
    y limpia los datos eliminando valores nulos y corrigiendo inconsistencias.
    """
    
    # Leer las bases
    base = pd.read_csv(base_path, usecols=[0, 1], names=['class', 'sms'], skiprows=1, encoding='latin1')

    # Convertir ham/spam a 0 y 1
    if 'ham' in base['class'].unique():
        base['class'] = base['class'].map({'ham': 0, 'spam': 1})
    
    # Asegurarse de trabajar con una copia
    base = base.copy()

    # Limpiar nombres de columnas para evitar errores
    base.columns = base.columns.str.strip()

    # Eliminar valores nulos en 'sms'
    base = base.dropna(subset=['sms'])

    # Convertir todo a cadenas válidas
    base['sms'] = base['sms'].astype(str)
    base['sms'] = base['sms'].str.strip()  # Eliminar espacios en blanco

    # Eliminar filas con contenido no interpretable
    base = base[base['sms'] != '']  # Eliminar cadenas vacías
    base = base[base['sms'].apply(lambda x: isinstance(x, str))]  # Asegurarse de que sean cadenas

    # Verificar y eliminar valores nulos en 'class'
    base = base.dropna(subset=['class'])  # Eliminar valores nulos en 'class'
    base = base[base['class'].isin([0, 1])]  # Asegurar etiquetas válidas

    return base  # Devolver el DataFrame limpio


def combine_deduplicate(bases):
    """
    Combina varias bases de datos, elimina duplicados y limpia caracteres innecesarios.
    """
    
    # Limpiar caracteres innecesarios
    for base in bases:
        base['sms'] = base['sms'].str.replace(r'\r\n', ' ', regex=True)  # Remover \r\n
        base['sms'] = base['sms'].str.replace(r'^Subject:\s*', '', regex=True)  # Remover "Subject:"
        base['sms'] = base['sms'].str.replace(r'\benron\b', '', regex=True)  # Remover "enron" como palabra completa
        base['sms'] = base['sms'].str.replace(r'\byour\b', '', regex=True)  # Remover "your" como palabra completa
        base['sms'] = base['sms'].str.replace(r'\bkaminski\b', '', regex=True)  # Remover "kaminski" como palabra completa
        base['sms'] = base['sms'].str.replace(r'\bvince\b', '', regex=True)  # Remover "vince" como palabra completa
        
    
    # Combinar las bases y eliminar duplicados
    combined_data = pd.concat(bases, ignore_index=True)
    combined_data = combined_data.drop_duplicates(subset='sms')
    return combined_data

def save_and_load(combined_data, output_path="local_data/spam_combined.csv"):
    """
    Guarda los datos combinados en un archivo CSV y los carga si el archivo existe.
    """
    
    # Guardar la base combinada
    combined_data.to_csv(output_path, index=False)

    # Intentar cargar la base combinada desde la ruta proporcionada
    if os.path.exists(output_path):
        # Mostrar un mensaje de carga usando la función `msj_start()`
        msj_start("Cargando base combinada...")
        data = pd.read_csv(output_path)
        return data  # Retornar los datos cargados correctamente
    else:
        # Mostrar un mensaje de error si no se encuentra la base combinada
        messagebox.showerror("Error", "No se encontró la base combinada.")
        return None  # Devolver None para indicar error en lugar de salir del programa

def create():
    """
    Realiza el proceso completo de leer, combinar, deduplicar y limpiar las bases de datos.
    """
    # Rutas de los archivos
    base1_path = "local_data/spam.csv"
    base2_path = "local_data/spam2.csv"

    # Leer y convertir las bases
    base1 = read_clean_convert(base1_path)
    base2 = read_clean_convert(base2_path)
    bases = [base1, base2]

    # Limpiar, combinar y deduplicar las bases
    combined_data = combine_deduplicate(bases)

    data = save_and_load(combined_data)

    return data

def msj_start(msj):
    """
    Muestra un mensaje emergente durante 2 segundos.
    
    Parámetros:
    msj (str): El mensaje que se mostrará en la ventana emergente.
    
    Esta función crea una ventana emergente utilizando `tkinter` que muestra 
    un mensaje centrado en la pantalla por un tiempo determinado (2 segundos)
    antes de cerrarse automáticamente.
    """
    
    root = tk.Tk()  # Crea la ventana principal
    root.title("Mensaje de Carga")  # Título de la ventana emergente
    label = ttk.Label(root, text=msj)  # Etiqueta que contiene el mensaje
    label.pack(expand=True, padx=20, pady=20)  # Ajusta el diseño del mensaje
    
    # Dimensiones de la ventana emergente
    ancho = 300
    alto = 150
    
    # Configura el tamaño de la ventana
    root.geometry(f"{ancho}x{alto}")
    root.update_idletasks()  # Fuerza a la interfaz a procesar cualquier actualización pendiente
    
    # Calcula las coordenadas para centrar la ventana en la pantalla
    x = (root.winfo_screenwidth() // 2) - (ancho // 2)
    y = (root.winfo_screenheight() // 2) - (alto // 2)
    root.geometry(f"+{x}+{y}")  # Coloca la ventana en la posición calculada
    
    # Programa el cierre automático de la ventana después de 2 segundos
    root.after(2000, root.destroy)
    
    root.mainloop()  # Inicia el bucle principal de eventos para mostrar la ventana

    
def create_button(frame, text, command, pady=5):
    """
    Crea un botón en la interfaz gráfica con las propiedades dadas.
    
    Parámetros:
    frame (tk.Frame o tk.Widget): El contenedor donde se agregará el botón.
    text (str): El texto que se mostrará en el botón.
    command (callable): La función que se ejecutará cuando se presione el botón.
    pady (int, opcional): El espaciado vertical alrededor del botón (por defecto, 5).
    
    Retorno:
    ttk.Button: El objeto botón creado.
    """
    
    btn = ttk.Button(frame, text=text, command=command)  # Crea un botón con las propiedades dadas
    btn.pack(pady=pady)  # Empaqueta el botón en el contenedor con el espaciado especificado
    return btn  # Devuelve el botón para posibles usos adicionales
