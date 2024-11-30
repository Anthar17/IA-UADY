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

def clean_null(data):
    """
    Limpia los datos eliminando los valores nulos y corrigiendo inconsistencias.
    """
    
    # Asegurarse de trabajar con una copia
    data = data.copy()

    # Limpiar nombres de columnas para evitar errores
    data.columns = data.columns.str.strip()

    # Eliminar valores nulos en 'sms'
    data = data.dropna(subset=['sms'])

    # Convertir todo a cadenas válidas
    data['sms'] = data['sms'].astype(str)
    data['sms'] = data['sms'].str.strip()  # Eliminar espacios en blanco

    # Eliminar filas con contenido no interpretable
    data = data[data['sms'] != '']  # Eliminar cadenas vacías
    data = data[~data['sms'].str.contains(r'^\s*$', regex=True)]  # Eliminar filas con solo espacios
    data = data[data['sms'].apply(lambda x: isinstance(x, str))]  # Asegurarse de que sean cadenas

    # Verificar y eliminar valores nulos en 'class'
    data = data.dropna(subset=['class'])  # Eliminar valores nulos en 'class'
    data = data[data['class'].isin([0, 1])]  # Asegurar etiquetas válidas

    return data  # Devolver el DataFrame limpio

def read_and_convert(base_path):
    """
    Lee un archivo CSV y convierte las etiquetas de clase a formato numérico.
    """
    
    # Leer las bases
    base = pd.read_csv(base_path, usecols=[0, 1], names=['class', 'sms'], skiprows=1, encoding='latin1')

    # Convertir ham/spam a 0 y 1
    if 'ham' in base['class'].unique():
        base['class'] = base['class'].map({'ham': 0, 'spam': 1})
    return base

def combine_deduplicate(bases):
    """
    Combina varias bases de datos, elimina duplicados y limpia caracteres innecesarios.
    """
    
    # Limpiar caracteres innecesarios
    for base in bases:
        base['sms'] = base['sms'].str.replace(r'\r\n', ' ', regex=True)  # Remover \r\n
        base['sms'] = base['sms'].str.replace(r'^Subject:\s*', '', regex=True)  # Remover "Subject:"
        base['sms'] = base['sms'].str.replace(r'\benron\b', '', regex=True)  # Remover "enron" como palabra completa
    
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
    base1 = read_and_convert(base1_path)
    base2 = read_and_convert(base2_path)
    bases = [base1, base2]

    # Limpiar, combinar y deduplicar las bases
    combined_data = combine_deduplicate(bases)

    # Limpiar valores nulos del combinado
    combined_data = clean_null(combined_data)

    data = save_and_load(combined_data)

    return data

def msj_start(msj):
    """
    Muestra un mensaje emergente durante 2 segundos.
    """
    
    root = tk.Tk()
    root.title("Mensaje de Carga")  # Título de la ventana
    label = ttk.Label(root, text=msj)
    label.pack(expand=True, padx=20, pady=20)
    ancho = 300
    alto = 150
    root.geometry(f"{ancho}x{alto}")
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (ancho // 2)
    y = (root.winfo_screenheight() // 2) - (alto // 2)
    root.geometry(f"+{x}+{y}")
    root.after(2000, root.destroy)
    root.mainloop()
    
def create_button(frame, text, command, pady=5):
    """
    Crea un botón en la interfaz gráfica con las propiedades dadas.
    """
    
    btn = ttk.Button(frame, text=text, command=command)
    btn.pack(pady=pady)
    return btn