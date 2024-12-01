"""Este script implementa un clasificador de SMS para identificar mensajes de spam usando Regresión Logística y TfidfVectorizer.
Incluye una interfaz gráfica que permite al usuario ingresar mensajes, clasificarlos y visualizar métricas del modelo.

Explicación de las bibliotecas utilizadas:
- `sklearn.model_selection.train_test_split`: Para dividir los datos en entrenamiento, validación y prueba.
- `sklearn.feature_extraction.text.TfidfVectorizer`: Para convertir los mensajes de texto en representaciones numéricas utilizando TF-IDF.
- `sklearn.linear_model.LogisticRegression`: Para entrenar un modelo de clasificación usando Regresión Logística.
- `numpy`: Para manejar cálculos matemáticos y trabajar con arrays.
- `sklearn.metrics`: Para calcular las métricas de evaluación del modelo, como la precisión, la matriz de confusión, la curva ROC, etc.
- `matplotlib.pyplot` y `seaborn`: Para visualizar gráficos y resultados, como las curvas ROC y las distribuciones de probabilidad.
- `tkinter`: Para crear la interfaz gráfica de usuario que permite la interacción con el clasificador.
- `utilities`: Contiene funciones personalizadas para manejar datos, botones de la interfaz gráfica y funciones auxiliares específicas del proyecto.
- `pandas`: Para manipular y analizar los datos en formato de tablas (DataFrames).
"""

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score, precision_recall_curve, ConfusionMatrixDisplay, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import ttk, messagebox, scrolledtext
from utilities import clean_null, read_and_convert, combine_deduplicate, create, save_and_load, create_button
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

# Definición de las rutas de los archivos CSV a utilizar para entrenamiento del modelo.
base1_path = "local_data/spam.csv"
base2_path = "local_data/spam2.csv"

# Lectura de los datos y conversión utilizando una función externa definida en utilities.
base1 = read_and_convert(base1_path)
base2 = read_and_convert(base2_path)
bases = [base1, base2]

# Combinar los datos de los dos archivos y eliminar duplicados para limpiar la base de datos.
combined_data = combine_deduplicate(bases)

# Se eliminan los valores nulos que puedan estar presentes en los datos combinados.
combined_data = clean_null(combined_data)

# Guardar y cargar los datos combinados (guardado en disco y carga posterior).
data = save_and_load(combined_data)

# Preparación de datos
# Se separan los datos en la columna 'sms' (entrada X) y la columna 'class' (objetivo y).
X = data['sms']
y = data['class']

# Dividir los datos en entrenamiento, validación y prueba.
# Primero, se usa train_test_split para dividir en entrenamiento (80%) y el resto (20%).
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Luego, se divide el 20% restante en validación y prueba (10% cada uno del total).
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Vectorización y entrenamiento del modelo
# Convertir los mensajes de texto en características numéricas usando TfidfVectorizer.
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# Entrenamiento de un modelo de Regresión Logística con los datos vectorizados.
model = LogisticRegression(max_iter=10000, class_weight='balanced')
model.fit(X_train_vec, y_train)

# Generación de predicciones para los conjuntos de validación y prueba.
y_val_probs = model.predict_proba(X_val_vec)[:, 1]
y_test_probs = model.predict_proba(X_test_vec)[:, 1]
y_val_pred = model.predict(X_val_vec)
y_test_pred = model.predict(X_test_vec)

# Funciones para mostrar gráficas y resultados
def show_roc_curve():
    """
    Genera y muestra la curva ROC (Receiver Operating Characteristic) tanto para los conjuntos de validación como de prueba.
    La curva ROC ayuda a evaluar la capacidad del modelo para diferenciar entre clases.
    """
    # Calcular la Tasa de Falsos Positivos (FPR) y Tasa de Verdaderos Positivos (TPR) para el conjunto de validación
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_probs)
    auc_val = roc_auc_score(y_val, y_val_probs)

    # Calcular FPR y TPR para el conjunto de prueba
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_probs)
    auc_test = roc_auc_score(y_test, y_test_probs)

    # Crear la figura para la curva ROC
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fpr_val, tpr_val, label=f'Validación (AUC = {auc_val:.2f})', color='blue')  # Curva ROC para validación
    ax.plot(fpr_test, tpr_test, label=f'Prueba (AUC = {auc_test:.2f})', color='green')  # Curva ROC para prueba
    ax.plot([0, 1], [0, 1], 'k--', label='Azar')  # Línea de referencia para el azar
    ax.set_title("Curva ROC")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    plt.show()

def show_precision_recall():
    """
    Genera y muestra la curva de Precisión-Recall para los conjuntos de validación y prueba.
    Esta gráfica ayuda a evaluar la precisión del modelo frente a la exhaustividad.
    """
    # Calcular precisión y recall para el conjunto de validación
    precision_val, recall_val, _ = precision_recall_curve(y_val, y_val_probs)

    # Calcular precisión y recall para el conjunto de prueba
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_probs)

    # Crear la figura para la curva de Precisión-Recall
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(recall_val, precision_val, label="Validación", color="blue")
    ax.plot(recall_test, precision_test, label="Prueba", color="green")
    ax.set_title("Curva Precision-Recall")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")

    plt.show()

def show_confusion_matrix():
    """
    Genera y muestra la matriz de confusión para los conjuntos de validación y prueba.
    La matriz de confusión permite visualizar cuántos mensajes se clasificaron correctamente e incorrectamente.
    """
    # Crear la figura con dos subgráficos (para validación y prueba)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Mostrar matriz de confusión para validación
    ConfusionMatrixDisplay.from_predictions(y_val, y_val_pred, ax=ax[0], cmap="Blues", colorbar=False)
    ax[0].set_title("Matriz de Confusión (Validación)")

    # Mostrar matriz de confusión para prueba
    ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, ax=ax[1], cmap="Greens", colorbar=False)
    ax[1].set_title("Matriz de Confusión (Prueba)")

    plt.tight_layout()
    plt.show()

def show_word_importance():
    """
    Muestra un gráfico de barras con las palabras más importantes según los coeficientes del modelo.
    Esto ayuda a entender qué palabras tienen más peso en la clasificación de mensajes como spam o no spam.
    """
    # Obtener los nombres de las características y los coeficientes del modelo
    features = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    # Seleccionar los 10 coeficientes más positivos y más negativos
    top_positive_indices = coefficients.argsort()[-10:]
    top_negative_indices = coefficients.argsort()[:10]

    # Crear un DataFrame con las palabras más importantes y sus coeficientes
    top_features = pd.DataFrame({
        "Palabra": np.concatenate([features[top_negative_indices], features[top_positive_indices]]),
        "Coeficiente": np.concatenate([coefficients[top_negative_indices], coefficients[top_positive_indices]])
    })

    # Crear el gráfico de barras
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_features, x="Coeficiente", y="Palabra", ax=ax, hue="Coeficiente", dodge=False, palette="coolwarm")
    ax.set_title("Importancia de Palabras según el Modelo")
    ax.set_xlabel("Peso del Coeficiente")
    ax.set_ylabel("Palabras")

    plt.show()

def show_probability_distributions():
    """
    Muestra la distribución de probabilidades predichas para las clases spam y no spam, tanto en el conjunto de validación como de prueba.
    """
    # Crear un DataFrame con las probabilidades predichas y las clases reales
    val_data = pd.DataFrame({'Probabilidad': y_val_probs, 'Clase': y_val})
    test_data = pd.DataFrame({'Probabilidad': y_test_probs, 'Clase': y_test})

    # Convertir las clases a etiquetas legibles ('spam' y 'ham')
    val_data['Clase'] = val_data['Clase'].map({1: 'spam', 0: 'ham'})
    test_data['Clase'] = test_data['Clase'].map({1: 'spam', 0: 'ham'})

    # Crear la figura con dos subgráficos: uno para validación y otro para prueba
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Graficar la distribución de probabilidades para validación
    sns.histplot(val_data[val_data['Clase'] == 'spam']['Probabilidad'], color="green", kde=True, label="Spam", bins=30, ax=ax[0])
    sns.histplot(val_data[val_data['Clase'] == 'ham']['Probabilidad'], color="red", kde=True, label="Ham", bins=30, ax=ax[0])
    ax[0].set_title("Distribución de Probabilidades (Validación)")
    ax[0].set_xlabel("Probabilidad Predicha")
    ax[0].set_ylabel("Frecuencia")
    ax[0].legend()

    # Graficar la distribución de probabilidades para prueba
    sns.histplot(test_data[test_data['Clase'] == 'spam']['Probabilidad'], color="green", kde=True, label="Spam", bins=30, ax=ax[1])
    sns.histplot(test_data[test_data['Clase'] == 'ham']['Probabilidad'], color="red", kde=True, label="Ham", bins=30, ax=ax[1])
    ax[1].set_title("Distribución de Probabilidades (Prueba)")
    ax[1].set_xlabel("Probabilidad Predicha")
    ax[1].set_ylabel("Frecuencia")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

def show_metrics():
    """
    Muestra las métricas de rendimiento del modelo para los conjuntos de validación y prueba en una ventana emergente.
    """
    # Calcular las predicciones para validación y prueba
    y_val_pred = model.predict(X_val_vec)
    y_test_pred = model.predict(X_test_vec)

    # Calcular las métricas de precisión y generar el reporte de clasificación
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_report = classification_report(y_val, y_val_pred)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_report = classification_report(y_test, y_test_pred)

    # Crear una ventana emergente para mostrar las métricas
    metrics_window = tk.Toplevel(root)
    metrics_window.title("Métricas del Modelo")

    # Texto para el conjunto de validación
    val_label = ttk.Label(metrics_window, text="Conjunto de Validación", font=("Arial", 12, "bold"))
    val_label.pack(pady=5)

    val_text = scrolledtext.ScrolledText(metrics_window, wrap=tk.WORD, width=70, height=10)
    val_text.insert(tk.END, f"Accuracy: {val_accuracy:.2f}\n\n")
    val_text.insert(tk.END, val_report)
    val_text.configure(state="disabled")
    val_text.pack(pady=5)

    # Texto para el conjunto de prueba
    test_label = ttk.Label(metrics_window, text="Conjunto de Prueba", font=("Arial", 12, "bold"))
    test_label.pack(pady=5)

    test_text = scrolledtext.ScrolledText(metrics_window, wrap=tk.WORD, width=70, height=10)
    test_text.insert(tk.END, f"Accuracy: {test_accuracy:.2f}\n\n")
    test_text.insert(tk.END, test_report)
    test_text.configure(state="disabled")
    test_text.pack(pady=5)

    close_button = ttk.Button(metrics_window, text="Cerrar", command=metrics_window.destroy)
    close_button.pack(pady=10)

def show_loss_curve():
    """
    Muestra la curva de Log Loss para los conjuntos de validación y prueba, lo cual refleja la penalización por predicciones incorrectas.
    """
    # Calcular Log Loss para validación y prueba
    validation_loss = log_loss(y_val, y_val_probs)
    test_loss = log_loss(y_test, y_test_probs)

    # Crear figura para la curva de Log Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([0, 1], [validation_loss, validation_loss], label='Validación', color='blue')
    ax.plot([0, 1], [test_loss, test_loss], label='Prueba', color='green')
    ax.set_title("Curva de Log Loss")
    ax.set_xlabel("Umbral")
    ax.set_ylabel("Log Loss")
    ax.legend(loc="upper right")

    plt.show()

def regenerar_archivo():
    """
    Regenera el archivo combinado de las bases de datos de spam. Si ocurre un error, muestra un mensaje de error.
    """
    try:
        # Llamar a la función `create` para limpiar, combinar y deduplicar los datos
        create()
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error al regenerar el archivo: {e}")

def show_dataset_stats():
    """
    Muestra estadísticas generales del dataset, como el total de mensajes y sus proporciones de spam y no spam.
    """
    stats_window = tk.Toplevel(root)
    stats_window.title("Estadísticas del Dataset")
    stats_window.geometry("400x200")

    # Calcular estadísticas del dataset
    total_messages = len(combined_data)
    spam_messages = len(combined_data[combined_data['class'] == 1])
    ham_messages = len(combined_data[combined_data['class'] == 0])
    avg_spam_length = combined_data[combined_data['class'] == 1]['sms'].str.len().mean()
    avg_ham_length = combined_data[combined_data['class'] == 0]['sms'].str.len().mean()

    # Mostrar las estadísticas en la ventana
    stats = f"""
    Total de mensajes: {total_messages}
    Mensajes SPAM: {spam_messages} ({spam_messages/total_messages*100:.2f}%)
    Mensajes NO SPAM: {ham_messages} ({ham_messages/total_messages*100:.2f}%)
    Longitud promedio SPAM: {avg_spam_length:.2f} caracteres
    Longitud promedio NO SPAM: {avg_ham_length:.2f} caracteres
    """
    
    ttk.Label(stats_window, text=stats, font=("Arial", 10)).pack(pady=10)
    ttk.Button(stats_window, text="Cerrar", command=stats_window.destroy).pack(pady=10)

# Interfaz gráfica con tkinter
def classify_message():
    """
    Clasifica un mensaje de texto ingresado por el usuario como "SPAM" o "NO SPAM" usando el modelo entrenado.
    Muestra una ventana emergente con el resultado de la clasificación y, opcionalmente, detalles adicionales.

    Función interna:
    - show_details(): Muestra detalles adicionales del mensaje clasificado, como las probabilidades predichas para ser spam o no.

    Argumentos: 
    - No recibe argumentos directos, usa el contenido del campo de texto de la interfaz gráfica.

    Ejemplo de uso:
    - Al escribir un mensaje en la interfaz y presionar el botón "Clasificar", esta función será invocada para clasificar el mensaje.

    """
    # Obtener el mensaje ingresado por el usuario
    message = entry_message.get()
    if not message.strip():  # Verificar que no esté vacío
        messagebox.showwarning("Advertencia", "Por favor, ingrese un mensaje.")  # Advertencia si no hay contenido
        return
    
    # Limpiar el cuadro de texto después de obtener el mensaje
    entry_message.delete(0, tk.END)

    # Vectorizar el mensaje para prepararlo para el modelo
    message_vec = vectorizer.transform([message])
    
    # Hacer la predicción usando el modelo entrenado
    prediction = model.predict(message_vec)[0]
    probabilities = model.predict_proba(message_vec)[0]  # Obtener las probabilidades de cada clase
    result = "SPAM" if prediction == 1 else "NO SPAM"  # Determinar si es SPAM o NO SPAM
    
    # Crear una ventana adicional para mostrar más detalles
    def show_details():
        """
        Muestra una ventana emergente con detalles sobre las probabilidades de la predicción.
        Informa la probabilidad de que el mensaje sea spam y no spam.
        """
        details_window = tk.Toplevel(root)  # Crear una ventana adicional
        details_window.title("Detalles del Mensaje")
        details_window.geometry("420x200")

        # Mostrar las probabilidades de que el mensaje sea spam o no spam
        spam_prob = f"Probabilidad de SPAM: {probabilities[1]*100:.2f}%"
        ham_prob = f"Probabilidad de NO SPAM: {probabilities[0]*100:.2f}%"
        
        # Etiquetas para mostrar las probabilidades calculadas
        ttk.Label(details_window, text=spam_prob, font=("Arial", 12)).pack(pady=5)
        ttk.Label(details_window, text=ham_prob, font=("Arial", 12)).pack(pady=5)
        
        # Analizar los datos relevantes del mensaje
        if prediction == 1:
            analysis = "Este mensaje tiene características comunes con SPAM."
        else:
            analysis = "Este mensaje tiene características comunes con mensajes normales."
        ttk.Label(details_window, text=analysis, font=("Arial", 10)).pack(pady=5)
        
        # Botón para cerrar la ventana de detalles
        ttk.Button(details_window, text="Cerrar", command=details_window.destroy).pack(pady=10)

    # Mostrar el resultado de la clasificación en una ventana emergente
    details_button = messagebox.askyesno("Resultado", f"El mensaje introducido es clasificado como: {result}.\n\n¿Desea ver más detalles?")
    if details_button:  # Si el usuario selecciona ver más detalles
        show_details()

root = tk.Tk()
root.title("Clasificador de SMS Spam")
root.geometry("500x450")

frame = ttk.Frame(root, padding="10")
frame.pack(fill=tk.BOTH, expand=True)

label = ttk.Label(frame, text="Ingrese un SMS para clasificar:")
label.pack(pady=5)

entry_message = ttk.Entry(frame, width=50)
entry_message.pack(pady=5)

btn_classify = create_button(frame, "Clasificar", classify_message, pady=10)
entry_message.bind("<Return>", lambda event: classify_message())

# Crear el resto de botones usando la función
create_button(frame, "Mostrar Curva ROC", show_roc_curve)
create_button(frame, "Mostrar Curva Precision-Recall", show_precision_recall)
create_button(frame, "Mostrar Matriz de Confusión", show_confusion_matrix)
create_button(frame, "Mostrar Importancia de Palabras", show_word_importance)
create_button(frame, "Ver Métricas del Modelo", show_metrics)
create_button(frame, "Ver Stats del Dataset", show_dataset_stats)
create_button(frame, "Regenerar Archivo", regenerar_archivo)
create_button(frame, "Ver Distribuciones de Probabilidades", show_probability_distributions)
create_button(frame, "Mostrar Curva de Pérdida", show_loss_curve)

# Ejecutar la interfaz gráfica
root.mainloop()
