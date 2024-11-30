import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score, precision_recall_curve, ConfusionMatrixDisplay, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

def limpiar(data):
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

# Rutas de los archivos
base1_path = "local_data/spam.csv"
base2_path = "local_data/spam2.csv"

# Leer las bases
base1 = pd.read_csv(base1_path, usecols=[0, 1], names=['class', 'sms'], skiprows=1, encoding='latin1')
base2 = pd.read_csv(base2_path, usecols=[0, 1], names=['class', 'sms'], skiprows=1)

# Convertir ham/spam a 0 y 1
base1['class'] = base1['class'].map({'ham': 0, 'spam': 1})


# Eliminar ruido como \r\n y "Subject:" en ambas bases
for base in [base1, base2]:
    base['sms'] = base['sms'].str.replace(r'\r\n', ' ', regex=True)  # Remover \r\n
    base['sms'] = base['sms'].str.replace(r'^Subject:\s*', '', regex=True)  # Remover "Subject:"
    base['sms'] = base['sms'].str.replace(r'\benron\b', '', regex=True)  # Remover "enron" como palabra completa

# Eliminar duplicados en ambas bases
base1 = base1.drop_duplicates(subset='sms')
base2 = base2.drop_duplicates(subset='sms')

# Combinar las bases
combined_data = pd.concat([base1, base2], ignore_index=True)

# Verificar duplicados en el conjunto combinado
combined_data = combined_data.drop_duplicates(subset='sms')

# Guardar la base combinada
combined_data.to_csv("local_data/spam_combined.csv", index=False)

# Rutas de los archivos combinados
combined_data_path = "local_data/spam_combined.csv"

# Verificar si la base combinada existe
if os.path.exists(combined_data_path):
    print("Cargando base combinada...")
    data = pd.read_csv(combined_data_path)
else:
    print("Error: No se encontró la base combinada. Por favor, verifica que el archivo exista.")
    exit()


combined_data = limpiar(combined_data)

# Preparación de datos
X = combined_data['sms']
y = combined_data['class']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Vectorización y entrenamiento
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=10000, class_weight='balanced')
model.fit(X_train_vec, y_train)

y_val_probs = model.predict_proba(X_val_vec)[:, 1]
y_test_probs = model.predict_proba(X_test_vec)[:, 1]
y_val_pred = model.predict(X_val_vec)
y_test_pred = model.predict(X_test_vec)

# Funciones para mostrar gráficas y resultados
def show_roc_curve():
    fpr_val, tpr_val, _ = roc_curve (y_val, y_val_probs)
    auc_val = roc_auc_score(y_val, y_val_probs)
    fpr_test, tpr_test, _ = roc_curve(y_test.map({'ham': 0, 'spam': 1}), y_test_probs)
    auc_test = roc_auc_score(y_test.map({'ham': 0, 'spam': 1}), y_test_probs)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fpr_val, tpr_val, label=f'Validación (AUC = {auc_val:.2f})', color='blue')
    ax.plot(fpr_test, tpr_test, label=f'Prueba (AUC = {auc_test:.2f})', color='green')
    ax.plot([0, 1], [0, 1], 'k--', label='Azar')
    ax.set_title("Curva ROC")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    plt.show()

def show_precision_recall():
    precision_val, recall_val, _ = precision_recall_curve(y_val.map({'ham': 0, 'spam': 1}), y_val_probs)
    precision_test, recall_test, _ = precision_recall_curve(y_test.map({'ham': 0, 'spam': 1}), y_test_probs)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(recall_val, precision_val, label="Validación", color="blue")
    ax.plot(recall_test, precision_test, label="Prueba", color="green")
    ax.set_title("Curva Precision-Recall")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")

    plt.show()

def show_confusion_matrix():
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ConfusionMatrixDisplay.from_predictions(
        y_val, y_val_pred, ax=ax[0], cmap="Blues", colorbar=False
    )
    ax[0].set_title("Matriz de Confusión (Validación)")

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_test_pred, ax=ax[1], cmap="Greens", colorbar=False
    )
    ax[1].set_title("Matriz de Confusión (Prueba)")

    plt.tight_layout()
    plt.show()

def show_word_importance():
    features = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    top_positive_indices = coefficients.argsort()[-10:]
    top_negative_indices = coefficients.argsort()[:10]

    top_features = pd.DataFrame({
        "Palabra": np.concatenate([features[top_negative_indices], features[top_positive_indices]]),
        "Coeficiente": np.concatenate([coefficients[top_negative_indices], coefficients[top_positive_indices]])
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_features, x="Coeficiente", y="Palabra", ax=ax, hue="Coeficiente", dodge=False, palette="coolwarm")
    ax.set_title("Importancia de Palabras según el Modelo")
    ax.set_xlabel("Peso del Coeficiente")
    ax.set_ylabel("Palabras")

    plt.show()
    
def show_probability_distributions():
    # Crear un DataFrame para las probabilidades y clases reales
    val_data = pd.DataFrame({'Probabilidad': y_val_probs, 'Clase': y_val})
    test_data = pd.DataFrame({'Probabilidad': y_test_probs, 'Clase': y_test})

    # Asegurar que las clases están correctamente etiquetadas como 'spam' o 'ham'
    val_data['Clase'] = val_data['Clase'].map(lambda x: 'spam' if x == 'spam' else 'ham')
    test_data['Clase'] = test_data['Clase'].map(lambda x: 'spam' if x == 'spam' else 'ham')

    # Crear figura con dos subgráficos
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))  # 1 fila, 2 columnas

    # Gráfico para Validación
    sns.histplot(
        val_data[val_data['Clase'] == 'spam']['Probabilidad'],
        color="green", kde=True, label="Spam", bins=30, ax=ax[0]
    )
    sns.histplot(
        val_data[val_data['Clase'] == 'ham']['Probabilidad'],
        color="red", kde=True, label="Ham", bins=30, ax=ax[0]
    )
    ax[0].set_title("Distribución de Probabilidades (Validación)")
    ax[0].set_xlabel("Probabilidad Predicha")
    ax[0].set_ylabel("Frecuencia")
    ax[0].legend()

    # Gráfico para Prueba
    sns.histplot(
        test_data[test_data['Clase'] == 'spam']['Probabilidad'],
        color="green", kde=True, label="Spam", bins=30, ax=ax[1]
    )
    sns.histplot(
        test_data[test_data['Clase'] == 'ham']['Probabilidad'],
        color="red", kde=True, label="Ham", bins=30, ax=ax[1]
    )
    ax[1].set_title("Distribución de Probabilidades (Prueba)")
    ax[1].set_xlabel("Probabilidad Predicha")
    ax[1].set_ylabel("Frecuencia")
    ax[1].legend()

    # Ajustar y mostrar
    plt.tight_layout()
    plt.show()
    
def show_metrics():
    # Evaluar en conjunto de validación
    y_val_pred = model.predict(X_val_vec)
    y_test_pred = model.predict(X_test_vec)

    # Calcular métricas
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_report = classification_report(y_val, y_val_pred)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_report = classification_report(y_test, y_test_pred)

    # Mostrar resultados en una ventana separada
    metrics_window = tk.Toplevel(root)
    metrics_window.title("Métricas del Modelo")

    # Texto para Validación
    val_label = ttk.Label(metrics_window, text="Conjunto de Validación", font=("Arial", 12, "bold"))
    val_label.pack(pady=5)

    val_text = scrolledtext.ScrolledText(metrics_window, wrap=tk.WORD, width=70, height=10)
    val_text.insert(tk.END, f"Accuracy: {val_accuracy:.2f}\n\n")
    val_text.insert(tk.END, val_report)
    val_text.configure(state="disabled")
    val_text.pack(pady=5)

    # Texto para Prueba
    test_label = ttk.Label(metrics_window, text="Conjunto de Prueba", font=("Arial", 12, "bold"))
    test_label.pack(pady=5)

    test_text = scrolledtext.ScrolledText(metrics_window, wrap=tk.WORD, width=70, height=10)
    test_text.insert(tk.END, f"Accuracy: {test_accuracy:.2f}\n\n")
    test_text.insert(tk.END, test_report)
    test_text.configure(state="disabled")
    test_text.pack(pady=5)

    # Botón para cerrar
    close_button = ttk.Button(metrics_window, text="Cerrar", command=metrics_window.destroy)
    close_button.pack(pady=10)
    
def show_loss_curve():
    # Crear un rango de probabilidades predichas (p.ej., 0 a 1)
    thresholds = np.linspace(0, 1, 100)

    # Calcular log loss en función de las probabilidades predichas
    validation_losses = []
    test_losses = []
    for threshold in thresholds:
        # Ajustar probabilidades según el umbral (binarización)
        val_probs_adjusted = (y_val_probs >= threshold).astype(int)
        test_probs_adjusted = (y_test_probs >= threshold).astype(int)
        
        # Calcular log loss
        validation_loss = log_loss(y_val.map({'ham': 0, 'spam': 1}), y_val_probs)
        test_loss = log_loss(y_test.map({'ham': 0, 'spam': 1}), y_test_probs)
        
        validation_losses.append(validation_loss)
        test_losses.append(test_loss)
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, validation_losses, label='Validación', color='blue')
    ax.plot(thresholds, test_losses, label='Prueba', color='green')
    ax.set_title("Curva de Log Loss")
    ax.set_xlabel("Probabilidad Predicha")
    ax.set_ylabel("Log Loss")
    ax.legend(loc="upper right")
    
    plt.show()
    
def regenerar_archivo():
    try:
        # Leer las bases
        base1 = pd.read_csv(base1_path, usecols=[0, 1], names=['class', 'sms'], skiprows=1, encoding='latin1')
        base2 = pd.read_csv(base2_path, usecols=[0, 1], names=['class', 'sms'], skiprows=1)
        base2['class'] = base2['class'].map({1: 'spam', 0: 'ham'})  # Convertir a texto

        # Eliminar ruido como \r\n y "Subject:" en ambas bases
        for base in [base1, base2]:
            base['sms'] = base['sms'].str.replace(r'\r\n', ' ', regex=True)  # Remover \r\n
            base['sms'] = base['sms'].str.replace(r'^Subject:\s*', '', regex=True)  # Remover "Subject:"
            base['sms'] = base['sms'].str.replace(r'\benron\b', '', regex=True)  # Remover "enron" como palabra completa

        # Eliminar duplicados en ambas bases
        base1 = base1.drop_duplicates(subset='sms')
        base2 = base2.drop_duplicates(subset='sms')

        # Combinar las bases
        combined_data = pd.concat([base1, base2], ignore_index=True)

        # Verificar duplicados en el conjunto combinado
        combined_data = combined_data.drop_duplicates(subset='sms')

        # Guardar la base combinada
        combined_data.to_csv("local_data/spam_combined.csv", index=False)
        
        # Mostrar mensaje de éxito
        messagebox.showinfo("Éxito", "Archivo combinado regenerado correctamente.")
    except Exception as e:
        # Mostrar mensaje de error
        messagebox.showerror("Error", f"Ocurrió un error al regenerar el archivo: {e}")
        
def show_dataset_stats():
    stats_window = tk.Toplevel(root)
    stats_window.title("Estadísticas del Dataset")
    stats_window.geometry("400x200")
    
    total_messages = len(combined_data)
    spam_messages = len(combined_data[combined_data['class'] == 'spam'])
    ham_messages = len(combined_data[combined_data['class'] == 'ham'])
    avg_spam_length = combined_data[combined_data['class'] == 'spam']['sms'].str.len().mean()
    avg_ham_length = combined_data[combined_data['class'] == 'ham']['sms'].str.len().mean()
    
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
    message = entry_message.get()
    if not message.strip():
        messagebox.showwarning("Advertencia", "Por favor, ingrese un mensaje.")
        return

    # Vectorizar el mensaje
    message_vec = vectorizer.transform([message])
    
    # Predicción
    prediction = model.predict(message_vec)[0]
    probabilities = model.predict_proba(message_vec)[0]
    result = "SPAM" if prediction == "spam" else "NO SPAM"
    
    # Crear una ventana adicional para mostrar más detalles
    def show_details():
        details_window = tk.Toplevel(root)
        details_window.title("Detalles del Mensaje")
        details_window.geometry("420x200")
        
        spam_prob = f"Probabilidad de SPAM: {probabilities[1]*100:.2f}%"
        ham_prob = f"Probabilidad de NO SPAM: {probabilities[0]*100:.2f}%"
        
        ttk.Label(details_window, text=spam_prob, font=("Arial", 12)).pack(pady=5)
        ttk.Label(details_window, text=ham_prob, font=("Arial", 12)).pack(pady=5)
        
        # Analizar datos relevantes del mensaje
        if prediction == "spam":
            analysis = "Este mensaje tiene características comunes con SPAM."
        else:
            analysis = "Este mensaje tiene características comunes con mensajes normales."
        ttk.Label(details_window, text=analysis, font=("Arial", 10)).pack(pady=5)
        
        # Botón para cerrar la ventana de detalles
        ttk.Button(details_window, text="Cerrar", command=details_window.destroy).pack(pady=10)

    # Mostrar el mensaje clasificado con un botón adicional
    details_button = messagebox.askyesno("Resultado", f"El mensaje introducido es clasificado como: {result}.\n\n¿Desea ver más detalles?")
    if details_button:
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

btn_classify = ttk.Button(frame, text="Clasificar", command=classify_message)
entry_message.bind("<Return>", lambda event: classify_message())
btn_classify.pack(pady=10)

btn_roc = ttk.Button(frame, text="Mostrar Curva ROC", command=show_roc_curve)
btn_roc.pack(pady=5)

btn_precision_recall = ttk.Button(frame, text="Mostrar Curva Precision-Recall", command=show_precision_recall)
btn_precision_recall.pack(pady=5)

btn_confusion_matrix = ttk.Button(frame, text="Mostrar Matriz de Confusión", command=show_confusion_matrix)
btn_confusion_matrix.pack(pady=5)

btn_word_importance = ttk.Button(frame, text="Mostrar Importancia de Palabras", command=show_word_importance)
btn_word_importance.pack(pady=5)

btn_word_metrics = ttk.Button(frame, text="Ver Métricas del Modelo", command=show_metrics)
btn_word_metrics.pack(pady=5)

btn_ds_info = ttk.Button(frame, text="Ver Stats del Dataset", command=show_dataset_stats)
btn_ds_info.pack(pady=5)

# Botón para regenerar el archivo combinado
btn_regenerate = ttk.Button(frame, text="Regenerar Archivo", command=regenerar_archivo)
btn_regenerate.pack(pady=5)

btn_prob_dist = ttk.Button(frame, text="Ver Distribuciones de Probabilidades", command=show_probability_distributions)
btn_prob_dist.pack(pady=5)

btn_loss_curve = ttk.Button(frame, text="Mostrar Curva de Pérdida", command=show_loss_curve)
btn_loss_curve.pack(pady=5)

root.mainloop()
