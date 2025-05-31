import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, recall_score

# 1. Crear datos sintéticos basados en reglas de rendimiento
np.random.seed(42)
n_samples = 1000

def generar_datos(n):
    data = []
    for _ in range(n):
        edad = np.random.randint(18, 40)
        constructor = np.random.choice(['Mercedes', 'Red Bull', 'Ferrari', 'McLaren', 'Alpine'], 
                                      p=[0.25, 0.25, 0.2, 0.15, 0.15])
        carreras = np.random.randint(15, 22)
        
        # Reglas de rendimiento basadas en constructor y edad
        if constructor == 'Mercedes':
            puntos_base = np.random.normal(300, 50)
        elif constructor == 'Red Bull':
            puntos_base = np.random.normal(350, 40)
        else:
            puntos_base = np.random.normal(200, 80)
            
        # Ajuste por edad (rendimiento óptimo 25-32 años)
        if 25 <= edad <= 32:
            puntos = puntos_base * 1.2
        else:
            puntos = puntos_base * 0.9
            
        # Variables dependientes
        podios = int(puntos / 25 + np.random.normal(0, 3))
        poles = int(puntos / 50 + np.random.normal(0, 2))
        dnf = np.random.randint(0, 5)
        victorias = int(puntos / 100 + np.random.normal(0, 1))
        
        data.append([constructor, edad, carreras, podios, poles, dnf, victorias, puntos])
    
    return pd.DataFrame(data, columns=['Constructor', 'Edad', 'Carreras_Disputadas', 
                                      'Podios', 'Pole_Positions', 'DNFs', 'Victorias', 'Puntos'])

df = generar_datos(n_samples)

# 2. Definir rangos de clasificación
bins = [0, 50, 100, 200, 300, 400, float('inf')]
labels = ['0-50', '51-100', '101-200', '201-300', '301-400', '401+']
df['Rango'] = pd.cut(df['Puntos'], bins=bins, labels=labels)

# 3. Dividir variables
X = df[['Constructor', 'Edad', 'Carreras_Disputadas', 'Podios', 'Pole_Positions', 'DNFs', 'Victorias']]
y = df['Rango']

# 4. Preprocesamiento
columnas_numericas = ['Edad', 'Carreras_Disputadas', 'Podios', 'Pole_Positions', 'DNFs', 'Victorias']
columnas_categoricas = ['Constructor']

preprocesador = ColumnTransformer(transformers=[
    ('num', StandardScaler(), columnas_numericas),
    ('cat', OneHotEncoder(handle_unknown='ignore'), columnas_categoricas)
])

# 5. Pipeline con regresión logística multinomial
modelo = Pipeline(steps=[
    ('preprocesamiento', preprocesador),
    ('clasificador', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000))
])

# 6. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Entrenamiento
modelo.fit(X_train, y_train)

# 8. Predicción
y_pred = modelo.predict(X_test)

# 9. Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', values_format='d')
plt.title("Matriz de Confusión - Rangos de Puntos")
plt.xticks(rotation=45)
plt.show()

# 10. Métricas
print("\nMétricas de Evaluación:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Recall por clase:")
recall = recall_score(y_test, y_pred, average=None)
for i, r in enumerate(recall):
    print(f"{labels[i]}: {r:.2f}")

# 11. Predicción para nuevos casos
nuevos_datos = pd.DataFrame({
    'Constructor': ['Red Bull', 'Mercedes', 'McLaren'],
    'Edad': [25, 37, 22],
    'Carreras_Disputadas': [22, 21, 20],
    'Podios': [15, 10, 5],
    'Pole_Positions': [5, 3, 1],
    'DNFs': [1, 2, 3],
    'Victorias': [10, 5, 1]
})

predicciones = modelo.predict(nuevos_datos)
print("\nPredicciones para nuevos datos:")
for i, pred in enumerate(predicciones):
    print(f"Piloto {i+1}: {pred}")