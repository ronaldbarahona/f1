import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Generación de datos sintéticos
def generar_datos_f1(n_pilotos=500):
    np.random.seed(42)
    constructores = ['Mercedes', 'Red Bull', 'Ferrari', 'McLaren', 'Alpine']
    data = []
    
    for _ in range(n_pilotos):
        edad = np.random.randint(18, 40)
        constructor = np.random.choice(constructores, p=[0.25, 0.25, 0.2, 0.15, 0.15])
        
        # Base de rendimiento por constructor
        if constructor == 'Red Bull':
            puntos = np.clip(np.random.normal(380, 30), 250, 500)
        elif constructor == 'Mercedes':
            puntos = np.clip(np.random.normal(350, 40), 200, 450)
        else:
            puntos = np.clip(np.random.normal(250, 60), 50, 400)
        
        # Ajuste por edad (pico de rendimiento 25-32 años)
        puntos *= 1.2 if 25 <= edad <= 32 else 0.9
        
        data.append({
            'Constructor': constructor,
            'Edad': edad,
            'Puntos': puntos
        })
    
    return pd.DataFrame(data)

df = generar_datos_f1()

# 2. Definición de rangos (simplificada)
rangos = {
    'Bajo (0-150)': (0, 150),
    'Medio (151-300)': (151, 300),
    'Alto (301-450)': (301, 450),
    'Élite (451+)': (451, float('inf'))
}

df['Rango'] = pd.cut(df['Puntos'], 
                    bins=[0, 150, 300, 450, float('inf')], 
                    labels=rangos.keys())

# 3. Preprocesamiento
preprocesador = ColumnTransformer([
    ('scaler', StandardScaler(), ['Edad']),
    ('encoder', OneHotEncoder(), ['Constructor'])
])

# 4. Pipeline con RandomForest (mejor para este caso)
modelo = Pipeline([
    ('preprocesamiento', preprocesador),
    ('clasificador', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. Entrenamiento
X = df[['Constructor', 'Edad']]
y = df['Rango']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo.fit(X_train, y_train)

# 6. Evaluación
print("\nEvaluación del Modelo:")
print(classification_report(y_test, modelo.predict(X_test)))

# 7. Predicción para nuevos pilotos
nuevos_pilotos = pd.DataFrame({
    'Constructor': ['Red Bull', 'Mercedes', 'McLaren'],
    'Edad': [26, 33, 24]
})

predicciones = modelo.predict(nuevos_pilotos)
print("\nPredicción de Rango de Puntos:")
for piloto, rango in zip(nuevos_pilotos['Constructor'], predicciones):
    print(f"- {piloto}: {rango}")