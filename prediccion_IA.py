import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def cargar_datos():
    """Carga los datos desde el CSV generado por extractor_datos.py."""
    try:
        df = pd.read_csv("f1_driver_standings_2005_2023.csv")
        # Limpieza y conversión de tipos
        df["Puntos"] = pd.to_numeric(df["Puntos"], errors="coerce")
        df["Posición_Final"] = pd.to_numeric(df["Posición_Final"], errors="coerce")
        df = df.dropna(subset=["Puntos", "Posición_Final"])
        return df
    except FileNotFoundError:
        print("Error: No se encontró 'datos_f1.csv'. Ejecuta primero extractor_datos.py.")
        exit()

def predecir_puntos(df, piloto, año_futuro):
    """Predice los puntos de un piloto en un año futuro."""
    datos_piloto = df[df["Nombre"] == piloto]
    if datos_piloto.empty:
        return f"\nError: No hay datos para {piloto}."
    
    X = datos_piloto[["Año"]]
    y = datos_piloto["Puntos"]
    
    modelo = LinearRegression()
    modelo.fit(X, y)
    prediccion = modelo.predict([[año_futuro]])
    return f"\nPredicción para {piloto} en {año_futuro}: {prediccion[0]:.1f} puntos"

def predecir_podio(df, piloto):
    """Predice la probabilidad de que un piloto termine en el podio (top 3)."""
    df["Podio"] = df["Posición_Final"].apply(lambda x: 1 if x <= 3 else 0)
    datos_piloto = df[df["Nombre"] == piloto]
    
    if datos_piloto.empty:
        return f"\nError: No hay datos para {piloto}."
    
    # Preprocesamiento: Calcular edad y escalar features
    df["Edad"] = pd.to_datetime(df["Fecha_Nacimiento"]).apply(
        lambda x: (pd.to_datetime("now") - x).dt.days // 365
    )
    features = ["Puntos", "Victorias", "DNFs", "Edad"]
    X = datos_piloto[features]
    y = datos_piloto["Podio"]
    
    # Entrenamiento del modelo
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    modelo = RandomForestClassifier(n_estimators=100)
    modelo.fit(X_train, y_train)
    
    proba = modelo.predict_proba(X_test)[:, 1].mean() * 100
    return f"\nProbabilidad de podio para {piloto}: {proba:.1f}%"

def main():
    df = cargar_datos()
    print("\n--- Predicciones Fórmula 1 ---")
    print("1. Predecir puntos para un piloto")
    print("2. Estimar probabilidad de podio")
    opcion = input("Selecciona una opción (1/2): ")
    
    if opcion == "1":
        piloto = input("Nombre del piloto (ej: Lewis Hamilton): ")
        año = int(input("Año futuro para predicción (ej: 2023): "))
        print(predecir_puntos(df, piloto, año))
    elif opcion == "2":
        piloto = input("Nombre del piloto (ej: Max Verstappen): ")
        print(predecir_podio(df, piloto))
    else:
        print("Opción no válida.")

if __name__ == "__main__":
    main()
