import pandas as pd

# Definición de rangos (personalizable)
RANGOS = {
    '0-50': (0, 50),
    '51-100': (51, 100),
    '101-200': (101, 200),
    '201-300': (201, 300),
    '301-400': (301, 400),
    '401+': (401, float('inf'))
}

def clasificar_corredores(df):
    # Calcular el máximo histórico por piloto
    max_puntos = df.groupby('Nombre')['Puntos'].max().reset_index()
    
    # Asignar rango a cada piloto
    max_puntos['Rango'] = max_puntos['Puntos'].apply(
        lambda x: next((r for r, (min_val, max_val) in RANGOS.items() if min_val <= x <= max_val), 'Desconocido')
    )
    
    # Fusionar con datos originales
    resultado = pd.merge(df, max_puntos[['Nombre', 'Rango']], on='Nombre', how='left')
    
    # Seleccionar columnas relevantes y eliminar duplicados
    columnas = ['Nombre', 'Constructor', 'Posición_Final', 'Puntos', 'Podios', 'Pole_Positions', 'Carreras_Disputadas', 'Rango']
    return resultado[columnas].drop_duplicates('Nombre').sort_values('Puntos', ascending=False)

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar datos (asegúrate de tener tu CSV)
    df = pd.read_csv('f1_driver_standings_2006_2023.csv')
    
    # Clasificar corredores
    clasificacion = clasificar_corredores(df)
    
    # Mostrar resultados en formato simple
    for _, row in clasificacion.iterrows():
        print(f"[{row['Nombre']}] ; [{row['Rango']}]")
