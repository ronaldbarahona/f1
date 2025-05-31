import requests
import pandas as pd

def get_driver_standings(year):
    """Obtiene los datos de la clasificación de pilotos para una temporada."""
    url = f"http://ergast.com/api/f1/{year}/driverStandings.json"
    response = requests.get(url)
    data = response.json()

    # Extraer los datos relevantes de la clasificación de pilotos
    standings = data['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']

    driver_data = []
    for driver in standings:
        driver_info = driver['Driver']
        constructor = driver['Constructors'][0] if driver['Constructors'] else {}

        driver_data.append({
            "Año": year,
            "Pos": driver['position'],
            "Nombre": f"{driver_info['givenName']} {driver_info['familyName']}",
            "Nacionalidad": driver_info.get('nationality', 'N/A'),
            "Fecha_Nacimiento": driver_info.get('dateOfBirth', 'N/A'),
            "Número": driver_info.get('permanentNumber', 'N/A'),
            "Código": driver_info.get('code', 'N/A'),
            "Constructor": constructor.get('name', 'N/A'),
            "Posición_Final": int(driver.get('position', 0)),
            "Puntos": driver['points'],
            "Victorias": driver['wins'],
            "URL_Info": driver_info.get('url', ''),
            "Carreras_Disputadas": 0,  # Se actualizará más tarde
            "Podios": 0,  # Se actualizará más tarde
            "Pole_Positions": 0,  # Se actualizará más tarde
            "DNFs": 0  # Se actualizará más tarde
        })

    return driver_data


def get_race_results(year):
    """Obtiene los resultados de todas las carreras de una temporada."""
    url = f"http://ergast.com/api/f1/{year}/results.json"
    response = requests.get(url)
    data = response.json()
    races = data['MRData']['RaceTable']['Races']
    return races


def actualizar_estadisticas(race_data, driver_data):
    """Actualiza las estadísticas de los pilotos (Carreras Disputadas, Podios, Pole Positions, DNFs)."""
    for race in race_data:
        for result in race['Results']:
            driver_code = result['Driver']['code']
            for driver in driver_data:
                if driver_code == driver['Código']:
                    # Contabilizar Carreras Disputadas
                    driver['Carreras_Disputadas'] += 1

                    # Verificar si 'positionOrder' está presente antes de acceder
                    if 'positionOrder' in result:
                        # Contabilizar Podios
                        if result['positionOrder'] in ['1', '2', '3']:
                            driver['Podios'] += 1

                    # Verificar si 'grid' está presente antes de acceder
                    if 'grid' in result and result['grid'] == '1':
                        # Contabilizar Pole Positions
                        driver['Pole_Positions'] += 1

                    # Verificar si el status está presente y es 'Retired' para DNF
                    if 'status' in result and result['status'] == 'Retired':
                        driver['DNFs'] += 1

    return driver_data


def obtener_todos_los_datos(start_year, end_year):
    """Obtiene los datos de clasificación y resultados de todas las temporadas dentro del rango especificado."""
    all_data = []
    for year in range(start_year, end_year + 1):
        print(f"Procesando datos de la temporada {year}...")

        # Obtener clasificaciones de pilotos
        driver_data = get_driver_standings(year)

        # Obtener resultados de carreras para esa temporada
        race_data = get_race_results(year)

        # Actualizar estadísticas de cada piloto
        driver_data = actualizar_estadisticas(race_data, driver_data)

        # Agregar los datos de esta temporada al conjunto de todos los datos
        all_data.extend(driver_data)

    return all_data


# Rango de años que el usuario desea procesar
start_year = int(input("Introduce el primer año de la temporada: "))
end_year = int(input("Introduce el último año de la temporada: "))

# Obtener todos los datos
all_data = obtener_todos_los_datos(start_year, end_year)

# Crear DataFrame de pandas con los datos obtenidos
df = pd.DataFrame(all_data)

# Guardar el DataFrame como archivo CSV
#csv_filename = f"f1_driver_standings_{start_year}_{end_year}.csv"
csv_filename = f"f1_driver_standings.csv"
df.to_csv(csv_filename, index=False)

print(f"✅ Los datos de las temporadas {start_year} a {end_year} han sido guardados en '{csv_filename}'.")

