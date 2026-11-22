"""Motor reproducible de detección de anomalías de tráfico de red simulado.
Caso académico inspirado en operaciones bancarias; NO son registros del BCP.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

VARIABLES = [
    'duracion_s', 'bytes_entrada', 'bytes_salida', 'paquetes',
    'conexiones_min', 'puertos_distintos', 'intentos_fallidos', 'latencia_ms'
]


def generar_trafico(n_normal: int = 1200, n_anomalias: int = 120, semilla: int = 42, intensidad: float = 1.0,
                    porcentaje_alto: int = 0) -> pd.DataFrame:
    """Genera tráfico ficticio correlacionado y anomalías con etiqueta solo de evaluación."""
    if n_normal < 40 or n_anomalias < 1:
        raise ValueError('Se requieren al menos 40 registros normales y una anomalía.')
    if not (0 < intensidad <= 1) or not (0 <= porcentaje_alto <= 30):
        raise ValueError('Intensidad entre 0 y 1 y porcentaje alto entre 0 y 30.')
    rng = np.random.default_rng(semilla)
    carga = rng.normal(0, 1, n_normal)
    volumen = rng.normal(0, 1, n_normal)
    normales = pd.DataFrame({
        'duracion_s': np.maximum(0.2, 5 + 1.0 * carga + 1.4 * volumen + rng.normal(0, .45, n_normal)),
        'bytes_entrada': np.maximum(100, 61000 + 13000 * carga + 12000 * volumen + rng.normal(0, 3500, n_normal)),
        'bytes_salida': np.maximum(100, 43000 + 8500 * carga + 11000 * volumen + rng.normal(0, 2900, n_normal)),
        'paquetes': np.maximum(2, 480 + 90 * carga + 64 * volumen + rng.normal(0, 22, n_normal)),
        'conexiones_min': np.maximum(1, 32 + 7 * carga + 5 * volumen + rng.normal(0, 2.5, n_normal)),
        'puertos_distintos': np.clip(np.round(3 + .35 * carga + rng.normal(0, .7, n_normal)), 1, 7),
        'intentos_fallidos': np.clip(np.round(.55 + .12 * carga + rng.normal(0, .7, n_normal)), 0, 4),
        'latencia_ms': np.maximum(1, 48 + 7 * carga + 2 * volumen + rng.normal(0, 4, n_normal)),
    })
    n_alto = round(n_normal * porcentaje_alto / 100)
    # Una carga legítima mayor conserva las relaciones conjuntas del generador.
    if n_alto:
        for col, delta in {'duracion_s': 2.4, 'bytes_entrada': 25000,
                           'bytes_salida': 19500, 'paquetes': 154,
                           'conexiones_min': 12, 'latencia_ms': 9}.items():
            normales.loc[:n_alto-1, col] += 2 * delta
    normales['escenario'] = 'trafico_habitual'
    if n_alto:
        normales.loc[:n_alto-1, 'escenario'] = 'alto_volumen_legitimo'
    normales['etiqueta_real'] = 0
    # Las anomalías rompen relaciones entre variables, no sólo aumentan el volumen.
    n_scan = n_anomalias // 2
    n_fuga = n_anomalias - n_scan
    escaneo = normales.sample(n_scan, replace=n_scan > len(normales), random_state=semilla + 1).copy()
    base_escaneo = escaneo[VARIABLES].copy()
    escaneo['conexiones_min'] = rng.integers(145, 310, n_scan)
    escaneo['puertos_distintos'] = rng.integers(18, 115, n_scan)
    escaneo['intentos_fallidos'] = rng.integers(8, 35, n_scan)
    escaneo['bytes_salida'] = rng.integers(500, 18000, n_scan)
    escaneo[VARIABLES] = base_escaneo + intensidad * (escaneo[VARIABLES] - base_escaneo)
    escaneo['escenario'] = 'escaneo_simulado'
    escaneo['etiqueta_real'] = 1
    fuga = normales.sample(n_fuga, replace=n_fuga > len(normales), random_state=semilla + 2).copy()
    base_fuga = fuga[VARIABLES].copy()
    fuga['bytes_salida'] = rng.integers(180000, 550000, n_fuga)
    fuga['bytes_entrada'] = rng.integers(1000, 35000, n_fuga)
    fuga['conexiones_min'] = rng.integers(2, 22, n_fuga)
    fuga[VARIABLES] = base_fuga + intensidad * (fuga[VARIABLES] - base_fuga)
    for parte in (escaneo, fuga):
        for col in ('paquetes', 'conexiones_min', 'puertos_distintos', 'intentos_fallidos'):
            parte[col] = parte[col].round()
    fuga['escenario'] = 'transferencia_atipica_simulada'
    fuga['etiqueta_real'] = 1
    registros = pd.concat([normales, escaneo, fuga], ignore_index=True)
    registros.insert(0, 'id_registro', [f'REG-{i+1:05d}' for i in range(len(registros))])
    return registros


def particionar_datos(datos: pd.DataFrame, semilla: int = 42):
    """700/250/restantes para 1200 normales; separa anomalías de aprendizaje y calibración."""
    verificar_entrada(datos)
    normales = datos.loc[datos['etiqueta_real'] == 0].sample(frac=1, random_state=semilla)
    anomalias = datos.loc[datos['etiqueta_real'] == 1].sample(frac=1, random_state=semilla + 1)
    n = len(normales)
    n_train, n_cal = int(n * .58), int(n * .21)
    n_train = max(10, n_train)
    n_cal = max(10, n_cal)
    if n_train + n_cal >= n:
        raise ValueError('No quedan registros normales para prueba.')
    train = normales.iloc[:n_train].copy()
    calibration = normales.iloc[n_train:n_train+n_cal].copy()
    test = pd.concat([normales.iloc[n_train+n_cal:], anomalias], ignore_index=True)
    test = test.sample(frac=1, random_state=semilla + 2).reset_index(drop=True)
    return train, calibration, test


def verificar_entrada(df: pd.DataFrame, exigir_etiqueta: bool = True):
    if df.empty:
        raise ValueError('El archivo no contiene registros.')
    req = set(VARIABLES + (['etiqueta_real'] if exigir_etiqueta else []))
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f'Faltan columnas: {sorted(missing)}')
    if df[VARIABLES].isna().any().any():
        raise ValueError('No se admiten valores faltantes en las variables numéricas.')
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in VARIABLES):
        raise ValueError('Todas las variables analíticas deben ser numéricas.')
    if not np.isfinite(df[VARIABLES].to_numpy(dtype=float)).all():
        raise ValueError('No se admiten valores infinitos.')
    if (df[VARIABLES] < 0).any().any():
        raise ValueError('Las variables de tráfico no pueden ser negativas.')
    if 'etiqueta_real' in df and not df['etiqueta_real'].isin([0, 1]).all():
        raise ValueError('etiqueta_real debe contener únicamente 0 (normal) o 1 (anomalía).')


@dataclass
class ModeloPCA:
    escalador: StandardScaler
    pca: PCA
    umbral: float
    cuantil: float
    var_objetivo: float


def error_reconstruccion(modelo: ModeloPCA, registros: pd.DataFrame) -> np.ndarray:
    z = modelo.escalador.transform(registros[VARIABLES])
    puntajes = modelo.pca.transform(z)
    z_aprox = modelo.pca.inverse_transform(puntajes)
    # Error cuadrático promedio entre variables tipificadas y reconstruidas.
    return np.mean((z - z_aprox) ** 2, axis=1)


def entrenar(train: pd.DataFrame, calibracion: pd.DataFrame,
             var_objetivo: float = .80, cuantil: float = .99) -> ModeloPCA:
    if not (.50 <= var_objetivo < 1):
        raise ValueError('var_objetivo debe estar entre 0.50 y menos de 1.00.')
    if not (.90 <= cuantil < 1):
        raise ValueError('cuantil debe estar entre 0.90 y menos de 1.00.')
    if len(train) < 10 or len(calibracion) < 10:
        raise ValueError('La calibración y entrenamiento necesitan al menos 10 registros.')
    if (train['etiqueta_real'] != 0).any() or (calibracion['etiqueta_real'] != 0).any():
        raise ValueError('Entrenamiento y calibración se realizan sólo con tráfico normal simulado.')
    escalador = StandardScaler()
    z_train = escalador.fit_transform(train[VARIABLES])
    pca = PCA(n_components=var_objetivo, svd_solver='full')
    pca.fit(z_train)
    temporal = ModeloPCA(escalador, pca, np.nan, cuantil, var_objetivo)
    errores_cal = error_reconstruccion(temporal, calibracion)
    temporal.umbral = float(np.quantile(errores_cal, cuantil))
    return temporal


def evaluar(modelo: ModeloPCA, test: pd.DataFrame):
    resultado = detectar(modelo, test)
    errores = resultado['error_reconstruccion'].to_numpy()
    pred = (errores > modelo.umbral).astype(int)
    etiquetas = test['etiqueta_real'].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(etiquetas, pred, labels=[0, 1]).ravel()
    resultado = test.copy()
    resultado['error_reconstruccion'] = errores
    resultado['prediccion_anomalia'] = pred
    resultado['estado'] = np.where(pred == 1, 'Revisión requerida', 'Dentro del patrón')
    metricas = {
        'n_entrenamiento': int(modelo.pca.n_samples_),
        'n_calibracion': None,
        'n_prueba': int(len(test)),
        'n_componentes': int(modelo.pca.n_components_),
        'varianza_explicada': float(modelo.pca.explained_variance_ratio_.sum()),
        'umbral': float(modelo.umbral),
        'VP': int(tp), 'VN': int(tn), 'FP': int(fp), 'FN': int(fn),
        'precision': float(precision_score(etiquetas, pred, zero_division=0)),
        'sensibilidad': float(recall_score(etiquetas, pred, zero_division=0)),
        'f1': float(f1_score(etiquetas, pred, zero_division=0)),
        'tasa_falsos_positivos': float(fp / (fp + tn)) if fp + tn else 0,
    }
    return resultado, metricas


def coordenadas_pca(modelo: ModeloPCA, datos: pd.DataFrame) -> pd.DataFrame:
    z = modelo.escalador.transform(datos[VARIABLES])
    coords = modelo.pca.transform(z)
    out = pd.DataFrame({'CP1': coords[:, 0]})
    out['CP2'] = coords[:, 1] if coords.shape[1] > 1 else 0.0
    return out


def ejecutar_experimento(n_normal=1200, n_anomalias=120, semilla=42,
                         var_objetivo=.80, cuantil=.99, intensidad=1.0, porcentaje_alto=0):
    datos = generar_trafico(n_normal, n_anomalias, semilla, intensidad, porcentaje_alto)
    train, calibracion, test = particionar_datos(datos, semilla)
    modelo = entrenar(train, calibracion, var_objetivo, cuantil)
    resultado, metricas = evaluar(modelo, test)
    metricas['n_entrenamiento'] = len(train)
    metricas['n_calibracion'] = len(calibracion)
    return datos, train, calibracion, test, modelo, resultado, metricas


def detectar(modelo, registros):
    """Inferencia sin reentrenar; las etiquetas reales son opcionales."""
    verificar_entrada(registros, exigir_etiqueta=False)
    resultado = registros.copy().reset_index(drop=True)
    if 'id_registro' not in resultado:
        resultado.insert(0, 'id_registro', [f'CSV-{i+1:05d}' for i in range(len(resultado))])
    if resultado['id_registro'].isna().any() or resultado['id_registro'].duplicated().any():
        raise ValueError('Los identificadores deben ser únicos y no estar vacíos.')
    resultado['error_reconstruccion'] = error_reconstruccion(modelo, resultado)
    resultado['prediccion_anomalia'] = (resultado['error_reconstruccion'] > modelo.umbral).astype(int)
    resultado['estado'] = np.where(resultado['prediccion_anomalia'] == 1,
                                   'Revisión requerida', 'Dentro del patrón')
    return resultado


def detalle_registro(modelo, registro):
    z = modelo.escalador.transform(registro[VARIABLES])
    reconstruido = modelo.pca.inverse_transform(modelo.pca.transform(z))
    original_unidades = modelo.escalador.inverse_transform(reconstruido)
    return pd.DataFrame({'Variable': VARIABLES,
                         'Valor observado': registro[VARIABLES].iloc[0].to_numpy(),
                         'Reconstrucción': original_unidades[0],
                         'Error cuadrático estandarizado': (z[0] - reconstruido[0]) ** 2})
