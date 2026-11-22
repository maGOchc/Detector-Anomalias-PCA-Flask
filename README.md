# Versión web HTML del detector PCA

Esta versión sustituye la interfaz Streamlit por una interfaz HTML/CSS/JavaScript servida por Flask. El motor matemático `motor_pca.py` se conserva y sigue realizando el entrenamiento, calibración, detección, evaluación y cálculo de contribuciones.

## Ejecutar en Visual Studio Code

1. Abre esta carpeta (`03 Codigo`) en VS Code.
2. Abre Terminal → Nueva terminal.
3. Ejecuta:

```bash
python -m pip install -r requirements_web.txt
python web_app.py
```

4. Abre `http://127.0.0.1:5000` en el navegador.
5. Para detener el servidor, vuelve a la terminal y presiona `Ctrl + C`.

En Windows también puedes hacer doble clic en `INICIAR_WEB.bat`.

## Funciones disponibles

- Ajuste de parámetros de simulación y entrenamiento PCA.
- Generación de tráfico sintético y detección de anomalías.
- Carga de CSV y evaluación opcional si contiene `etiqueta_real`.
- Visualización de proyección PCA, errores y umbral.
- Consulta de registros y descarga de resultados y alertas CSV.
- Métricas de evaluación y matriz de confusión cuando hay etiquetas reales.

Los gráficos utilizan Plotly desde CDN, por lo que se requiere conexión a internet para cargarlos. El servidor Flask debe permanecer ejecutándose mientras se utiliza la página. No es un HTML autónomo: Python sigue ejecutando el modelo PCA en el servidor.
