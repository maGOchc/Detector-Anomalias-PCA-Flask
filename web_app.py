"""Interfaz web HTML para el detector PCA. Conserva motor_pca.py sin modificar."""
from flask import Flask, render_template, request, jsonify, Response
import pandas as pd
import numpy as np
import io
from motor_pca import ejecutar_experimento, detectar, evaluar, coordenadas_pca, detalle_registro

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024


def json_records(df):
    return df.replace({np.nan: None}).to_dict(orient='records')


def run_analysis(params, uploaded=None):
    n = int(params.get('n', 1200)); na = int(params.get('na', 120))
    seed = int(params.get('seed', 42)); variance = float(params.get('variance', .80))
    quantile = float(params.get('quantile', .99)); intensity = float(params.get('intensity', 1.0))
    high = int(params.get('high', 10))
    _, train, cal, test, model, simulated, metrics = ejecutar_experimento(
        n, na, seed, variance, quantile, intensity, high)
    if uploaded is not None:
        try:
            incoming = pd.read_csv(uploaded, sep=None, engine='python', encoding='utf-8-sig')
            results = detectar(model, incoming)
            metrics_out = evaluar(model, results)[1] if 'etiqueta_real' in results else None
            source = 'Archivo CSV'
        except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
            raise ValueError(str(exc))
    else:
        results, metrics_out, source = simulated, metrics, 'Prueba simulada'
    coords = coordenadas_pca(model, results)
    points = coords.assign(estado=results.estado.values, id_registro=results.id_registro.values)
    ordered = results.assign(orden=np.arange(1, len(results)+1))
    details = None
    alerts = results[results.prediccion_anomalia == 1].sort_values('error_reconstruccion', ascending=False)
    if not alerts.empty:
        details = detalle_registro(model, alerts.iloc[[0]])
    return {
        'source': source, 'metrics': metrics_out,
        'summary': {'components': int(model.pca.n_components_),
                    'variance': float(model.pca.explained_variance_ratio_.sum()),
                    'alerts': int(results.prediccion_anomalia.sum()),
                    'threshold': float(model.umbral), 'quantile': quantile,
                    'train': len(train), 'calibration': len(cal), 'test': len(test)},
        'rows': json_records(results), 'example': json_records(test), 'points': json_records(points),
        'errors': json_records(ordered[['orden','error_reconstruccion','estado','id_registro']]),
        'details': json_records(details) if details is not None else [],
        'evaluation': metrics_out,
    }


@app.get('/')
def index():
    return render_template('index.html')


@app.post('/api/analyze')
def analyze():
    try:
        payload = request.form if request.form else (request.get_json(silent=True) or {})
        uploaded = request.files.get('file')
        result = run_analysis(payload, uploaded.stream if uploaded else None)
        return jsonify(result)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400


@app.post('/api/detail')
def detail():
    try:
        p = request.get_json(force=True)
        _, _, _, _, model, simulated, _ = ejecutar_experimento(
            int(p.get('n',1200)), int(p.get('na',120)), int(p.get('seed',42)),
            float(p.get('variance',.80)), float(p.get('quantile',.99)),
            float(p.get('intensity',1.0)), int(p.get('high',10)))
        frame = pd.DataFrame([p['row']])
        data = detalle_registro(model, frame).sort_values('Error cuadrático estandarizado', ascending=False)
        return jsonify(json_records(data))
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400


@app.post('/api/download')
def download():
    try:
        payload = request.form
        uploaded = request.files.get('file')
        result = run_analysis(payload, uploaded.stream if uploaded else None)
        df = pd.DataFrame(result['rows'])
        output = io.StringIO(); df.to_csv(output, index=False)
        return Response('\ufeff' + output.getvalue(), mimetype='text/csv',
                        headers={'Content-Disposition':'attachment; filename=resultados_pca.csv'})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
