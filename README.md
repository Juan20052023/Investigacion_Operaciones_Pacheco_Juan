# Optimización No Lineal — Programa Interactivo (Streamlit)

Autores: Juan Pacheco, Jean Sumba
Institución: Universidad de Cuenca — Julio 2026

## Contenido
- `app.py` — aplicación principal.
- `requirements.txt` — dependencias necesarias.
- `.streamlit/config.toml` — tema visual (fondo blanco, texto negro).

## Módulos y algoritmo implementado
1. Optimización no restringida de una variable → **Sección Dorada**
2. Optimización no restringida de varias variables → **Newton-Raphson**
3. Optimización restringida linealmente → **Frank-Wolfe**
4. Optimización cuadrática → **Conjunto Activo (KKT)**

## Instalación y ejecución local

```bash
python -m venv venv
source venv/bin/activate      # En Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

La aplicación se abrirá en `http://localhost:8501`.

## Notas importantes

- **Módulo 1 (Sección Dorada):** requiere un intervalo `[a, b]` donde la función
  sea unimodal (un único mínimo dentro del intervalo).
- **Módulo 3 (Frank-Wolfe):** requiere que la región factible `A x ≤ b` sea un
  politopo **acotado**. Si olvida acotar alguna variable (por ejemplo, no incluye
  `x_i ≥ 0`), el programa mostrará un mensaje de error explicando cómo corregirlo,
  en lugar de devolver una respuesta incorrecta en silencio.
- Cada módulo compara el resultado del algoritmo propio contra SciPy
  (BFGS o SLSQP según el caso) como referencia de validación.
