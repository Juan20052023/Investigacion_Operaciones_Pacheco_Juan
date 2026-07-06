import streamlit as st
import numpy as np
import sympy as sp
from scipy.optimize import minimize
import plotly.graph_objects as go
import pandas as pd
import itertools

# ============================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ============================================================
st.set_page_config(page_title="Programación No Lineal", layout="wide")

st.markdown("""
    <style>
    .main-header {
        background-color: #0F3050;
        padding: 10px 20px;
        border-radius: 8px;
        color: white;
        margin-bottom: 15px;
    }
    .main-header h1 { color: white; margin-bottom: 2px; font-size: 1.5rem;}
    .main-header h3 { color: #A0C0DF; margin-top: 0px; margin-bottom: 5px; font-size: 1rem; font-weight: normal;}
    .main-header p { color: #D0E0EF; font-size: 0.85rem; margin-bottom: 0px;}
    .stButton>button {
        background-color: #008C7A;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        width: 100%;
        margin-top: 15px;
    }
    .stButton>button:hover { background-color: #006B5D; color: white; }
    </style>
    <div class="main-header">
        <h1>PROGRAMACIÓN NO LINEAL</h1>
        <h3>Universidad de Cuenca</h3>
        <p><b>Materia:</b> Investigación de Operaciones | <b>Carrera:</b> Ingeniería en Telecomunicaciones | <b>Autores:</b> Juan Pacheco, Jean Sumba</p>
    </div>
""", unsafe_allow_html=True)

# ============================================================
# PARSERS Y MATEMÁTICA SIMBÓLICA
# ============================================================
def crear_funcion_1d(expr_str):
    x = sp.Symbol('x')
    expr = sp.sympify(expr_str)
    f_prime = sp.diff(expr, x)
    f_double_prime = sp.diff(f_prime, x)
    
    f_lamb = sp.lambdify(x, expr, "numpy")
    fp_lamb = sp.lambdify(x, f_prime, "numpy")
    fdp_lamb = sp.lambdify(x, f_double_prime, "numpy")
    
    return lambda v: float(f_lamb(v)), lambda v: float(fp_lamb(v)), lambda v: float(fdp_lamb(v)), expr, f_prime, f_double_prime

def crear_funcion_nd(expr_str, n):
    simbolos = [sp.Symbol(f'x{i+1}') for i in range(n)] if n > 3 else sp.symbols('x1 x2 x3')[:n]
    expr = sp.sympify(expr_str)
    grad_expr = [sp.diff(expr, s) for s in simbolos]
    hess_expr = [[sp.diff(g, s2) for s2 in simbolos] for g in grad_expr]

    f_lamb = sp.lambdify(simbolos, expr, "numpy")
    grad_lamb = [sp.lambdify(simbolos, g, "numpy") for g in grad_expr]
    hess_lamb = [[sp.lambdify(simbolos, h, "numpy") for h in fila] for fila in hess_expr]

    def f(x):
        return float(f_lamb(*x))

    def grad(x):
        return np.array([float(g(*x)) for g in grad_lamb])

    def hess(x):
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                H[i, j] = float(hess_lamb[i][j](*x))
        return H

    return f, grad, hess, expr, grad_expr

def parse_vector(text):
    return np.array([float(x.strip()) for x in text.split(',')])

def parse_matrix(text):
    return np.array([[float(x.strip()) for x in line.split(',')] for line in text.strip().split('\n')])

def parse_restrictions(text):
    lines = text.strip().split('\n')
    A = []
    b = []
    for line in lines:
        if not line.strip(): continue
        parts = line.split(',')
        coeffs = [float(p.strip()) for p in parts[:-2]]
        val = float(parts[-2].strip())
        signo = parts[-1].strip()
        
        if signo == '>=':
            A.append([-c for c in coeffs])
            b.append(-val)
        else:
            A.append(coeffs)
            b.append(val)
    return np.array(A), np.array(b)

# ============================================================
# ALGORITMOS DE OPTIMIZACIÓN
# ============================================================
# TEMA 1
def biseccion_1d(f_prime, a, b, tol, max_iter):
    hist = []
    c = a
    if f_prime(a) * f_prime(b) > 0:
        st.warning("El intervalo [a, b] no garantiza una raíz para f'(x) = 0 porque f'(a) y f'(b) tienen el mismo signo.")
        return c, hist
    
    for i in range(max_iter):
        c = (a + b) / 2
        fpc = f_prime(c)
        hist.append({'Iteración': i+1, 'a': a, 'b': b, 'x (c)': c, "f'(c)": fpc})
        if abs(fpc) < tol or (b - a) / 2 < tol:
            break
        if f_prime(a) * fpc < 0:
            b = c
        else:
            a = c
    return c, hist

def newton_1d(f_prime, f_double_prime, x0, tol, max_iter):
    hist = []
    x = x0
    for i in range(max_iter):
        fp = f_prime(x)
        fdp = f_double_prime(x)
        hist.append({'Iteración': i+1, 'x': x, "f'(x)": fp, "f''(x)": fdp})
        if abs(fp) < tol:
            break
        if fdp == 0:
            st.warning("La segunda derivada es cero, el método de Newton se detiene.")
            break
        x = x - fp / fdp
    return x, hist

# TEMA 2
def gradiente_nd(f, grad_f, x0, alpha, tol, max_iter, tipo="Minimizar"):
    hist = []
    x = np.array(x0, dtype=float)
    signo = -1 if tipo == "Minimizar" else 1
    for i in range(max_iter):
        g = np.array(grad_f(x))
        norm_g = np.linalg.norm(g)
        row = {'Iteración': i+1}
        for j, val in enumerate(x): row[f'x{j+1}'] = val
        row['f(x)'] = f(x)
        row['||grad||'] = norm_g
        hist.append(row)
        
        if norm_g < tol: break
        x = x + signo * alpha * g
    return x, hist

def newton_nd(f, grad_f, hess_f, x0, tol, max_iter, tipo="Minimizar"):
    hist = []
    x = np.array(x0, dtype=float)
    for i in range(max_iter):
        g = np.array(grad_f(x))
        norm_g = np.linalg.norm(g)
        row = {'Iteración': i+1}
        for j, val in enumerate(x): row[f'x{j+1}'] = val
        row['f(x)'] = f(x)
        row['||grad||'] = norm_g
        hist.append(row)
        
        if norm_g < tol: break
        H = hess_f(x)
        try:
            p = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            p = -np.linalg.pinv(H) @ g
            
        if tipo == "Maximizar": p = -p 
        x = x + p
    return x, hist

# TEMA 3
def vertices_factibles(f, A, b_vec, tipo="Minimizar"):
    n = A.shape[1]
    m = A.shape[0]
    vertices = []
    
    for indices in itertools.combinations(range(m), min(n, m)):
        A_sub = A[list(indices), :]
        b_sub = b_vec[list(indices)]
        if A_sub.shape[0] == n and np.linalg.matrix_rank(A_sub) == n:
            x = np.linalg.solve(A_sub, b_sub)
            if np.all(A @ x <= b_vec + 1e-6):
                vertices.append(x)
                
    unique_vertices = []
    for v in vertices:
        if not any(np.linalg.norm(v - uv) < 1e-6 for uv in unique_vertices):
            unique_vertices.append(v)
            
    hist = []
    for i, v in enumerate(unique_vertices):
        row = {'Vértice': i+1}
        for j, val in enumerate(v): row[f'x{j+1}'] = val
        row['f(x)'] = f(v)
        hist.append(row)
    
    if not hist: return None, []
    opt = min(hist, key=lambda d: d['f(x)']) if tipo == "Minimizar" else max(hist, key=lambda d: d['f(x)'])
    return np.array([opt[f'x{j+1}'] for j in range(n)]), hist

def proyectar(y, A, b_vec):
    res = minimize(lambda x: np.linalg.norm(x - y)**2, x0=y, constraints={'type': 'ineq', 'fun': lambda x: b_vec - A @ x})
    return res.x

def gradiente_proyectado(f, grad_f, A, b_vec, x0, alpha, tol, max_iter, tipo="Minimizar"):
    hist = []
    x = np.array(x0, dtype=float)
    x = proyectar(x, A, b_vec)
    signo = -1 if tipo == "Minimizar" else 1
    
    for i in range(max_iter):
        g = np.array(grad_f(x))
        row = {'Iteración': i+1}
        for j, val in enumerate(x): row[f'x{j+1}'] = val
        row['f(x)'] = f(x)
        hist.append(row)
        
        y = x + signo * alpha * g
        x_new = proyectar(y, A, b_vec)
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
        
    return x, hist

# TEMA 4
def resolver_cuadratica(Q, c, const, A, b_vec, tipo="Minimizar"):
    def f_obj(x):
        val = 0.5 * x.T @ Q @ x + c.T @ x + const
        return val if tipo == "Minimizar" else -val
    
    n = Q.shape[0]
    res = minimize(f_obj, x0=np.zeros(n), constraints={'type': 'ineq', 'fun': lambda x: b_vec - A @ x})
    x_opt = res.x
    f_opt = 0.5 * x_opt.T @ Q @ x_opt + c.T @ x_opt + const
    return x_opt, f_opt

# ============================================================
# FUNCIONES DE GRAFICACIÓN
# ============================================================
def graficar_1d(f, x_opt, a, b):
    xs = np.linspace(min(a, x_opt) - 2, max(b, x_opt) + 2, 400)
    ys = [f(xi) for xi in xs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name='f(x)', line=dict(color='#0F3050', width=2)))
    fig.add_trace(go.Scatter(x=[x_opt], y=[f(x_opt)], mode='markers', name='Óptimo', marker=dict(color='red', size=10)))
    fig.update_layout(title="Gráfica de la función", xaxis_title="x", yaxis_title="f(x)")
    return fig

def graficar_nd(f, n, x_opt, hist, A=None, b_vec=None):
    if n != 2:
        return None
    
    margin = 5
    x_range = np.linspace(x_opt[0] - margin, x_opt[0] + margin, 100)
    y_range = np.linspace(x_opt[1] - margin, x_opt[1] + margin, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            val = f([X[i, j], Y[i, j]])
            if A is not None and b_vec is not None:
                if any(A @ np.array([X[i, j], Y[i, j]]) - b_vec > 1e-6):
                    Z[i, j] = np.nan
                else:
                    Z[i, j] = val
            else:
                Z[i, j] = val

    fig = go.Figure(data=go.Contour(x=x_range, y=y_range, z=Z, colorscale='Viridis', contours_coloring='lines'))
    
    if hist and 'x1' in hist[0]:
        hx = [h['x1'] for h in hist]
        hy = [h['x2'] for h in hist]
        fig.add_trace(go.Scatter(x=hx, y=hy, mode='lines+markers', name='Iteraciones', marker=dict(color='red', size=6)))
    
    fig.add_trace(go.Scatter(x=[x_opt[0]], y=[x_opt[1]], mode='markers', name='Óptimo', marker=dict(color='black', size=10, symbol='star')))
    fig.update_layout(title="Curvas de Nivel y Región Factible", xaxis_title="x1", yaxis_title="x2")
    return fig

# ============================================================
# INTERFAZ PRINCIPAL (Panel Lateral y Lógica)
# ============================================================
st.sidebar.markdown("### TEMAS")
opcion = st.sidebar.radio("Seleccione el método:", [
    "1. Una variable",
    "2. Varias variables",
    "3. Restringida linealmente",
    "4. Optimización cuadrática"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### Estado del programa")
st.sidebar.info(f"Módulo Activo:\n\n{opcion}")

if opcion.startswith("1"):
    st.markdown("## 1. Optimización no restringida de una sola variable")
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.markdown("#### Datos del problema")
        expr_str = st.text_input("Función f(x):", value="x**3 - 6*x**2 + 9*x + 1")
        metodo = st.selectbox("Método:", ["Bisección", "Newton"])
        tipo = st.selectbox("Tipo:", ["Minimizar", "Maximizar"])
        c1, c2 = st.columns(2)
        a = c1.number_input("a:", value=0.0)
        b = c2.number_input("b:", value=3.0)
        x0 = st.number_input("x0 Newton:", value=1.5)
        c3, c4 = st.columns(2)
        tol = c3.number_input("Tolerancia:", value=0.0001, format="%f")
        max_iter = c4.number_input("Iteraciones:", value=50, step=1)
        btn_resolver = st.button("Resolver")
        
    with col2:
        if btn_resolver:
            f, f_prime, f_double_prime, expr, expr_p, expr_dp = crear_funcion_1d(expr_str)
            
            if metodo == "Bisección":
                x_opt, hist = biseccion_1d(f_prime, a, b, tol, max_iter)
            else:
                x_opt, hist = newton_1d(f_prime, f_double_prime, x0, tol, max_iter)
                
            f_opt = f(x_opt)
            segunda_der = f_double_prime(x_opt)
            if segunda_der > 0: clasificacion = "Mínimo local"
            elif segunda_der < 0: clasificacion = "Máximo local"
            else: clasificacion = "No concluyente"
            
            st.success("Cálculo finalizado exitosamente.")
            t1, t2, t3 = st.tabs(["Respuesta final", "Tabla de iteraciones", "Gráfica"])
            with t1:
                st.markdown(f"**Método:** {metodo}")
                st.markdown(f"**Objetivo seleccionado:** {tipo}")
                st.markdown(f"**f(x) =** {expr}")
                st.markdown(f"**f'(x) =** {expr_p}")
                st.markdown(f"**f''(x) =** {expr_dp}")
                st.markdown(f"**x óptimo aproximado =** {x_opt:.8f}")
                st.markdown(f"**f(x óptimo) =** {f_opt:.8f}")
                st.markdown(f"**f''(x óptimo) =** {segunda_der:.8f}")
                st.markdown(f"**Clasificación:** {clasificacion}")
            with t2:
                if hist: st.dataframe(pd.DataFrame(hist), use_container_width=True)
            with t3:
                st.plotly_chart(graficar_1d(f, x_opt, a, b), use_container_width=True)

elif opcion.startswith("2"):
    st.markdown("## 2. Optimización no restringida de varias variables")
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.markdown("#### Datos del problema")
        n = st.selectbox("Número de variables:", [2, 3])
        expr_str = st.text_input("Función:", value="x1**2 + x2**2 - 4*x1 - 6*x2 + 13" if n==2 else "x1**2 + x2**2 + x3**2")
        metodo = st.selectbox("Método:", ["Gradiente", "Newton"])
        tipo = st.selectbox("Tipo:", ["Minimizar", "Maximizar"])
        p0_str = st.text_input("Punto inicial (separado por comas):", value="0,0" if n==2 else "0,0,0")
        alpha = st.number_input("Alpha:", value=0.1)
        c1, c2 = st.columns(2)
        tol = c1.number_input("Tolerancia:", value=0.0001, format="%f")
        max_iter = c2.number_input("Iteraciones:", value=50, step=1)
        btn_resolver = st.button("Resolver")
        
    with col2:
        if btn_resolver:
            x0 = parse_vector(p0_str)
            f, grad_f, hess_f, expr, grad_expr = crear_funcion_nd(expr_str, n)
            
            if metodo == "Gradiente":
                x_opt, hist = gradiente_nd(f, grad_f, x0, alpha, tol, max_iter, tipo)
            else:
                x_opt, hist = newton_nd(f, grad_f, hess_f, x0, tol, max_iter, tipo)
                
            f_opt = f(x_opt)
            H = hess_f(x_opt)
            eigvals = np.linalg.eigvals(H)
            if np.all(eigvals > 0): clasificacion = "Mínimo local (Definida positiva)"
            elif np.all(eigvals < 0): clasificacion = "Máximo local (Definida negativa)"
            elif np.any(eigvals > 0) and np.any(eigvals < 0): clasificacion = "Punto silla (Indefinida)"
            else: clasificacion = "No concluyente"
            
            st.success("Cálculo finalizado exitosamente.")
            t1, t2, t3 = st.tabs(["Respuesta final", "Tabla de iteraciones", "Gráfica"])
            with t1:
                st.markdown(f"**Método:** {metodo}")
                st.markdown(f"**Objetivo:** {tipo}")
                st.markdown(f"**x óptimo =** {np.round(x_opt, 6)}")
                st.markdown(f"**f(x óptimo) =** {f_opt:.8f}")
                st.markdown(f"**Clasificación Hessiana:** {clasificacion}")
            with t2:
                if hist: st.dataframe(pd.DataFrame(hist), use_container_width=True)
            with t3:
                fig = graficar_nd(f, n, x_opt, hist)
                if fig: st.plotly_chart(fig, use_container_width=True)
                else: st.info("La visualización topológica solo está disponible para 2 variables.")

elif opcion.startswith("3"):
    st.markdown("## 3. Optimización restringida linealmente")
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.markdown("#### Datos del problema")
        n = st.selectbox("Número variables:", [2, 3])
        expr_str = st.text_input("Función:", value="x1*x2 + 2*x1 + x2")
        metodo = st.selectbox("Método:", ["Vértices factibles", "Gradiente proyectado"])
        tipo = st.selectbox("Tipo:", ["Maximizar", "Minimizar"])
        st.markdown("Restricciones (Ej. 1,0,4,<= que es 1*x1 + 0*x2 <= 4):")
        restricciones_str = st.text_area("", value="1,0,4,<=\n0,1,6,<=\n3,2,18,<=")
        p0_str = st.text_input("Punto inicial:", value="1,1" if n==2 else "1,1,1")
        alpha = st.number_input("Alpha:", value=0.15)
        c1, c2 = st.columns(2)
        tol = c1.number_input("Tol:", value=0.0001, format="%f")
        max_iter = c2.number_input("Iter.:", value=80, step=1)
        btn_resolver = st.button("Resolver")
        
    with col2:
        if btn_resolver:
            x0 = parse_vector(p0_str)
            A, b_vec = parse_restrictions(restricciones_str)
            f, grad_f, _, expr, _ = crear_funcion_nd(expr_str, n)
            
            if metodo == "Vértices factibles":
                x_opt, hist = vertices_factibles(f, A, b_vec, tipo)
                if x_opt is None: st.error("No se encontraron vértices factibles con el sistema provisto.")
            else:
                x_opt, hist = gradiente_proyectado(f, grad_f, A, b_vec, x0, alpha, tol, max_iter, tipo)
                
            if x_opt is not None:
                f_opt = f(x_opt)
                st.success("Cálculo finalizado exitosamente.")
                t1, t2, t3 = st.tabs(["Respuesta final", "Tabla / Candidatos", "Gráfica"])
                with t1:
                    st.markdown(f"**Método:** {metodo}")
                    st.markdown(f"**Objetivo:** {tipo}")
                    st.markdown(f"**x óptimo =** {np.round(x_opt, 6)}")
                    st.markdown(f"**f(x óptimo) =** {f_opt:.8f}")
                with t2:
                    if hist: st.dataframe(pd.DataFrame(hist), use_container_width=True)
                with t3:
                    fig = graficar_nd(f, n, x_opt, hist if metodo=="Gradiente proyectado" else [], A, b_vec)
                    if fig: st.plotly_chart(fig, use_container_width=True)
                    else: st.info("La visualización topológica solo está disponible para 2 variables.")

elif opcion.startswith("4"):
    st.markdown("## 4. Optimización cuadrática")
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.markdown("#### Datos del problema cuadrático 2D")
        q_str = st.text_area("Q 2x2:", value="2,0\n0,2")
        c_str = st.text_input("c = [c1, c2]:", value="-4,-6")
        constante = st.number_input("Constante:", value=13.0)
        tipo = st.selectbox("Tipo:", ["Minimizar", "Maximizar"])
        restricciones_str = st.text_area("Restricciones 2D:", value="1,0,4,<=\n0,1,6,<=\n3,2,18,<=")
        btn_resolver = st.button("Resolver cuadrática")
        
    with col2:
        if btn_resolver:
            Q = parse_matrix(q_str)
            Q = (Q + Q.T) / 2 # Aseguramos simetría
            c_vec = parse_vector(c_str)
            A, b_vec = parse_restrictions(restricciones_str)
            
            x_opt, f_opt = resolver_cuadratica(Q, c_vec, constante, A, b_vec, tipo)
            
            def f_eval(x): return 0.5 * np.array(x).T @ Q @ np.array(x) + c_vec.T @ np.array(x) + constante
            
            eigvals = np.linalg.eigvals(Q)
            if np.all(eigvals > 0): clasificacion = "Q definida positiva: función convexa. Recomendado para minimizar."
            elif np.all(eigvals < 0): clasificacion = "Q definida negativa: función cóncava. Recomendado para maximizar."
            else: clasificacion = "Q indefinida o semidefinida: se revisan candidatos factibles."
            
            st.success("Cálculo finalizado exitosamente.")
            t1, t2 = st.tabs(["Respuesta final", "Gráfica"])
            with t1:
                st.markdown(f"**Tipo:** {tipo}")
                st.markdown(f"**Modelo:** f(x,y)=1/2*[x y]*Q*[x;y] + c'*[x;y] + k")
                st.markdown(f"**x óptimo aproximado =** {x_opt[0]:.8f}")
                st.markdown(f"**y óptimo aproximado =** {x_opt[1]:.8f}")
                st.markdown(f"**f(x,y) óptimo =** {f_opt:.8f}")
                st.markdown(f"**Análisis de Q:** {clasificacion}")
            with t2:
                fig = graficar_nd(f_eval, 2, x_opt, [], A, b_vec)
                st.plotly_chart(fig, use_container_width=True)