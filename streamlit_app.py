import streamlit as st
import numpy as np
import sympy as sp
from scipy.optimize import linprog, minimize_scalar, minimize
import plotly.graph_objects as go

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(page_title="Optimización No Lineal", layout="wide", page_icon="📐")

st.title("Investigación de Operaciones — Juan Pacheco, Jean Sumba")
st.markdown(
    "Programa informático para la solución de problemas de programación no lineal (PNL). "
    "Desarrollado en Python con Streamlit. Cada módulo implementa el algoritmo solicitado "
    "en la guía del proyecto: **Sección Dorada**, **Newton-Raphson**, **Frank-Wolfe** y "
    "**Conjunto Activo (KKT)**."
)

# ============================================================
# FUNCIONES AUXILIARES Y MATEMÁTICAS
# ============================================================

def crear_funcion(expr_str, nombres_variables):
    """
    Convierte un string matemático en funciones evaluables de f(x) y su gradiente.
    """
    simbolos = [sp.Symbol(n) for n in nombres_variables]
    expr = sp.sympify(expr_str)
    grad_expr = [sp.diff(expr, s) for s in simbolos]

    f_lamb = sp.lambdify(simbolos, expr, "numpy")
    grad_lamb = [sp.lambdify(simbolos, g, "numpy") for g in grad_expr]

    def f(x):
        val = f_lamb(*x)
        return float(val)

    def grad(x):
        return np.array([float(g(*x)) for g in grad_lamb])

    return f, grad


def crear_funcion_hessiana(expr_str, nombres_variables):
    """
    Igual que crear_funcion, pero además construye la matriz Hessiana
    analítica (necesaria para Newton-Raphson multivariable).
    """
    simbolos = [sp.Symbol(n) for n in nombres_variables]
    expr = sp.sympify(expr_str)
    grad_expr = [sp.diff(expr, s) for s in simbolos]
    hess_expr = [[sp.diff(g, s2) for s2 in simbolos] for g in grad_expr]

    f_lamb = sp.lambdify(simbolos, expr, "numpy")
    grad_lamb = [sp.lambdify(simbolos, g, "numpy") for g in grad_expr]
    hess_lamb = [[sp.lambdify(simbolos, h, "numpy") for h in fila] for fila in hess_expr]

    def f(x):
        val = f_lamb(*x)
        return float(val)

    def grad(x):
        return np.array([float(g(*x)) for g in grad_lamb])

    def hess(x):
        n = len(x)
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                H[i, j] = float(hess_lamb[i][j](*x))
        return H

    return f, grad, hess


def punto_factible_inicial(A, b, n):
    """
    Encuentra un punto inicial que cumpla A x <= b usando programación lineal
    (problema de fase 1 con función objetivo nula).
    """
    res = linprog(np.zeros(n), A_ub=A, b_ub=b, bounds=[(None, None)] * n, method="highs")
    return res.x if res.success else None


# ============================================================
# ALGORITMOS DE OPTIMIZACIÓN (uno por módulo, según la guía)
# ============================================================

def busqueda_seccion_dorada(f, a, b, tol=1e-6, max_iter=200):
    """
    Módulo 1: Optimización NO restringida de UNA variable.
    Método de la Sección Dorada (Golden Section Search) sobre un
    intervalo [a, b] en el que se asume que f es unimodal.
    """
    if a >= b:
        raise ValueError("El límite inferior 'a' debe ser menor que el límite superior 'b'.")

    razon_aurea = (np.sqrt(5) - 1) / 2  # ≈ 0.618
    c = b - razon_aurea * (b - a)
    d = a + razon_aurea * (b - a)

    historial = [np.array([(a + b) / 2.0])]
    it = 0
    while abs(b - a) > tol and it < max_iter:
        fc, fd = f([c]), f([d])
        if fc < fd:
            b = d
        else:
            a = c
        c = b - razon_aurea * (b - a)
        d = a + razon_aurea * (b - a)
        historial.append(np.array([(a + b) / 2.0]))
        it += 1

    x_opt = np.array([(a + b) / 2.0])
    return x_opt, f(x_opt), historial


def newton_raphson(f, grad_f, hess_f, x0, tol=1e-6, max_iter=100):
    """
    Módulo 2: Optimización NO restringida de VARIAS variables.
    Método de Newton-Raphson: x_(k+1) = x_k - H(x_k)^-1 * grad f(x_k).
    Se añade una búsqueda de línea (backtracking/Armijo) sobre la
    dirección de Newton para garantizar el descenso (Newton amortiguado),
    lo cual evita divergencia cuando el punto inicial está lejos del óptimo
    o la Hessiana no es definida positiva en esa zona.
    """
    x = np.array(x0, dtype=float)
    historial = [x.copy()]

    for _ in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break

        H = hess_f(x)
        try:
            p = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            p = -np.linalg.pinv(H) @ g  # Hessiana singular: usar pseudo-inversa

        # Si la dirección no es de descenso, se recurre al descenso por gradiente
        if np.dot(g, p) > 0:
            p = -g

        alpha, c1, rho = 1.0, 1e-4, 0.5
        while f(x + alpha * p) > f(x) + c1 * alpha * np.dot(g, p) and alpha > 1e-10:
            alpha *= rho

        x_nuevo = x + alpha * p
        historial.append(x_nuevo.copy())

        if np.linalg.norm(x_nuevo - x) < tol:
            x = x_nuevo
            break
        x = x_nuevo

    return x, f(x), historial


def frank_wolfe(f, grad_f, A, b, x0, max_iter=100, tol=1e-6):
    """
    Módulo 3: Optimización RESTRINGIDA LINEALMENTE.
    Método de Frank-Wolfe (gradiente condicional) para min f(x) s.a. A x <= b.
    """
    x = np.array(x0, dtype=float)
    n = len(x)
    historial = [x.copy()]

    for _ in range(max_iter):
        c = grad_f(x)
        res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * n, method="highs")
        if not res.success:
            if res.status == 3:  # LP no acotada
                raise ValueError(
                    "El subproblema lineal de Frank-Wolfe resultó no acotado. "
                    "Este método requiere que la región factible (A x ≤ b) sea un "
                    "politopo acotado. Agregue restricciones que acoten todas las "
                    "variables (por ejemplo, x_i ≥ 0 expresado como -x_i ≤ 0, "
                    "o cotas superiores adicionales)."
                )
            break
        y = res.x
        d = y - x

        if np.linalg.norm(d) < tol:
            break

        def phi(t):
            return f(x + t * d)

        res_line = minimize_scalar(phi, bounds=(0, 1), method="bounded")
        t_star = res_line.x

        x_nuevo = x + t_star * d
        historial.append(x_nuevo.copy())

        if np.linalg.norm(x_nuevo - x) < tol:
            x = x_nuevo
            break
        x = x_nuevo

    return x, f(x), historial


def resolver_qp_activo(Q, c, A, b, x0, max_iter=50, tol=1e-8):
    """
    Módulo 4: Optimización CUADRÁTICA.
    Minimiza 1/2 x^T Q x + c^T x sujeto a A x <= b usando el método
    del Conjunto Activo (Active Set / condiciones KKT).
    """
    x = np.array(x0, dtype=float)
    n = len(x)
    m = A.shape[0]
    activo = [i for i in range(m) if abs(A[i] @ x - b[i]) < 1e-8]
    historial = [x.copy()]

    for _ in range(max_iter):
        g = Q @ x + c
        Aw = A[activo] if activo else np.zeros((0, n))
        k = len(activo)

        M = np.zeros((n + k, n + k))
        M[:n, :n] = Q
        if k > 0:
            M[:n, n:] = Aw.T
            M[n:, :n] = Aw
        rhs = np.concatenate([-g, np.zeros(k)])

        sol = np.linalg.lstsq(M, rhs, rcond=None)[0]
        p = sol[:n]
        lam = sol[n:] if k > 0 else np.array([])

        if np.linalg.norm(p) < tol:
            if k == 0 or np.all(lam >= -tol):
                break
            j = int(np.argmin(lam))
            activo.pop(j)
            continue

        alpha = 1.0
        bloqueante = None
        for i in range(m):
            if i in activo:
                continue
            Ap = A[i] @ p
            if Ap > tol:
                ai = (b[i] - A[i] @ x) / Ap
                if ai < alpha:
                    alpha = ai
                    bloqueante = i

        x = x + alpha * p
        historial.append(x.copy())

        if bloqueante is not None and alpha < 1.0:
            activo.append(bloqueante)

    return x, 0.5 * x @ Q @ x + c @ x, historial


# ============================================================
# FUNCIONES DE GRAFICACIÓN (tema claro para consistencia visual)
# ============================================================

def graficar_1d(f, x_opt, historial):
    xs = np.linspace(x_opt[0] - 5, x_opt[0] + 5, 200)
    ys = [f([xi]) for xi in xs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f(x)", line=dict(color="blue")))

    hx = [p[0] for p in historial]
    hy = [f([p[0]]) for p in historial]
    fig.add_trace(go.Scatter(x=hx, y=hy, mode="lines+markers", name="Trayectoria Iterativa",
                              marker=dict(color="red", size=8)))

    fig.update_layout(title="Visualización 1D - Convergencia", xaxis_title="x1", yaxis_title="f(x1)")
    st.plotly_chart(fig, use_container_width=True)


def graficar_2d(f, x_opt, historial):
    x_range = np.linspace(x_opt[0] - 5, x_opt[0] + 5, 60)
    y_range = np.linspace(x_opt[1] - 5, x_opt[1] + 5, 60)
    Z = np.array([[f([xi, yi]) for xi in x_range] for yi in y_range])

    fig = go.Figure(data=go.Contour(x=x_range, y=y_range, z=Z, colorscale="Viridis",
                                     contours_coloring="lines"))

    hx = [p[0] for p in historial]
    hy = [p[1] for p in historial]
    fig.add_trace(go.Scatter(x=hx, y=hy, mode="lines+markers", name="Trayectoria Iterativa",
                              marker=dict(color="red", size=8)))

    fig.update_layout(title="Curvas de Nivel 2D - Convergencia (proyección x1 vs x2)",
                       xaxis_title="x1", yaxis_title="x2")
    st.plotly_chart(fig, use_container_width=True)


def graficar_proyeccion_2d(f, x_opt, historial):
    """
    Proyecta y grafica una función de N variables (N >= 2) sobre el plano
    x1-x2, fijando el resto de las variables en el valor óptimo encontrado.
    Permite reutilizar graficar_2d aunque el problema tenga 3 variables.
    """
    x_opt = np.asarray(x_opt, dtype=float)

    def f_proyectada(par):
        x_completo = x_opt.copy()
        x_completo[0], x_completo[1] = par[0], par[1]
        return f(x_completo)

    historial_2d = [np.asarray(p[:2], dtype=float) for p in historial]
    graficar_2d(f_proyectada, x_opt[:2], historial_2d)


# ============================================================
# INTERFAZ PRINCIPAL (SIDEBAR Y MÓDULOS)
# ============================================================

st.sidebar.header("Módulos del Sistema")
opcion = st.sidebar.radio("Seleccione el tipo de problema:", [
    "1. No restringida (1 variable) — Sección Dorada",
    "2. No restringida (Varias variables) — Newton-Raphson",
    "3. Restringida linealmente — Frank-Wolfe",
    "4. Cuadrática — Conjunto Activo",
])

# ------------------------------------------------------------
# MÓDULO 1: OPTIMIZACIÓN NO RESTRINGIDA (1 VARIABLE) — SECCIÓN DORADA
# ------------------------------------------------------------
if opcion.startswith("1"):
    st.header("1. Optimización no restringida de una variable (Sección Dorada)")
    expr_str = st.text_input("Función objetivo f(x1)", value="x1**2 - 4*x1 + 4")

    col_a, col_b, col_tol = st.columns(3)
    a = col_a.number_input("Límite inferior del intervalo (a)", value=-10.0)
    b = col_b.number_input("Límite superior del intervalo (b)", value=10.0)
    tol = col_tol.number_input("Tolerancia", value=1e-6, format="%.8f")

    st.caption("La Sección Dorada requiere un intervalo [a, b] donde f(x) sea unimodal "
               "(un único mínimo dentro del intervalo).")

    if st.button("Resolver Problema 1"):
        try:
            if a >= b:
                st.error("El límite inferior debe ser menor que el límite superior.")
            else:
                f, grad_f = crear_funcion(expr_str, ["x1"])
                x_opt, f_opt, historial = busqueda_seccion_dorada(f, a, b, tol=tol)

                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Óptimo encontrado:** x1 = {x_opt[0]:.4f}")
                    st.info(f"**f(x) óptimo:** {f_opt:.6f}")
                with col2:
                    res_scipy = minimize(f, [(a + b) / 2], method="BFGS")
                    st.warning(f"**SciPy (BFGS):** x1 = {res_scipy.x[0]:.4f} | f(x) = {res_scipy.fun:.6f}")
                    st.write(f"Diferencia de evaluación: {abs(f_opt - res_scipy.fun):.2e}")

                st.subheader("Historial de iteraciones")
                st.dataframe({"Iteración": range(len(historial)), "x1": [p[0] for p in historial]},
                             use_container_width=True)
                graficar_1d(f, x_opt, historial)
        except Exception as e:
            st.error(f"Ocurrió un error al resolver el problema: {e}")

# ------------------------------------------------------------
# MÓDULO 2: OPTIMIZACIÓN NO RESTRINGIDA (VARIAS VARIABLES) — NEWTON-RAPHSON
# ------------------------------------------------------------
elif opcion.startswith("2"):
    st.header("2. Optimización no restringida de varias variables (Newton-Raphson)")
    st.caption("Escriba f(x) usando hasta 3 variables: x1, x2, x3. Si su problema solo "
               "necesita 2 variables, simplemente no incluya x3 en la expresión "
               "(su valor inicial no afectará el resultado).")
    n = 3
    variables = ["x1", "x2", "x3"]
    expr_str = st.text_input("Función objetivo f(x)", value="x1**2 + x2**2 - x1*x2")

    st.write("Punto inicial x0:")
    cols_x0 = st.columns(n)
    x0 = np.array([cols_x0[i].number_input(f"x{i+1}_0", value=2.0, key=f"nr_x0_{i}") for i in range(n)])

    if st.button("Resolver Problema 2"):
        try:
            f, grad_f, hess_f = crear_funcion_hessiana(expr_str, variables)
            x_opt, f_opt, historial = newton_raphson(f, grad_f, hess_f, x0)

            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Óptimo encontrado:** x = {np.round(x_opt, 4)}")
                st.info(f"**f(x) óptimo:** {f_opt:.6f}")
            with col2:
                res_scipy = minimize(f, x0, method="BFGS")
                st.warning(f"**SciPy (BFGS):** x = {np.round(res_scipy.x, 4)} | f(x) = {res_scipy.fun:.6f}")
                st.write(f"Diferencia de evaluación: {abs(f_opt - res_scipy.fun):.2e}")

            st.subheader("Historial de iteraciones")
            st.dataframe({f"x{i+1}": [p[i] for p in historial] for i in range(n)})
            graficar_proyeccion_2d(f, x_opt, historial)
        except Exception as e:
            st.error(f"Ocurrió un error al resolver el problema: {e}")

# ------------------------------------------------------------
# MÓDULO 3: OPTIMIZACIÓN RESTRINGIDA LINEALMENTE — FRANK-WOLFE
# ------------------------------------------------------------
elif opcion.startswith("3"):
    st.header("3. Optimización restringida linealmente (Frank-Wolfe)")
    st.caption("Escriba f(x) usando hasta 3 variables: x1, x2, x3. Si su problema solo "
               "necesita 2 variables, deje los coeficientes de x3 en 0.")
    n = 3
    variables = ["x1", "x2", "x3"]
    expr_str = st.text_input("Función objetivo f(x)", value="x1**2 + x2**2 - x1*x2")

    st.subheader("Restricciones lineales  A x ≤ b")
    st.caption("Frank-Wolfe requiere que la región factible sea un politopo **acotado**. "
               "Incluya suficientes restricciones para cerrar la región en todas las "
               "direcciones (por ejemplo, x_i ≥ 0 escrito como -x_i ≤ 0, además de cotas "
               "superiores para cada variable).")
    num_restricciones = st.number_input("Número de restricciones", min_value=1, max_value=10, value=2)

    A, b = [], []
    for i in range(num_restricciones):
        cols = st.columns(n + 2)
        fila = [cols[j].number_input(f"a_{i+1}{j+1}", value=1.0, key=f"a_{i}_{j}") for j in range(n)]
        cols[n].markdown("<div style='text-align: center; margin-top: 30px;'>≤</div>",
                          unsafe_allow_html=True)
        bi = cols[n + 1].number_input(f"b_{i+1}", value=10.0, key=f"b_{i}")
        A.append(fila)
        b.append(bi)

    A = np.array(A)
    b = np.array(b)

    if st.button("Resolver Problema 3"):
        try:
            f, grad_f = crear_funcion(expr_str, variables)
            x0 = punto_factible_inicial(A, b, n)

            if x0 is None:
                st.error("No se encontró un punto factible inicial para las restricciones dadas.")
            else:
                x_opt, f_opt, historial = frank_wolfe(f, grad_f, A, b, x0)

                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Óptimo encontrado:** x = {np.round(x_opt, 4)}")
                    st.info(f"**f(x) óptimo:** {f_opt:.6f}")
                with col2:
                    cons = [{"type": "ineq", "fun": lambda x, i=i: b[i] - A[i] @ x} for i in range(len(b))]
                    res_scipy = minimize(f, x0, constraints=cons, method="SLSQP")
                    st.warning(f"**SciPy (SLSQP):** x = {np.round(res_scipy.x, 4)} | f(x) = {res_scipy.fun:.6f}")
                    st.write(f"Diferencia de evaluación: {abs(f_opt - res_scipy.fun):.2e}")

                st.subheader("Historial de iteraciones")
                st.dataframe({f"x{i+1}": [p[i] for p in historial] for i in range(n)})
                graficar_proyeccion_2d(f, x_opt, historial)
        except Exception as e:
            st.error(f"Ocurrió un error al resolver el problema: {e}")

# ------------------------------------------------------------
# MÓDULO 4: OPTIMIZACIÓN CUADRÁTICA — CONJUNTO ACTIVO
# ------------------------------------------------------------
elif opcion.startswith("4"):
    st.header("4. Optimización cuadrática (Conjunto Activo / KKT)")
    st.caption("Minimiza ½xᵀQx + cᵀx para hasta 3 variables. Si su problema solo necesita "
               "2 variables, deje la fila/columna de x3 en Q como 0 y c3 = 0.")
    n = 3

    st.subheader("Restricciones lineales  A x ≤ b")
    num_restricciones = st.number_input("Número de restricciones", min_value=1, max_value=10, value=2)

    A, b = [], []
    for i in range(num_restricciones):
        cols = st.columns(n + 2)
        fila = [cols[j].number_input(f"a_{i+1}{j+1}", value=1.0, key=f"q_a_{i}_{j}") for j in range(n)]
        cols[n].markdown("<div style='text-align: center; margin-top: 30px;'>≤</div>",
                          unsafe_allow_html=True)
        bi = cols[n + 1].number_input(f"b_{i+1}", value=10.0, key=f"q_b_{i}")
        A.append(fila)
        b.append(bi)

    A = np.array(A)
    b = np.array(b)

    st.subheader("Matriz Q (½ xᵀQx) y vector c (cᵀx)")
    Q = []
    st.write("Matriz Q:")
    for i in range(n):
        cols = st.columns(n)
        fila = [cols[j].number_input(f"Q_{i+1}{j+1}", value=2.0 if i == j else 0.0, key=f"Q_{i}_{j}")
                for j in range(n)]
        Q.append(fila)
    Q = np.array(Q)

    st.write("Vector c:")
    cols_c = st.columns(n)
    c = np.array([cols_c[i].number_input(f"c_{i+1}", value=0.0, key=f"c_{i}") for i in range(n)])

    if st.button("Resolver Problema 4"):
        try:
            x0 = punto_factible_inicial(A, b, n)

            if x0 is None:
                st.error("No se encontró un punto factible inicial para las restricciones dadas.")
            else:
                x_opt, f_opt, historial = resolver_qp_activo(Q, c, A, b, x0)

                def f_obj(x):
                    # Convertir siempre a arreglo numpy: si x llega como lista
                    # (p. ej. desde la grafica de contornos), "0.5 * x" fallaría
                    # con "can't multiply sequence by non-int of type 'float'".
                    x = np.asarray(x, dtype=float)
                    return 0.5 * x @ Q @ x + c @ x

                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Óptimo encontrado:** x = {np.round(x_opt, 4)}")
                    st.info(f"**f(x) óptimo:** {f_opt:.6f}")
                with col2:
                    cons = [{"type": "ineq", "fun": lambda x, i=i: b[i] - A[i] @ x} for i in range(len(b))]
                    res_scipy = minimize(f_obj, x0, constraints=cons, method="SLSQP")
                    st.warning(f"**SciPy (SLSQP):** x = {np.round(res_scipy.x, 4)} | f(x) = {res_scipy.fun:.6f}")
                    st.write(f"Diferencia de evaluación: {abs(f_opt - res_scipy.fun):.2e}")

                st.subheader("Historial de iteraciones")
                st.dataframe({f"x{i+1}": [p[i] for p in historial] for i in range(n)})
                graficar_proyeccion_2d(f_obj, x_opt, historial)
        except Exception as e:
            st.error(f"Ocurrió un error al resolver el problema: {e}")
