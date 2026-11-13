"""
Regresión logística programada desde cero con NumPy.

Se ajusta por máxima verosimilitud con el algoritmo de Newton-Raphson
(la misma regla de actualización que en las diapositivas de clase) y se
añade una penalización ridge opcional, controlada por el parámetro lam (λ):

    maximizar   ℓ(β) − λ·‖β‖²

Con lam = 0 tenemos exactamente la regresión logística vista en clase.
"""
import numpy as np


def sigmoide(z):
    """
    σ(z) = 1 / (1 + e^(−z)).

    Recortamos z al intervalo [−500, 500] porque, si z es muy negativo,
    e^(−z) es tan grande que el ordenador lo convierte en infinito y aparece
    un aviso de desbordamiento. Fuera de ese intervalo la sigmoide ya vale
    0 o 1 en la práctica, así que el resultado no cambia.
    """
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def anadir_columna_unos(X):
    """Añade una columna de unos a la izquierda para el intercepto β0 (matriz de diseño)."""
    return np.column_stack([np.ones(X.shape[0]), X])


class RegresionLogistica:
    """
    Regresión logística binaria (y = 0 o y = 1).

    Parámetros:
        lam      -> fuerza de la regularización ridge (λ). Con 0 no hay regularización.
        max_iter -> número máximo de iteraciones de Newton-Raphson.
        tol      -> paramos cuando ‖β_(t+1) − β_t‖ < tol (convergencia).
    """

    def __init__(self, lam=0.0, max_iter=50, tol=1e-8):
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol

    def log_verosimilitud(self, X1, y, beta):
        """ℓ(β) = Σ [ y·log(p) + (1 − y)·log(1 − p) ]  (fórmula de clase)."""
        p = sigmoide(X1 @ beta)
        eps = 1e-12  # evita log(0) cuando p vale exactamente 0 o 1
        return np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

    def ajustar(self, X, y):
        X1 = anadir_columna_unos(X)          # matriz de diseño con columna de unos
        n_parametros = X1.shape[1]
        beta = np.zeros(n_parametros)        # inicialización β = 0, como en clase

        # El intercepto β0 NO se penaliza (no tiene sentido obligarlo a ser
        # pequeño). Por eso la penalización lleva un 0 en la primera posición.
        penalizar = np.ones(n_parametros)
        penalizar[0] = 0.0

        # Guardamos la evolución para poder dibujarla después
        self.historial_paso = []     # ‖Δβ‖ en cada iteración
        self.historial_logver = []   # ℓ(β) en cada iteración
        self.historial_beta = []     # β después de cada iteración
        self.convergio = False
        self.error = None

        for t in range(self.max_iter):
            # 1) Probabilidades predichas p_t = σ(Xβ_t)
            p = sigmoide(X1 @ beta)

            # 2) Pesos W_t = diag(p(1 − p)). No construimos la matriz diagonal
            #    entera (sería n x n, enorme); basta con el vector de pesos.
            w = p * (1 - p)

            # 3) Gradiente: Xᵀ(y − p) − 2λβ   (el término −2λβ viene del ridge)
            gradiente = X1.T @ (y - p) - 2 * self.lam * penalizar * beta

            # 4) Matriz XᵀWX + 2λI. Multiplicar cada fila de X por su peso
            #    equivale a calcular W·X sin crear la matriz diagonal.
            XtWX = X1.T @ (X1 * w[:, None]) + 2 * self.lam * np.diag(penalizar)

            # 5) Paso de Newton: resolvemos (XᵀWX + 2λI)·paso = gradiente.
            #    Es lo mismo que paso = (XᵀWX + 2λI)⁻¹·gradiente, pero
            #    np.linalg.solve no calcula la inversa explícitamente y es más
            #    rápido y más preciso.
            try:
                paso = np.linalg.solve(XtWX, gradiente)
            except np.linalg.LinAlgError:
                # La matriz no se puede invertir (Hessiana singular)
                self.error = "Hessiana singular: no se puede calcular el paso de Newton"
                break

            beta = beta + paso
            self.historial_paso.append(np.linalg.norm(paso))
            self.historial_logver.append(self.log_verosimilitud(X1, y, beta))
            self.historial_beta.append(beta.copy())

            # 6) Criterio de parada: el cambio en β es despreciable
            if np.linalg.norm(paso) < self.tol:
                self.convergio = True
                break

        self.beta = beta
        self.n_iter = len(self.historial_paso)
        # Guardamos la última matriz para estudiar si estaba mal condicionada
        self.matriz_newton = XtWX
        return self

    def predecir_proba(self, X):
        """P(y = 1 | x) para cada fila de X."""
        return sigmoide(anadir_columna_unos(X) @ self.beta)

    def predecir(self, X, umbral=0.5):
        """Clase 1 si la probabilidad supera el umbral (frontera de decisión en z = 0)."""
        return (self.predecir_proba(X) >= umbral).astype(int)


class UnoContraResto:
    """
    Extiende la regresión logística binaria a varias clases.

    Para cada género se entrena una regresión logística que responde a
    "¿es de este género o no?". Para clasificar una canción se calcula la
    probabilidad que da cada uno de los modelos y se elige el género con la
    probabilidad más alta.
    """

    def __init__(self, lam=0.0, max_iter=50, tol=1e-8):
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol

    def ajustar(self, X, y):
        self.clases = np.unique(y)
        self.modelos = []
        for clase in self.clases:
            y_binaria = (y == clase).astype(float)   # 1 si es esta clase, 0 si no
            modelo = RegresionLogistica(self.lam, self.max_iter, self.tol)
            self.modelos.append(modelo.ajustar(X, y_binaria))
        return self

    def predecir_proba(self, X):
        """Matriz (canciones x clases) con la probabilidad que da cada modelo."""
        return np.column_stack([m.predecir_proba(X) for m in self.modelos])

    def predecir(self, X):
        indice_mayor = np.argmax(self.predecir_proba(X), axis=1)
        return self.clases[indice_mayor]


def exactitud(y_real, y_predicho):
    """Proporción de aciertos (accuracy)."""
    return np.mean(np.asarray(y_real) == np.asarray(y_predicho))
