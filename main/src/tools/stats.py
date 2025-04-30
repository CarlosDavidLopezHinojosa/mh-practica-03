import scipy.stats as hyp
import numpy as np

def mean(data, axis=None):
    """
    Calcula la media de un conjunto de datos.
    Args:
        data (list or np.array): Datos numéricos.
    Returns:
        float: Media de los datos.
    """
    return np.mean(data,axis=axis)

def stdev(data):
    """
    Calcula la desviación estándar de un conjunto de datos.
    Args:
        data (list or np.array): Datos numéricos.
    Returns:
        float: Desviación estándar de los datos.
    """
    return np.std(data)

def wilcoxon(data1, data2):
    """
    Realiza la prueba de Wilcoxon para muestras emparejadas.
    Args:
        data1 (list or np.array): Primer conjunto de datos.
        data2 (list or np.array): Segundo conjunto de datos.
    Returns:
        dict: Resultados de la prueba con p-valor y estadístico.
    """
    test = hyp.wilcoxon(data1, data2)
    return {'pvalue': test.pvalue.item(), 'stat-test': 'Wilcoxon'}

def friedman(data):
    """
    Realiza la prueba de Friedman para muestras relacionadas.
    Args:
        data (list of lists or np.array): Datos organizados en varias muestras.
    Returns:
        dict: Resultados de la prueba con p-valor y estadístico.
    """
    test = hyp.friedmanchisquare(*data)
    return {'pvalue': test.pvalue.item(), 'stat-test': 'Friedman'}  
    
def multitest(data, alpha=0.05):
    """
    Realiza la prueba de Friedman y, si es significativa, aplica pruebas post hoc.
    Args:
        data (list of lists or np.array): Datos organizados en varias muestras.
        alpha (float): Nivel de significancia.
    Returns:
        dict: Resultados de la prueba de Friedman y pruebas post hoc si es necesario.
    """
    reject = friedman(data)['pvalue'] <= alpha
    test = {'stat-test': 'Friedman', 'reject': reject}
    return test

def dualtest(data1, data2, alpha=0.05):
    """
    Realiza la prueba de Wilcoxon para comparar dos muestras.
    Args:
        data1 (list or np.array): Primer conjunto de datos.
        data2 (list or np.array): Segundo conjunto de datos.
        alpha (float): Nivel de significancia.
    Returns:
        dict: Resultados de la prueba de Wilcoxon con la decisión sobre H0.
    """
    reject = wilcoxon(data1, data2)['pvalue'] <= alpha
    return {'stat-test': 'Wilcoxon', 'reject': reject}

def statistical_test(data: np.array, alpha):
    """
    Determina y ejecuta la prueba estadística adecuada según el número de muestras.
    Args:
        data (np.array): Datos organizados en muestras.
        alpha (float): Nivel de significancia.
    Returns:
        dict: Resultados de la prueba estadística adecuada.
    """
    return dualtest(*data, alpha) if len(data) == 2 else multitest(data, alpha)

def critical_distance(data, alpha=0.05):
    """
    Calcula la distancia crítica para la prueba de Nemenyi.
    Args:
        data (list of lists or np.array): Datos organizados en varias muestras.
        alpha (float): Nivel de significancia.
    Returns:
        float: Distancia crítica calculada.
    """
    k = len(data)
    N = len(data[0])  # Número de muestras por grupo
    q_alpha = hyp.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return q_alpha * np.sqrt(k * (k + 1) / (6 * N))

def nemenyi(data):
    """
    Realiza la prueba post hoc de Nemenyi tras la prueba de Friedman.
    Args:
        data (list of lists or np.array): Datos organizados en varias muestras.
    Returns:
        dict: Matriz de p-valores resultante de la prueba de Nemenyi.
    """
    ranks = mean(hyp.rankdata(data, axis=0), axis=1)
    return {'post-hoc': 'Nemenyi', 'ranks': ranks, 'critical-distance': critical_distance(data)}

def bonferroni(data: np.array, control: int, alpha=0.05):
    """
    Realiza una comparación One vs All utilizando la corrección de Bonferroni.
    Args:
        data (list of lists or np.array): Datos organizados en varias muestras.
        control (int): Índice del grupo de control.
    Returns:
        dict: Resultados de la prueba de Bonferroni.
    """

    algorithms, cases = data.shape
    ranks = np.mean(hyp.rankdata(data, axis=0),axis=1)
    z_friedman = lambda i, j: np.abs(ranks[i] - ranks[j]) / np.sqrt(algorithms * (algorithms + 1) / (6 * cases))
    pvalues = 2 * hyp.norm.sf(np.array([z_friedman(control, i) for i in range(algorithms) if i != control]))
    ajusted_pvalues = np.minimum(pvalues * (algorithms - 1), 1)

    # reject = ajusted_pvalues <= alpha
    return {'post-hoc': 'Bonferroni', 'control': control, 'ranks': ranks, 'adjusted-pvalues': ajusted_pvalues}  