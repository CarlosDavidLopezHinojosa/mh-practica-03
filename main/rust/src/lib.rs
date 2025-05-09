use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::PyReadonlyArray1;
use rayon::prelude::*;

/// Z‑score “precomputado” del patrón.
fn compute_pattern_z(pat: &[f64]) -> Option<Vec<f64>> {
    let n = pat.len() as f64;
    // media y suma de cuadrados
    let (sum, sum_sq) = pat.iter().fold((0.0, 0.0), |(s, sq), &v| (s+v, sq+v*v));
    let mean = sum / n;
    let var = sum_sq / n - mean*mean;
    if var <= 1e-12 { return None }
    let std = var.sqrt();
    Some(pat.iter().map(|&v| (v-mean)/std).collect())
}

/// Calcula en O(L) la correlación de un slice `win` con el patrón z `p_z`,
/// sin hacer allocations intermedias.
fn corr_with_pattern_z(win: &[f64], p_z: &[f64]) -> f64 {
    let n = win.len() as f64;
    // En un solo fold sacamos sum(win), sum(win²) y sum(win * p_z)
    let (sum_w, sum_w2, sum_wz) = win.iter()
        .zip(p_z.iter())
        .fold((0.0, 0.0, 0.0), |(s, sq, swz), (&w, &z)| {
            (s + w, sq + w*w, swz + w*z)
        });
    let mean_w = sum_w / n;
    // varianza poblacional
    let var_w = sum_w2 / n - mean_w*mean_w;
    let std_w = var_w.sqrt().max(1e-12);
    // dot(p_z, zscore(win))  = sum_wz / std_w
    // corr = (sum(z_p * z_w) / n)  = (sum_wz / std_w) / n
    (sum_wz / std_w) / n
}

#[pyfunction]
fn fastfitness(
    _py: Python<'_>,
    S: PyReadonlyArray1<'_, f64>,
    pattern: PyReadonlyArray1<'_, f64>,
    threshold: f64,
) -> PyResult<f64> {
    let s = S.as_slice().map_err(|_| PyValueError::new_err("Invalid S"))?;
    let pat = pattern.as_slice().map_err(|_| PyValueError::new_err("Invalid pattern"))?;

    // Decodifica: primer elemento = L, resto = coeficientes
    if pat.is_empty() {
        return Err(PyValueError::new_err("Pattern vacío"));
    }
    let L = pat[0] as usize;
    if pat.len() < L + 1 {
        return Err(PyValueError::new_err("Pattern length mismatch"));
    }
    let coeffs = &pat[1..=L];

    // Patrón z‑score
    let p_z = match compute_pattern_z(coeffs) {
        Some(v) => v,
        None => {
            // Si std(pat)==0 devolvemos L (ocurrencias=0 + L)
            return Ok(L as f64);
        }
    };

    // Número de ventanas
    if s.len() < L {
        return Ok(L as f64);
    }
    let windows = s.len() - L + 1;

    // Paralelizamos sobre índices si la serie es muy grande
    let occs: usize = (0..windows)
        .into_par_iter()
        .map(|i| {
            let win = &s[i..i + L];
            // Si var(win)==0, compute_pattern_z nos daría None; saltamos
            let dot = {
                // Calcular sum((w-mean)/std * p_z) 
                // pero usando corr_with_pattern_z
                corr_with_pattern_z(win, &p_z)
            };
            if dot >= threshold { 1 } else { 0 }
        })
        .sum();

    // fitness = ocurrencias + L
    Ok(occs as f64 + L as f64)
}

/// This module is implemented in Rust.
#[pymodule]
fn fastevaluate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fastfitness, m)?);
    // m.add_function(wrap_pyfunction!(fastcorrcoeff, m)?);
    Ok(())
}
