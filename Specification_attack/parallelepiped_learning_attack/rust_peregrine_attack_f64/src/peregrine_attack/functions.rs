use libm::{cos, sin};
use nalgebra::DMatrix;
use num_complex::Complex;
use rand::distributions::{Distribution, Uniform};
use rayon::prelude::*;
use std::f64::consts::{E, PI};

// mean value
fn mean_poly(a: &Vec<Vec<f64>>) -> Vec<f64> {
    let m = a.len();
    let n = a[0].len() / 2;
    let mut mean: Vec<f64> = vec![0.0; n * 2];
    let mut mean_val: f64;
    let i_mat: Vec<f64> = vec![1.0; n];
    for i in 0..n * 2 {
        mean_val = 0.0;
        for j in 0..m {
            mean_val += a[j][i];
        }
        mean[i] = mean_val / m as f64;
    }
    let mean_mid1 = mul(&mean[0..n].to_vec(), &i_mat);
    let mean_mid2 = mul(&mean[n..n * 2].to_vec(), &i_mat);
    for i in 0..n * 2 {
        if i < n {
            mean[i] = mean_mid1[i] / n as f64;
        } else {
            mean[i] = mean_mid2[i - n] / n as f64;
        }
    }
    mean
}

// covariance
pub fn cov_poly(a: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mean = mean_poly(a);
    assert!(mean.len() == a[0].len());
    let n = a[0].len() / 2;
    let m = a.len();
    let mut cov: Vec<Vec<f64>> = vec![vec![0.0; n]; 4];
    let mut cov_paral: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; n]; 3]; m];

    cov_paral.par_iter_mut().enumerate().for_each(|(i, x)| {
        let mut a_mean = vec![0.0; n * 2];
        for j in 0..n * 2 {
            a_mean[j] = a[i][j] - mean[j];
        }
        let tmp00 = mul(&a_mean[0..n].to_vec(), &adj(&a_mean[0..n].to_vec()));
        let tmp01 = mul(&a_mean[n..n * 2].to_vec(), &adj(&a_mean[0..n].to_vec()));
        let tmp11 = mul(&a_mean[n..n * 2].to_vec(), &adj(&a_mean[n..n * 2].to_vec()));
        for j in 0..n {
            x[0][j] = tmp00[j];
            x[1][j] = tmp01[j];
            x[2][j] = tmp11[j];
        }
    });
    for i in 0..n {
        let tmp: Vec<f64> = cov_paral.iter().map(|x| x[0][i]).collect();
        cov[0][i] = tmp.par_iter().sum();
        let tmp: Vec<f64> = cov_paral.iter().map(|x| x[1][i]).collect();
        cov[1][i] = tmp.par_iter().sum();
        let tmp: Vec<f64> = cov_paral.iter().map(|x| x[2][i]).collect();
        cov[3][i] = tmp.par_iter().sum();
    }

    for i in 0..n {
        cov[0][i] = cov[0][i] / (n * m) as f64;
        cov[1][i] = cov[1][i] / (n * m) as f64;
        cov[3][i] = cov[3][i] / (n * m) as f64;
    }
    cov[2] = adj(&cov[1]);
    cov
}

// the inverse of gram matrix
pub fn g_inv(cov: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let m = cov.len();
    let n = cov[0].len();
    assert!(m == 4);
    let tmp1 = mul(&cov[0], &cov[3]);
    let tmp2 = mul(&cov[1], &cov[2]);
    let tmp = add(&tmp1, &neg(&tmp2));
    let mut i_mat: Vec<f64> = vec![0.0; n];
    i_mat[0] = 1.0;
    let tmp_inv = div(&i_mat, &tmp);

    let mut ginv: Vec<Vec<f64>> = vec![vec![0.0; n]; 4];

    ginv[0] = mul(&tmp_inv, &cov[3]);
    ginv[1] = mul(&tmp_inv, &neg(&cov[1]));
    ginv[2] = mul(&tmp_inv, &neg(&cov[2]));
    ginv[3] = mul(&tmp_inv, &cov[0]);
    ginv
}

// ldl decomposition
pub fn ldl(g: &Vec<Vec<f64>>) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = g[0].len();
    assert!(g.len() == 4);
    let mut l: Vec<Vec<f64>> = vec![vec![0.0; n]; 4];
    let mut d: Vec<Vec<f64>> = vec![vec![0.0; n]; 4];

    let mut zero = vec![0.0; n];
    let mut one = vec![0.0; n];
    let mut d00 = vec![0.0; n];
    let mut d10 = vec![0.0; n];

    one[0] = 1.0;
    for i in 0..n {
        zero[i] = 0.0;
        if i >= 1 {
            one[i] = 0.0;
        }
        d00[i] = g[0][i];
    }

    let l10 = div(&g[2], &g[0]);
    let d11 = add(&g[3], &neg(&mul(&mul(&l10, &adj(&l10)), &g[0])));

    for i in 0..n {
        l[0][i] = one[i];
        d[0][i] = d00[i];

        l[1][i] = zero[i];
        d[1][i] = zero[i];

        l[2][i] = l10[i];
        d[2][i] = zero[i];

        l[3][i] = one[i];
        d[3][i] = d11[i];
    }
    (l, d)
}

fn iteration_dbs(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    assert!(a.len() == b.len());
    let n = a.len();
    let mut res: Vec<f64> = vec![0.0; n];
    let mut id: Vec<f64> = vec![0.0; n];
    id[0] = 1.0;
    let b_inv = div(&id, &b);
    for i in 0..n {
        res[i] = (a[i] + b_inv[i]) / 2.0;
    }
    res
}

fn abs_sub_p(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    assert!(a.len() == b.len());
    let mut c = vec![0.0; a.len()];
    for i in 0..a.len() {
        c[i] = (a[i] - b[i]).abs();
    }
    c
}

// Denman-Beavers iteration
fn denman_beavers_sqrt(a: &Vec<f64>) -> Vec<f64> {
    let n = a.len();
    let mut err: f64 = 1.0;
    let err_t: f64 = 1.5e-8;
    let mut y_1: Vec<f64>;
    let mut y: Vec<f64> = a.clone();

    let mut z: Vec<f64> = vec![0.0; n];
    z[0] = 1.0;
    while err > err_t {
        y_1 = iteration_dbs(&y, &z);
        z = iteration_dbs(&z.clone(), &y);
        let err_p: Vec<f64> = abs_sub_p(&y_1, &y);

        err = 0.0;
        if err_p
            .iter()
            .fold(std::f64::NEG_INFINITY, |max, &x| max.max(x))
            > err
        {
            err = err_p
                .iter()
                .fold(std::f64::NEG_INFINITY, |max, &x| max.max(x));
        }
        y = y_1.clone();
    }
    y
}

// compute L
pub fn l_poly(g_inv: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    assert!(g_inv.len() == 4);
    let (l, d): (Vec<Vec<f64>>, Vec<Vec<f64>>) = ldl(&g_inv);
    let n = g_inv[0].len();
    let d1: Vec<f64>;
    let d2: Vec<f64>;
    let d1_gl: Vec<f64>;
    let mut lpoly: Vec<Vec<f64>> = vec![vec![0.0; n]; 4];

    d1 = denman_beavers_sqrt(&d[0]);
    d2 = denman_beavers_sqrt(&d[3]);
    d1_gl = mul(&d1, &l[2]);
    lpoly[0] = d1;
    lpoly[2] = d1_gl;
    lpoly[3] = d2;
    lpoly
}

// compute the fourth moment
fn moment(sig_adj: &Vec<Vec<f64>>, l_adj: &Vec<Vec<f64>>, w: &Vec<f64>) -> f64 {
    let m = sig_adj.len();
    let n = sig_adj[0].len() / 2;
    let mut mean: f64 = 0.0;
    let mut w_new: Vec<f64> = vec![0.0; n * 2];
    let tmp0 = mul(&w[0..n].to_vec(), &l_adj[0]);
    let tmp1 = mul(&w[0..n].to_vec(), &l_adj[2]);
    let tmp2 = mul(&w[n..n * 2].to_vec(), &l_adj[3]);
    let tmp3 = add(&tmp1, &tmp2);
    // paral
    let mut mean_paral: Vec<f64> = vec![0.0; m];
    w_new.par_iter_mut().enumerate().for_each(|(i, x)| {
        if i < n {
            *x = tmp0[i];
        } else {
            *x = tmp3[i - n];
        }
    });
    mean_paral.par_iter_mut().enumerate().for_each(|(i, m)| {
        let sw0 = mul(&sig_adj[i][0..n].to_vec(), &w_new[0..n].to_vec());
        let sw1 = mul(&sig_adj[i][n..n * 2].to_vec(), &w_new[n..n * 2].to_vec());
        let mut tmp_mean: f64 = 0.0;
        for j in 0..n {
            let sw = sw0[j] + sw1[j];
            tmp_mean += sw.powi(4);
        }
        *m = tmp_mean;
    });
    mean = mean_paral.par_iter().sum();
    // paral
    mean = mean / (m * n) as f64;
    mean
}

// compute the gradient of fourth moment
fn gradient(
    sig: &Vec<Vec<f64>>,
    sig_adj: &Vec<Vec<f64>>,
    l_poly: &Vec<Vec<f64>>,
    l_adj: &Vec<Vec<f64>>,
    w: &Vec<f64>,
) -> Vec<f64> {
    let m: usize = sig_adj.len();
    let n: usize = sig_adj[0].len() / 2;
    let mut grad: Vec<f64> = vec![0.0; n * 2];
    let mut w_new: Vec<f64> = vec![0.0; n * 2];
    // let mut mid: Vec<f64> = vec![0.0; n];
    let mut sig_mid: Vec<f64> = vec![0.0; n * 2];
    let tmp0 = mul(&w[0..n].to_vec(), &l_adj[0]);
    let tmp1 = mul(&w[0..n].to_vec(), &l_adj[2]);
    let tmp2 = mul(&w[n..n * 2].to_vec(), &l_adj[3]);
    let tmp3 = add(&tmp1, &tmp2);
    for i in 0..n {
        w_new[i] = tmp0[i];
        w_new[i + n] = tmp3[i];
    }
    let mut sig_mid_paral: Vec<Vec<f64>> = vec![vec![0.0; n * 2]; m];
    sig_mid_paral
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, sigmid)| {
            let sw0 = mul(&sig_adj[i][0..n].to_vec(), &w_new[0..n].to_vec());
            let sw1 = mul(&sig_adj[i][n..n * 2].to_vec(), &w_new[n..n * 2].to_vec());
            let mut mid: Vec<f64> = vec![0.0; n];
            for j in 0..n {
                let sw = sw0[j] + sw1[j];
                mid[j] = 4.0 * sw.powi(3);
            }
            let mid_sig1 = mul(&mid, &sig[i][0..n].to_vec());
            let mid_sig2 = mul(&mid, &sig[i][n..n * 2].to_vec());
            for j in 0..n {
                sigmid[j] = mid_sig1[j];
                sigmid[j + n] = mid_sig2[j];
            }
        });
    sig_mid.iter_mut().enumerate().for_each(|(i, x)| {
        let m: Vec<f64> = sig_mid_paral.iter().map(|z| z[i]).collect();
        *x = m.par_iter().sum();
    });
    sig_mid.par_iter_mut().for_each(|x| *x /= (m * n) as f64);
    let grad1 = add(
        &mul(&sig_mid[0..n].to_vec(), &l_poly[0]),
        &mul(&sig_mid[n..n * 2].to_vec(), &l_poly[2]),
    );
    let grad2 = add(
        &mul(&sig_mid[0..n].to_vec(), &l_poly[1]),
        &mul(&sig_mid[n..n * 2].to_vec(), &l_poly[3]),
    );
    for i in 0..n {
        grad[i] = grad1[i];
        grad[i + n] = grad2[i];
    }
    grad
}

// perform gradient descent to find w, in Algorithm 3
pub fn gradient_descent(
    sigs: &Vec<Vec<f64>>,
    sigs_adj: &Vec<Vec<f64>>,
    l_poly: &Vec<Vec<f64>>,
    average: &Vec<f64>,
    max_desc: usize,
) -> Vec<f64> {
    let n = sigs[0].len() / 2;
    let inner_inner: f64 = 0.0;
    let phimin: f64 = 0.005;
    let nu: f64 = 0.8;
    let mut phi0: f64 = 0.25;
    let mut phi: f64;
    let mut moment0: f64;
    let mut moment1: f64 = 0.0;
    let mut g: Vec<f64> = vec![0.0; n * 2];
    let mut h: Vec<f64> = vec![0.0; n * 2];
    let mut l_adj: Vec<Vec<f64>> = vec![vec![0.0; n]; 4];
    for i in 0..4 {
        l_adj[i] = adj(&l_poly[i]);
    }
    // 1. pick w0 at random from unit sphere
    let mut w: Vec<f64> = vec![0.0; n * 2];
    let mut w_new: Vec<f64> = vec![0.0; n * 2];

    let unif = Uniform::from(-1.0..1.0);
    let mut rng = rand::thread_rng();
    for i in 0..n * 2 {
        if average[i] == 0.0 {
            w[i] = unif.sample(&mut rng) as f64;
        } else {
            w[i] = average[i];
        }
    }
    normalize(&mut w);
    moment0 = -1.0 * moment(&sigs_adj, &l_adj, &w);
    println!("moment : {:?}", moment0);

    // 2. grad mom_4(w) from empirical distribution
    for n_loop in 0..max_desc {
        // 3. descent
        g = gradient(&sigs, &sigs_adj, &l_poly, &l_adj, &w);
        let gw_iprod: f64 = inner_product(&neg(&g), &w);
        for i in 0..n * 2 {
            h[i] = neg(&g)[i] - gw_iprod * w[i];
        }
        normalize(&mut h);

        phi = phi0;
        while phi >= phimin {
            for i in 0..n * 2 {
                w_new[i] = cos(phi) * w[i] + sin(phi) * h[i];
            }
            // 4. Normalize w_new
            normalize(&mut w_new);
            moment1 = -1.0 * moment(&sigs_adj, &l_adj, &w_new);
            if moment1 > moment0 + 0.5 * phi * inner_product(&h, &neg(&g)) {
                break;
            }
            phi *= nu;
        }

        // 5. Check and return
        w = w_new.clone();
        println!("{} :  {}", n_loop, moment1);
        if phi < phi0 {
            phi0 = phi / nu;
        }
        if phi < phimin {
            break;
        }
        moment0 = moment1;
    }
    w
}

fn normalize(a: &mut Vec<f64>) {
    let mut norm: f64;
    let n = a.len();
    let mut norm_paral: Vec<f64> = vec![0.0; n];
    norm_paral.par_iter_mut().enumerate().for_each(|(i, x)| {
        *x = a[i] * a[i];
    });
    norm = norm_paral.par_iter().sum();
    norm = norm.sqrt();
    a.par_iter_mut().for_each(|x| *x /= norm);
}

pub fn inverse_mtx(a: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a.len();
    assert!(a.len() == a[0].len());
    let flat_a: Vec<f64> = a.clone().into_iter().flatten().collect::<Vec<f64>>();
    let mut mtx_a = DMatrix::from_vec(n, n, flat_a);
    for i in 0..n {
        for j in 0..n {
            mtx_a[(i, j)] = a[i][j];
        }
    }
    let mtx_inv = mtx_a.try_inverse().unwrap();
    let mut inv: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = mtx_inv[(i, j)] as f64;
        }
    }
    inv
}

pub fn mtx_i64_to_f64(a: Vec<Vec<i64>>) -> Vec<Vec<f64>> {
    a.iter()
        .map(|row| row.iter().map(|&x| x as f64).collect())
        .collect()
}

fn add(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    assert!(a.len() == b.len());
    let n = a.len();
    let mut c: Vec<f64> = vec![0.0; n];
    c.par_iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = a[i] + b[i]);
    c
}

fn mul(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    assert!(a.len() == b.len());
    let a_fft = fft(a);
    let b_fft = fft(b);
    let n = a.len();
    let mut c_fft: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];
    for i in 0..n {
        c_fft[i] = a_fft[i] * b_fft[i];
    }
    let c = ifft(&c_fft);
    c
}
fn div(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    assert!(a.len() == b.len());
    let n = a.len();
    let a_fft = fft(a);
    let b_fft = fft(b);
    let mut c_fft: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];
    for i in 0..n {
        c_fft[i] = a_fft[i] / b_fft[i];
    }
    let c = ifft(&c_fft);
    c
}
pub fn neg(a: &Vec<f64>) -> Vec<f64> {
    let mut c = vec![0.0; a.len()];
    c.par_iter_mut().enumerate().for_each(|(i, x)| *x = -a[i]);
    c
}

fn inner_product(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    let prod: f64;
    assert!(a.len() == b.len());
    let n = a.len();
    let mut prod_paral: Vec<f64> = vec![0.0; n];
    prod_paral.par_iter_mut().enumerate().for_each(|(i, p)| {
        *p = a[i] * b[i];
    });
    prod = prod_paral.par_iter().sum();
    prod
}

fn fft(a: &Vec<f64>) -> Vec<Complex<f64>> {
    let n = a.len();
    let mut a_fft: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];
    let w0: Complex<f64> = Complex::new(E, 0.0).powc(Complex::new(0.0, PI / (n as f64)));
    fft_step(&mut a_fft, &a, n, n, w0);
    a_fft
}

fn fft_step(f_fft: &mut Vec<Complex<f64>>, f: &Vec<f64>, n: usize, n0: usize, w0: Complex<f64>) {
    if n == 1 {
        f_fft[0] = Complex::new(f[0], 0.0);
    } else {
        if n == 2 {
            f_fft[0] = Complex::new(f[0], f[1]);
            f_fft[1] = Complex::new(f[0], -f[1]);
        } else {
            assert!(n % 2 == 0);
            let mut f0: Vec<f64> = vec![0.0; n0 / 2];
            let mut f1: Vec<f64> = vec![0.0; n0 / 2];
            let mut f0_fft: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n0 / 2];
            let mut f1_fft: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n0 / 2];
            let w02: Complex<f64> = w0 * w0;
            let mut wk: Complex<f64> = w0;
            for i in 0..(n / 2) {
                f0[i] = f[2 * i];
                f1[i] = f[2 * i + 1];
            }
            fft_step(&mut f0_fft, &f0, n / 2, n0, w02);
            fft_step(&mut f1_fft, &f1, n / 2, n0, w02);
            for i in 0..n {
                f_fft[i] = f0_fft[i % (n / 2)] + wk * f1_fft[i % (n / 2)];
                wk *= w02;
            }
        }
    }
}

// adjoint
pub fn adj(f: &Vec<f64>) -> Vec<f64> {
    let c = ifft(&adj_fft(&fft(&f)));
    c
}

fn adj_fft(f_fft: &Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    let n = f_fft.len();
    let mut adj_f_fft: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];
    for i in 0..n {
        adj_f_fft[i] = f_fft[i].conj();
    }
    adj_f_fft
}

fn ifft(f_fft: &Vec<Complex<f64>>) -> Vec<f64> {
    let n = f_fft.len();
    let mut fc: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];
    let mut f: Vec<f64> = vec![0.0; n];
    let w0: Complex<f64> = Complex::new(E, 0.0).powc(Complex::new(0.0, -PI / (n as f64)));

    ifft_step(&mut fc, &f_fft, n, n, w0);

    for i in 0..n {
        f[i] = fc[i].re;
    }
    f
}

fn ifft_step(
    f: &mut Vec<Complex<f64>>,
    f_fft: &Vec<Complex<f64>>,
    n: usize,
    n0: usize,
    w0: Complex<f64>,
) {
    if n != 2 {
        assert!(n % 2 == 0);
        let mut f0: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n0 / 2];
        let mut f1: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n0 / 2];
        let mut f0_fft: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n0 / 2];
        let mut f1_fft: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n0 / 2];
        let w02: Complex<f64> = w0 * w0;
        let mut wk: Complex<f64> = w0;
        for i in 0..n / 2 {
            f0_fft[i] = (f_fft[i] + f_fft[i + (n / 2)]) * Complex::new(0.5, 0.0);
            f1_fft[i] = wk * (f_fft[i] - f_fft[i + (n / 2)]) * Complex::new(0.5, 0.0);
            wk *= w02;
        }
        ifft_step(&mut f0, &f0_fft, n / 2, n0, w02);
        ifft_step(&mut f1, &f1_fft, n / 2, n0, w02);

        for i in 0..(n / 2) {
            f[2 * i] = f0[i];
            f[2 * i + 1] = f1[i];
        }
    } else {
        f[0] = (f_fft[0] + f_fft[1]) * Complex::new(0.5, 0.0);
        f[1] = (f_fft[0] - f_fft[1]) * Complex::new(0.0, -0.5);
    }
}

fn poly_to_anti_circulant_mtx(a: &Vec<f64>) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut mtx: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    mtx[0] = a.clone();
    for i in 1..n {
        for j in 0..n {
            if j >= 1 {
                mtx[i][j] = mtx[i - 1][j - 1];
            } else {
                mtx[i][j] = -mtx[i - 1][n - 1];
            }
        }
    }
    mtx
}

pub fn poly_to_mtx(a: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    assert!(a.len() == 4);
    let n = a[0].len();
    let mut mtx: Vec<Vec<f64>> = vec![vec![0.0; n * 2]; n * 2];
    let l0_mtx = poly_to_anti_circulant_mtx(&a[0]);
    let l1_mtx = poly_to_anti_circulant_mtx(&a[1]);
    let l2_mtx = poly_to_anti_circulant_mtx(&a[2]);
    let l3_mtx = poly_to_anti_circulant_mtx(&a[3]);
    for i in 0..n * 2 {
        for j in 0..n * 2 {
            if i < n && j < n {
                mtx[i][j] = l0_mtx[i][j]
            } else if i < n && j >= n {
                mtx[i][j] = l1_mtx[i][j - n];
            } else if i >= n && j < n {
                mtx[i][j] = l2_mtx[i - n][j];
            } else if i >= n && j >= n {
                mtx[i][j] = l3_mtx[i - n][j - n];
            }
        }
    }
    mtx
}
