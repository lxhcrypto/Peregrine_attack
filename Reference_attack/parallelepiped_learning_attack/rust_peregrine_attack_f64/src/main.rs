mod peregrine_attack;
mod peregrine_scheme;
use crate::peregrine_attack::functions::{
    adj, cov_poly, g_inv, gradient_descent, inverse_mtx, l_poly, mtx_i64_to_f64, poly_to_mtx,
};
use peregrine_scheme::sig_gen::key_sig_gen;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use rayon::ThreadPoolBuilder;
use std::{time::Instant, usize};

fn main() {
    // ====================================================
    // Parameter Sets
    let n_of_exp: usize = 5; // the number of trials for the gradient descent.
    let log_dim: usize = 9; // ^2 = the number of dimension(9:n=512,10:n=1024)
    let log_nsigs: usize = 4; // ^10 = the number of signatures
    let n_of_threads: usize = rayon::current_num_threads() - 10;
    let v_nsigs: usize = 1;

    // ====================================================
    // paral

    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(n_of_threads)
        .build()
        .unwrap();
    thread_pool.install(|| {
    //
    // ====================================================
    let n: usize = 2_i64.pow(log_dim as u32) as usize;
    let n_of_sigs: usize = 10_i64.pow(log_nsigs as u32) as usize * v_nsigs;
    println!("=======================");
    println!(
        "The number of dimensions:  {},\nThe number of signatures:  {}\n The number of threads:  {}",
        n, n_of_sigs,n_of_threads
    );
    println!("=======================\nGenerating keys and signatures...");
    let start = Instant::now();
    let (key, sigs) = key_sig_gen(log_dim, n_of_sigs);
    let calc_time = start.elapsed().as_secs();
    println!("\n\nTime for signature gen:  {:?} seconds\n\n ",calc_time);
    // Print signatures
    // for sig in &sigs {
    //     println!("{:?} ", sig);
    // }
    println!("{}", sigs.len());
    let sigs = mtx_i64_to_f64(sigs);
    println!("Computing the covariance matrix...");
    println!("\nCOV");
   let start = Instant::now();
 
    let mut cov = cov_poly(&sigs);
    // Print covariance mtx

    for row in &cov {
        print!("{:?}\n", row);
    }
    for i in 0..4 {
        for j in 0..n {
            cov[i][j] = cov[i][j]; // * (12.0 / 79.0);
        }
    }

    println!("\n\nGinv");
    let ginv = g_inv    (&cov);
    // Print G inverse
    for row in &ginv {
        print!("{:?}\n", row);
    }

    println!("\n\nLpoly");
    let mut lpoly = l_poly(&ginv);
    // Print L
    for row in &lpoly {
        println!("{:?}", row);
    }

    println!("\n\nLmat");
    let lmat = poly_to_mtx(&lpoly);
    // Print L (mtx)
    for row in &lmat {
        println!("{:?}", row)
    }

    println!("\n\nLinv");
    let linv: Vec<Vec<f64>> = inverse_mtx(&lmat);
    // Print L -1
    for row in &linv {
        println!("{:?}", row);
    }

    // sig adj
    let mut sig_adj: Vec<Vec<f64>> = vec![vec![0.0; n * 2]; sigs.len()];
    // Parallel
    sig_adj.par_iter_mut().enumerate().for_each(|(i, sigadj)| {
        let adj1 = adj(&sigs[i][0..n].to_vec());
        let adj2 = adj(&sigs[i][n..n * 2].to_vec());
        for j in 0..n {
            sigadj[j] = adj1[j];
            sigadj[j + n] = adj2[j];
        }
    });

    let average: Vec<f64> = vec![0.0; n * 2];
    let calc_time = start.elapsed().as_secs();

    println!("\n\nTime for computing Cov,etc.:  {:?} seconds\n\n ",calc_time);

    println!("Gradient Descent...");
    // GRADIENT DESCENT
    //
    for exp_num in 0..n_of_exp {
        let start = Instant::now();
        let w = gradient_descent(&sigs, &sig_adj, &lpoly, &average, 500);
        let calc_time = start.elapsed().as_secs();
        println!("w:  {:?}",w);
        // Check the answer
        //
        // Extend poly mtx to inverse circular matrix
        let mut b_mtx: Vec<Vec<i64>> = vec![vec![0; n * 2]; n * 2];
        let g: Vec<i64> = key[1].clone();
        let f: Vec<i64> = key[0].clone().into_iter().map(|x| -1 * x).collect();
        let gg: Vec<i64> = key[3].clone();
        let ff: Vec<i64> = key[2].clone().into_iter().map(|x| -1 * x).collect();
        b_mtx.par_iter_mut().enumerate().for_each(|(i, x)| {
            if i < n {
                for j in i..n {
                    x[j] = g[j - i];
                    x[j + n] = f[j - i];
                }
                for j in 0..i {
                    let index = (j as i64 - i as i64 + n as i64) as usize;
                    x[j] = -1 * g[index];
                    x[j + n] = -1 * f[index];
                }
            } else {
                let i_n = i - n;
                for j in i_n..n {
                    x[j] = gg[j - i_n];
                    x[j + n] = ff[j - i_n];
                }
                for j in 0..i_n {
                    let index = (j as i64 - i as i64 + 2 * n as i64) as usize;
                    x[j] = -1 * gg[index];
                    x[j + n] = -1 * ff[index];
                }
            }
        });

        let mut recovered_vector: Vec<f64> = vec![0.0; 2 * n];
        for i in 0..2 * n {
            for j in 0..2 * n {
                recovered_vector[i] += w[j] * linv[j][i];
            }
        }
        
        let tmp = (12.0 as f64).sqrt();
        // let tmp = ((12.0 / 19.0) as f64).sqrt();
        for i in 0..2 * n {
            recovered_vector[i] = recovered_vector[i] * tmp;
        }
        println!("recovered vector:");
        for elem in &recovered_vector {
            print!("{},  ", elem)
        }
        let mut min_norm: f64 = 1000.0;
        println!("\nkey:\n");
        for row in b_mtx {
            let mut error_vec1: Vec<f64> = vec![0.0; n * 2];
            let mut error_vec2: Vec<f64> = vec![0.0; n * 2];
            for i in 0..2 * n {
                error_vec1[i] = row[i] as f64 - recovered_vector[i].round();
                error_vec2[i] = row[i] as f64 + recovered_vector[i].round();
            }
            let mut l1norm_1: f64 = 0.0;
            let mut l1norm_2: f64 = 0.0;
            for i in 0..2 * n {
                l1norm_1 += error_vec1[i].powi(2);
                l1norm_2 += error_vec2[i].powi(2);
            }
            l1norm_1 = l1norm_1.sqrt();
            l1norm_2 = l1norm_2.sqrt();
            if min_norm > l1norm_1 {
                min_norm = l1norm_1;
            }
            if min_norm > l1norm_2 {
                min_norm = l1norm_2;
            }

            println!("{:?} -> {},  {}", row, l1norm_1, l1norm_2);
        }
        
        println!("\nRecovered vector: \n\n{:?}",recovered_vector);
        println!("\n\nw:\n\n{:?}",w);

        println!("\n\n\n MIN NORM: {}\n\n\n", min_norm);
        println!(
            "=======================\nExp number:{} / {}\ndim:{}\nnofsigs:{}\ncalc time: {} seconds\n=======================",
            exp_num+1, n_of_exp, n,n_of_sigs,calc_time
        );
    }
});
}
