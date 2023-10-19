# Cryptanalysis of the Peregrine Signature Scheme

[Peregrine](https://kpqc.or.kr/) signature scheme, as the fastest variant of [Falcon](https://falcon-sign.info/), is one of the signature candidates in the 1-st round of the ongoing Korean PQC competition. In order to obtain better performance, [Peregrine](https://kpqc.or.kr/) removes the lattice Gaussian sampler in the Falcon and selects centered binomial distribution as a substitute. Without provable guarantee, [Peregrine](https://kpqc.or.kr/) is potential to demonstrate key recovery attack via statistical method. We observe that the support of Peregrine signatures is a hidden transformation of some public distribution and still leak information about the signing key. In our attack, we are successful to break the reference implementation and the specification version of [Peregrine](https://kpqc.or.kr/)-512. 

This attack can be divided into two steps:
* Based on the parallelepiped-learning technique of Nguyen and Regev [NR06](https://iacr.org/archive/eurocrypt2006/40040273/40040273.pdf), we can extend this technique to [Peregrine](https://kpqc.or.kr/) and obtain the approximations of the signing key. We implement the learning attack in Rust using f64 precision.

* By adapting the decoding technique of [Pre23](https://eprint.iacr.org/2023/157), we can mount key recovery attack against [Peregrine](https://kpqc.or.kr/) scheme with small number of signatures. And the decoding technique part is implemented in Python.

## Parallelepiped-learning attack
To generate approximations of the signing key, we launch parallelepiped-learning attack to leak secret information from the signatures of [Peregrine](https://kpqc.or.kr/). We first set some important parameters in `/Reference_attack/parallelepiped_learning_attack/rust_peregrine_attack_f64/src/main.rs` or `/Specification_attack/parallelepiped_learning_attack/rust_peregrine_attack_f64/src/main.rs`:

* n_of_exp: the number of starting points for the gradient descent.
* log_dim: degree of the underlying ring
* v_nsigs * 10 ^{log_nsigs}: the number of signatures
* n_of_threads: the number of threads using to parallel

Then, to perform learning attack for the reference version:

```
$ cd /Reference_attack/parallelepiped_learning_attack/rust_peregrine_attack_f64/
$ cargo run --release > your_file_name.txt
```
and for the specification version:

```
$ cd /Specification_attack/parallelepiped_learning_attack/rust_peregrine_attack_f64/
$ cargo run --release > your_file_name.txt
```
## Decoding technique

Next, we employ decoding technique to mount key recovery attack. Since we prestore some binary files in `data` folder, we can directly read them and demonstrate the experiments of decoding.

For the reference version:
```
$ cd /Reference_attack/decoding_technique
$ python3 improved_decoding.py > your_file_name.txt
```

For the specification version:
```
$ cd /Specification_attack/decoding_technique
$ python3 improved_decoding.py > your_file_name.txt
```