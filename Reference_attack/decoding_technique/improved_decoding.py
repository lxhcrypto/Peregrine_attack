import numpy as np
from utils import *
from scipy.stats import rankdata
from key_process_reference import *

nb_keys = 10    # The number of keys
nb_points = 5   # The number of starting points used to perform gradient descent
nb_sigs = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]   # The sample size S
nb_sigs = [nb_sigs[i] * (10**4) for i in range(len(nb_sigs))]

# The l2-norm of error of e, where e = b - round(b')
norm_array = np.fromfile('../data/norm_array_reference.ha', dtype=np.float64)
norm_array = norm_array.reshape((nb_keys, len(nb_sigs), nb_points))

# The approximate keys are recovered by parallelepiped_learning_attack
key_recovered = np.fromfile('../data/key_recovered_array_reference.ha', dtype=np.float64)
key_recovered = key_recovered.reshape((nb_keys, len(nb_sigs), nb_points, m))

# The public key of Peregrine-512
pk_array = np.fromfile('../data/pk_array_reference.ha', dtype=np.int64)
pk_array = pk_array.reshape((nb_keys, n))

# The inverse matrix of L is used to compute Gran matrix G
Linv_array = np.fromfile('../data/Linv_array_reference.ha', dtype=np.float64)
Linv_array = Linv_array.reshape((nb_keys, len(nb_sigs), 4, n))

success_array = np.zeros((nb_keys, len(nb_sigs)), dtype=np.int64)

for nb_key in range(nb_keys):
    print("\n\n\n\n")
    print("keys: ", nb_key)
    # generate the anti-circulant matrix of sk and pk
    key_array = key_512_array[nb_key]
    f = key_array[0]
    g = key_array[1]
    F = key_array[2]
    G = key_array[3]
    h = pk_array[nb_key]
    ZZ_q = IntegerModRing(12289)
    A, B = key_matrix(f, g, F, G, h)
    H = anti_cir(h)
    Ht_q = matrix(ZZ_q, H.transpose())
    for idx, item in enumerate(nb_sigs):
        s = item
        counter_success = 0
        Linv = Linv_array[nb_key][idx]
        Linv_0 = np.array(anti_cir(Linv[0]))
        Linv_1 = np.array(anti_cir(Linv[1]))
        Linv_2 = np.array(anti_cir(Linv[2]))
        Linv_3 = np.array(anti_cir(Linv[3]))
        Linv_01 = np.hstack((Linv_0, Linv_1))
        Linv_23 = np.hstack((Linv_2, Linv_3))
        Linv = np.vstack((Linv_01, Linv_23))
        Linv = matrix(Linv)
        for nb_point in range(nb_points):
            print("sample size: ", item, "points: ", nb_point)
            b = key_recovered[nb_key][idx][nb_point]
            g_prime = vector(ZZ_q, [round(b[i]) for i in range(N)])
            f_prime = vector(ZZ_q, [round(-b[i]) for i in range(N, 2 * N)])
            rounded_b = [round(b[i]) for i in range(2 * N)]
            # ========================================
            # Estimate std for each coefficient
            print("\nExecuting the decoding technique...\n")
            # Gram matrix G
            G = Linv.transpose() * Linv
            # 0.14720853262850522 is obtained by curve fitting and
            # sigma is the std of error of w which is outputted by gradient descent
            sigma = 0.14720853262850522 / sqrt(s)
            C = sigma ** 2 * G
            sigma_i = [(sqrt(C[i,i])) for i in range(N*2)]
            # x = b' - rounded b'
            x = [(b[i] - round(b[i])) for i in range(N*2)]
            # list of probability of being correctly rounded
            p = [(prob(sigma_i[i], x[i])) for i in range(N*2)]
            # rank the probability
            rank = rankdata(p)
            rank = [int(rank[i])-1 for i in range(len(rank))]
            # separate the top N coefficients and other
            g0_index = []
            ge_index = []
            f0_index = []
            fe_index = []
            for i in range(N):
                if rank[i] >= N:
                    g0_index.append(i)
                else :
                    ge_index.append(i)
            for i in range(N,2*N):
                if rank[i] >= N:
                    f0_index.append(i-N)
                else:
                    fe_index.append(i-N)
            print("g0:",len(g0_index),"  ge:",len(ge_index),"  f0:",len(f0_index),"  fe",len(fe_index))
            # If the l1-norm is less than 7, the exhaustive search is effective
            if norm_array[nb_key][idx][nb_point] < np.sqrt(7):
                counter_success += 1
                print("keys: ", nb_key, "sample size: ", item, "points: ", nb_point)
                print("Norm: ", norm_array[nb_key][idx][nb_point], "  Exhaustive search can be used to do key recovery attack.")
                continue
            if len(g0_index) + len(f0_index) < len(ge_index) + len(fe_index):
                print("The rearrange matrix is not a square matrix. ")
                continue
            # ===========================
            # Prest trick
            # ===========================
            tmp = matrix(ZZ,len(fe_index),N)
            for i in range(len(fe_index)):
                for j in range(N):
                    tmp[i,j] = H[fe_index[i],j]
            tmp = tmp.transpose()
            M = []
            for i in range(len(g0_index)):
                M.append(tmp[g0_index[i]])
            M = matrix(ZZ_q,M[0:len(fe_index)][0:len(fe_index)])
            v = vector(ZZ_q,g_prime - Ht_q * f_prime)
            tmp = []
            for i in range(len(g0_index)):
                tmp.append(int(v[g0_index[i]]))
            v = vector(ZZ_q,tmp[0:len(fe_index)])
            tmp = M.inverse() * v
            e_f = zero_vector(ZZ_q,N)
            for i in range(len(fe_index)):
                e_f[fe_index[i]] = int(tmp[i])
            recovered_f = f_prime + e_f
            e_g = (g_prime - Ht_q * f_prime)  - Ht_q * e_f
            recovered_g = - g_prime + e_g
            # Check
            B_q = matrix(ZZ_q,B)
            error = []
            for i in range(2*N):
                incorrect_f = 0
                incorrect_g = 0
                for j in range(N):
                    if recovered_g[j] != B_q[i,j]:
                        incorrect_g+=1
                    if recovered_f[j] != B_q[i,j+N]:
                        incorrect_f+=1
                error.append(incorrect_g+incorrect_f)
            B_q = matrix(ZZ_q,-B)
            for i in range(2*N):
                incorrect_f = 0
                incorrect_g = 0
                for j in range(N):
                    if recovered_g[j] != B_q[i,j]:
                        incorrect_g+=1
                    if recovered_f[j] != B_q[i,j+N]:
                        incorrect_f+=1
                error.append(incorrect_g+incorrect_f)
            print("============Result============")
            if min(error) != 0:
                print("Fail")
            else:
                counter_success += 1
                print("Success")
            print("===============================")

        success_array[nb_key][idx] = counter_success

for nb_key in range(nb_keys):
    print(success_array[nb_key].tolist())


