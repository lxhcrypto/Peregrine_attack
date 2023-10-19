from sage.all import *

N = 512     # degree of the underlying ring
q = 12289   # modulus


# rho/rho
def rho_x_z(sigma,x):
    sum = 0.0
    for k in range(-50,50):
        sum += exp(-(x+k)**2 / (2 * sigma ** 2))
    return sum


def rho(sigma,x):
    return exp(-(x)**2 / (2 * sigma ** 2))


# compute probability
def prob(sigma,x):
    return rho(sigma,x) / rho_x_z(sigma,x)


# generate the anti-circulant matrix of secret key and public key
def key_matrix(f,g,F,G,h):
    A = matrix(ZZ,2*N)
    B = matrix(ZZ,2*N)
    for i in range(N*2):
        if i < N :
            for j in range(i,N):
                B[i,j] = g[j-i]
                B[i,j+N] = -f[j-i]
                A[i,j+N] = h[j-i]
            for j in range(i):
                k = (j-i+N)
                B[i,j] = -g[k]
                B[i,j+N] = f[k]
                A[i,j+N] = -h[k]
            A[i,i] = 1
        else:
            i_n = i-N
            for j in range(i_n,N):
                B[i,j] = G[j-i_n]
                B[i,j+N] = -F[j-i_n]
            for j in range(i_n):
                k = (j-i+2*N)
                B[i,j] = -G[k]
                B[i,j+N] = F[k]
            A[i,i] = 12289
    return A,B

# generate an anti-circulant matrix
def anti_cir(poly):
    mtx = matrix(RR,len(poly))

    for j in range(len(poly)):
        mtx[0,j] = poly[j]

    for i in range(1,len(poly)) :
        for j in range(len(poly)):
            if j - 1 >= 0:
                mtx[i,j] = mtx[i-1,j-1]
            else:
                mtx[i,j] = - mtx[i-1,len(poly)-1]

    return mtx