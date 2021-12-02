import numpy as np

def read_data(imagefile, labelfile, N, M):
    x = np.zeros((N,M), dtype='uint8')
    images = open(imagefile, 'rb')
    images.read(16)  # skip the magic_number
    for n in range(N):
        x[n,:] = np.fromstring(images.read(M), dtype='uint8')
    images.close()
    labels = open(labelfile, 'rb')
    labels.read(8)  # skip the magic_number
    xl = np.fromstring(labels.read(N), dtype='uint8')
    labels.close()
    print(imagefile,", ", labelfile)
    return (x, xl)

def get_one_image(x, L, l):
    z = x.reshape((L,L))
    y = np.zeros(((L-l-4)*(L-l-4), l*l))
    n = 0
    for i in range(2,L-l-2):
        for j in range(2,L-l-2):
            y[n,:] = z[i:i+l,j:j+l].flatten()
            y[n,:] = y[n,:] - np.mean(y[n,:])
            y[n,:] = y[n,:]/np.linalg.norm(y[n,:])
            n = n + 1
    return y

def features(x, xl, K, L, l, Q, tm):
    (N, M), LL, ll = x.shape, (L-l-4)*(L-l-4), l*l
    u = np.random.rand(K*Q, l*l) # centroids
    ul = np.zeros(K*Q, dtype='uint8') # centroids labels
    t = 0
    for k in range(K):
        for q in range(Q):
            u[t,:] = u[t,:] - np.mean(u[t,:])
            u[t,:] = u[t,:]/np.linalg.norm(u[t,:])
            ul[t] = k
            t = t + 1
    s = np.zeros(K*Q, dtype='int')
    for k in range(K):
        kQ, kkQ = k*Q, (k+1)*Q
        for t in range(tm):
            v, s[kQ:kkQ] = np.zeros((Q, ll)), np.zeros(Q)
            for n in range(N):
                if xl[n] == k:
                    y = get_one_image(x[n,:], L, l)
                    r = np.dot(u[kQ:kkQ,:], y.T)
                    for i in range(LL):
                        q = np.argmax(r[:,i])
                        v[q,:] = v[q,:] + y[i,:]
                        s[q+kQ] = s[q+kQ] + 1
            for q in range(Q):
                if s[q+kQ] > 0:
                    u[q+kQ,:] = v[q,:]/np.linalg.norm(v[q,:])
            print(round((k*tm+t+1)*100.0/(K*tm),2), '%', end='\r', flush=True)
    print()
    QQ = np.sum(s>0)
    v, vl, n = np.zeros((QQ,ll)), np.zeros(QQ, dtype='uint8'), 0
    for q in range(K*Q):
        if s[q] > 0:
            v[n,:], vl[n] = u[q,:], ul[q]
            n = n + 1
    print("number of features=", len(vl))
    return (v, vl)

def duck_test(y, yl, u, ul, K, L, l):
    (J, M), acc, LL = y.shape, 0, (L-l-4)*(L-l-4)
    for n in range(J):
        z = get_one_image(y[n,:], L, l)
        r = np.dot(u, z.T)
        h = np.zeros(K).astype(int)
        for i in range(LL):
            q = np.argmax(r[:,i])
            h[ul[q]] = h[ul[q]] + 1
        k = np.argmax(h)
        if k == yl[n]:
            acc = acc + 1
        if (n+1) % 100 == 0:
            print(round((n+1)*100.0/J,2), '%', end='\r', flush=True)
    print()
    return acc*100.0/J

if __name__ == '__main__':
    np.random.seed(1234) # for reproducibility
    print("Read data:")
    switch = False # Switch training data with test data
    if not switch:
        (x, xl) = read_data("train-images.idx3-ubyte",
                            "train-labels.idx1-ubyte", 60000, 784)
        (y, yl) = read_data("t10k-images.idx3-ubyte",
                            "t10k-labels.idx1-ubyte", 10000, 784)
    else:
        (y, yl) = read_data("train-images.idx3-ubyte",
                            "train-labels.idx1-ubyte", 60000, 784)
        (x, xl) = read_data("t10k-images.idx3-ubyte",
                            "t10k-labels.idx1-ubyte", 10000, 784)
    print("Extract features:")
    (u, ul) = features(x, xl, 10, 28, 18, 3000, 3)
    print("Apply the duck test:")
    acc = duck_test(y, yl, u, ul, 10, 28, 18)
    print("Accuracy =", round(acc, 3), "%")
