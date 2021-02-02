
def aligg(X,Y,d,cm,mo,ep,kn):
    

    corr=cm
    Wxy=corr
    num_dims=d
    #     mu=0.5
    #     eps=1e-5
    mu=mo
    eps=ep

    # from distance import SquaredL2
    import numpy as np
    from scipy.linalg import block_diag, eigh, svd
    from scipy.sparse.csgraph import laplacian
    from mpl_toolkits.mplot3d import Axes3D
    from distance import SquaredL2




    def neighbor_graph(X, metric=SquaredL2, k=None, epsilon=None, symmetrize=True):
      '''Construct an adj matrix from a matrix of points (one per row)'''
      assert (k is None) ^ (epsilon is None), "Must provide `k` xor `epsilon`"
      dist = metric.within(X)
      adj = np.zeros(dist.shape)  # TODO: scipy.sparse support, or at least use a smaller dtype
      if k is not None:
        # do k-nearest neighbors
        nn = np.argsort(dist)[:,:min(k+1,len(X))]
        # nn's first column is the point idx, rest are neighbor idxs
        if symmetrize:
          for inds in nn:
            adj[inds[0],inds[1:]] = 1
            adj[inds[1:],inds[0]] = 1
        else:
          for inds in nn:
            adj[inds[0],inds[1:]] = 1
      else:
        # do epsilon-ball
        p_idxs, n_idxs = np.nonzero(dist<=epsilon)
        for p_idx, n_idx in zip(p_idxs, n_idxs):
          if p_idx != n_idx:  # ignore self-neighbor connections
            adj[p_idx,n_idx] = 1
        # epsilon-ball is typically symmetric, assuming a normal distance metric
      return adj


    Wx = neighbor_graph(X,k=kn)
    Wy = neighbor_graph(Y,k=kn)
    

    # def manifold_nonlinear(X,Y,corr,num_dims,Wx,Wy,mu=0.9,eps=1e-8):
    #   L = _manifold_setup(Wx,Wy,corr.matrix(),mu)
    #   return _manifold_decompose(L,X.shape[0],Y.shape[0],num_dims,eps)

    # def _manifold_setup(Wx,Wy,Wxy,mu):
    #   Wxy = mu * (Wx.sum() + Wy.sum()) / (2 * Wxy.sum()) * Wxy
    #   W = np.asarray(np.bmat(((Wx,Wxy),(Wxy.T,Wy))))
    #   return laplacian(W)

    # def _manifold_decompose(L,d1,d2,num_dims,eps,vec_func=None):
    #   vals,vecs = np.linalg.eig(L)
    #   idx = np.argsort(vals)
    #   for i in xrange(len(idx)):
    #     if vals[idx[i]] >= eps:
    #       break
    #   vecs = vecs.real[:,idx[i:]]
    #   if vec_func:
    #     vecs = vec_func(vecs)
    #   for i in xrange(vecs.shape[1]):
    #     vecs[:,i] /= np.linalg.norm(vecs[:,i])
    #   map1 = vecs[:d1,:num_dims]
    #   map2 = vecs[d1:d1+d2,:num_dims]
    #   return map1,map2

    Wxy = mu * (Wx.sum() + Wy.sum()) / (2 * Wxy.sum()) * Wxy
    W = np.asarray(np.bmat(((Wx,Wxy),(Wxy.T,Wy))))
    L = laplacian(W)

    d1 = X.shape[0]
    d2 = Y.shape[0]
    vec_func = None

    vals,vecs = np.linalg.eig(L)
    idx = np.argsort(vals)
    #for i in xrange(len(idx)):
    for i in range(len(idx)):
        if vals[idx[i]] >= eps:
            break
    vecs = vecs.real[:,idx[i:]]
    if vec_func:
        vecs = vec_func(vecs)
    #for i in xrange(vecs.shape[1]):
    for i in range(vecs.shape[1]):
        vecs[:,i] /= np.linalg.norm(vecs[:,i])
    map11 = vecs[:d1,:num_dims]
    map22 = vecs[d1:d1+d2,:num_dims]
    import numpy as np
    import matplotlib.pyplot as plt
    if num_dims==2:
        plt.scatter(map11[:,0],map11[:,1],s=30,color='black')
        plt.scatter(map22[:,0],map22[:,1],s=10,color='green')
        plt.show()
    if num_dims==3:
        # Import libraries
        from mpl_toolkits import mplot3d
        # Creating figure
        fig = plt.figure(figsize = (10, 7))
        ax = plt.axes(projection ="3d")
        # Creating plot
        ax.scatter3D(map11[:,0],map11[:,1], map11[:,2],s=30, color = "red")
        ax.scatter3D(map22[:,0],map22[:,1], map22[:,2],s=10, color = "blue")
        #plt.title("simple 3D scatter plot")
        # show plot
        plt.show()
    return map11,map22
