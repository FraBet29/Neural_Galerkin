import numpy as np
from wasserstein import wasserstein_1d


class SVGD():

    def __init__(self):
        pass
    
    def svgd_kernel(self, theta, h = -1):
        theta_norm_squared = np.sum(theta ** 2, axis=1, keepdims=True)
        pairwise_dists = theta_norm_squared + theta_norm_squared.T - 2 * np.dot(theta, theta.T)
        if h < 0: # if h < 0, using median trick
            h = np.median(pairwise_dists)  
            h = np.sqrt(0.5 * h / np.log(theta.shape[0]+1))

        Kxy = np.exp(- pairwise_dists / h ** 2 / 2) # RBF kernel

        dxkxy = - np.matmul(Kxy, theta)
        sumkxy = np.sum(Kxy, axis=1)
        dxkxy += np.multiply(theta, np.expand_dims(sumkxy, axis=1)) # vectorized
        dxkxy = dxkxy / (h**2)
        return (Kxy, dxkxy)
    
 
    def update(self, x0, lnprob, n_iter=1000, stepsize=1e-3, h=-1, alpha=0.9, debug=False):
        
        theta = np.copy(x0)
        theta_old = np.copy(x0)
        theta_hist = []
        wass_hist = []
        
        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0

        for iter in range(n_iter):

            if debug and (iter+1) % 100 == 0:
                print('iter ' + str(iter+1))

            # record history
            theta_hist.append(theta)
            
            lnpgrad = lnprob(theta)
            # calculating the kernel matrix
            kxy, dxkxy = self.svgd_kernel(theta, h=0.05)  
            grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / x0.shape[0]  
            
            # adagrad 
            if iter == 0:
                historical_grad = historical_grad + grad_theta ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
            adj_grad = np.divide(grad_theta, fudge_factor + np.sqrt(historical_grad))

            theta = theta + stepsize * adj_grad

            # vanilla update
            # theta = theta + stepsize * grad_theta

            # wasserstein
            wass = wasserstein_1d(theta_old.squeeze(), theta.squeeze(), p=2)
            wass_hist.append(wass)
            theta_old = np.copy(theta)
            
        return theta, theta_hist, wass_hist
    
