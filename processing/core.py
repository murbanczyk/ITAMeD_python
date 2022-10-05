import numpy as np
from alive_progress import alive_bar


def itamed1d(iter, diffusion_range, signal, b, llambda, expclass):
    # Function for main processing of 1D Laplace data
    # Inputs:
    # * iter -            maximum number of iterations
    # * diffusion_range - vector of 3 parameters [Minimal Diffusion, Maximum diffusion, number of points]
    #                     In case of T1 and T2 experiments T1 or T2 time respectively
    # * signal -          Signal (Diffusion decay, CPMG, T1 saturation or inversion

    # * b     -           vector of b in eq.   I = exp(-D * b)

    # * llambda -            Lagrangian multiplier
    # * expclass - Experiment class. Available choices:
    #                                               'D' - diffusion experiment,
    #                                               'T2'- CPMG,
    #                                               'T1' T1 inversion recovery,
    #                                               'T1sat', T1 saturation recovery
    #
    #
    # Outputs:
    # * d_scale  - Vector of Diffusion coefficient of relaxation times
    # * out_array - Result of ITAMeD ILT

    signal = np.array(signal)
    b = np.array(b)
    d_scale = (np.logspace(np.log10(diffusion_range[0]),
                           np.log10(diffusion_range[1]), diffusion_range[2]))
    a = np.zeros(d_scale.shape)
    mat = generate_matrix_1d(d_scale, b, expclass)
    out_array = fista(a, signal, llambda, mat, iter)
    return d_scale, out_array

def itamed_ftilt(iter,diffusion_range, signal, schedule, SW, indirect_size, llambda, threshold):
    # This is the main function for reconstruction of data from sparse joined
    #Sampled FT-ILT experiment e.g. HSQCiDOSY with Non-uniformly sampled indirect dimension and diffusion dimension
    #Variables:
    #Signal - complex matrix 1st dimension if FT of directly detected dimension 2nd dimension is joined sampled FT-ILT dimension
    #schedule - sampling schedule 1st column is increment in indirect dimension 2nd column is b-value from diffusion dim. Like in eq. I=exp(-D*b);
    #SW= spectral width of indirect dimension
    #indirect_size - size of reconstruction of indirect fourier dimension
    #diffusion_range the grid on which you want to reconstruct diffusion dimension
    #iter - number of iterations. Suggested 1e4;
    #llambda - regularization parameter. Sugested value: 100;
    # threshold - cutoff value for direct dimension. Below this value FTILT is not processed and replaced with zeros.
    d_scale = (np.logspace(np.log10(diffusion_range[0]),
                           np.log10(diffusion_range[1]), diffusion_range[2]))
    gradients = schedule[:, 1]
    # for k,l in enumerate(d_scale):
    #     exp_vectors=[:,k]=np.exp(-l*gradients)
    lipshitz = 0.0047
    for k,fid in enumerate(signal):
        if np.abs(np.sum(fid))>threshold:
            flat_spectrum = fista_ftilt(fid,  llambda, schedule, indirect_size,exp_vectors,iter,lipshitz)

        else:
            flat_spectrum=np.zeros(indirect_size,diffusion_range[2])

        final_spectrum[k,:,:]=flat_spectrum

    return final_spectrum, d_scale






def itamed2d(iter, diffusion_range, diffusion_range2, signal, b, b2, llambda, expclass, expclass2):
    # Function for main processing of 2D Laplace data
    # Inputs:
    # * iter -             maximum number of iterations
    # * diffusion_range -  vector of 3 parameters [Minimal Diffusion, Maximum diffusion, number of points]
    #                      for 1st dimension
    #                      In case of T1 and T2 experiments T1 or T2 time respectively
    # * diffusion_range2 - vector of 3 parameters [Minimal Diffusion, Maximum diffusion, number of points]
    #                      for 2nd dimension
    #                      In case of T1 and T2 experiments T1 or T2 time respectively
    # * signal -           Signal (Diffusion decay, CPMG, T1 saturation or inversion ) Combination of those two

    # * b     -           vector of b in eq.   I = exp(-D * b) 1st dimension
    # * b2     -           vector of b in eq.   I = exp(-D * b) 2nd dimension

    # * llambda -            Lagrangian multiplier
    # * expclass - Experiment class for 1st dimension. Available choices:
    #                                               'D' - diffusion experiment,
    #                                               'T2'- CPMG,
    #                                               'T1' T1 inversion recovery,
    #                                               'T1sat', T1 saturation recovery
    # * expclass2 - Experiment class for 2nd dimension. Available choices:
    #                                               'D' - diffusion experiment,
    #                                               'T2'- CPMG,
    #                                               'T1' T1 inversion recovery,
    #                                               'T1sat', T1 saturation recovery
    #
    #
    # Outputs:
    # * d_scale  - Vector of Diffusion coefficient of relaxation times for 1st dimension
    # * d_scale2  - Vector of Diffusion coefficient of relaxation times for 2nd dimension
    # * out_array - Result of ITAMeD ILT

    signal = np.matrix(signal)
    b = np.array(b)
    b2 = np.array(b2)
    d_scale = (np.logspace(np.log10(diffusion_range[0]),
                           np.log10(diffusion_range[1]), diffusion_range[2]))
    d_scale2 = (np.logspace(np.log10(diffusion_range2[0]),
                            np.log10(diffusion_range2[1]), diffusion_range2[2]))
    a = np.zeros([d_scale.shape[0], d_scale2.shape[0]])
    mat, mat2 = generate_matrix_2d(d_scale, d_scale2, b, b2, expclass, expclass2)
    out_array = fista(a, signal, llambda, mat, iter, False, 0, mat2)
    return d_scale, d_scale2, out_array


def generate_matrix_1d(d, data, expclass):
    size_d = len(d)
    mat = np.zeros([len(data), len(d)])
    for i in range(0, size_d):
        for j in range(0, len(data)):
            if expclass == 'D':
                mat[j, i] = np.exp(-d[i] * data[j])
            elif expclass == 'T2':
                mat[j, i] = np.exp(-1 / d[i] * data[j])
            elif expclass == 'T1':
                mat[j, i] = 1 - 2 * np.exp(-1 / d[i] * data[j])
            elif expclass == 'T1sat':
                mat[j, i] = 1 - np.exp(-1 / d[i] * data[j])
            else:
                raise Exception("This class of experiments: %s is not included in ITAMeD" % expclass)

    return mat


def generate_matrix_2d(d1, d2, b1, b2, expclass1, expclass2):
    mat1 = generate_matrix_1d(d1, b1, expclass1)
    mat2 = generate_matrix_1d(d2, b2, expclass2)
    return mat1, mat2


def lt_lt(y, mat1, mat2):
    b = np.matrix(np.zeros([mat2.shape[0], mat1.shape[0]]))
    b1 = np.matrix(np.zeros([mat2.shape[1], mat1.shape[0]]))
    for i in range(0, y.shape[0]):
        b1[i, :] = np.array(mat1 * y[i, :].T)[:, 0]
    for i in range(0, b1.shape[1]):
        b[:, i] = (mat2 * b1[:, i])
    return b


def ilt_ilt(y, mat1, mat2):
    b = np.matrix(np.zeros([mat2.shape[1], mat1.shape[1]]))
    b1 = np.matrix(np.zeros([mat2.shape[1], mat1.shape[0]]))
    for i in range(0, y.shape[1]):
        b1[:, i] = (mat2.T * y[:, i])
    for i in range(0, b1.shape[0]):
        b[i, :] = (mat1.T * (b1[i, :]).T).T
    return b
def ift_lt(y, matal, matal2):
    pass

def ft_ilt(x,sched,NIMAX,exp_vector):
    uniq=len(np.unique(sched[:,0]))
#     # UniqueSamp=length(unique(sched(:,1)));
    y=np.zeros(NIMAX[0], np.shape(exp_vector,1))
    # Y=np.zeros(NIMAX[0], np.shape(exp_vector,1))

    # Y=zeros(NIMAX(1),size(exp_vector,2));
    for k in range(0, NIMAX[0]):
    # for k=1:NIMAX(1)
        Y[k,:]= exp_vector.T * x[k,:].T
#                Y(k,:)=exp_vector'* ((x(k,:))');
#
#     % =W;
#
#     end
    Y=np.fft.fftshift(np.fft.fft(Y,0),0)/np.sqrt((NIMAX[0]*np.sqrt((uniq/NIMAX[0]))))
    return Y
#     Y=fftshift(fft(Y,[],1),1)/sqrt(NIMAX(1))*sqrt(UniqueSamp/NIMAX(1));
#         pass
def fista_ftilt(fid,  lambdal, schedule, NIMAX,exp_vectors,iter,t):

    s = 1;
    signal = np.matrix(np.zeros(NIMAX[0], np.shape(exp_vectors,2)))
    for i, sch in enumerate(schedule):
        signal[sch[0], i] = fid[i]
    c = np.matrix(np.zeros(np.shape(signal)))
    y =  np.matrix(np.zeros(np.shape(signal)))
    x =  np.matrix(np.zeros(np.shape(signal)))
    x1 =  np.matrix(np.zeros(np.shape(signal)))
    for k in range(0, iter):
        b = ift_lt(y, matal, matal2)
        c = y - 2 * t * ft_ilt(b - signal, sched, NIMAX, exp_vector)
        x1 = np.abs(c) - lambdal * t
        x1 = (x1 + np.abs(x1)) / 2
        x1 = np.multiply(np.sign(c), x1)
        x1 = (x1 + np.abs(x1)) / 2
        s1 = (1 + np.sqrt((1 + 4 * s ** 2))) / 2
        y = x1 + (s - 1) / s1 * (x1 - x)
        s = s1
        x = np.matrix(x1)
    return x


def fista(a, signal, lambdal, matal, iterk, onedim=True, t=0, matal2=0):
    iterk = int(iterk)
    signal = np.matrix(signal).T
    matal = np.matrix(matal)
    matal2 = np.matrix(matal2)
    if t == 0:
        if onedim:
            t = 1 / np.max(np.real(2 * np.linalg.eig(np.matmul(matal.T, matal))[0]))
        else:
            t = 1 / np.max(np.real(2 * np.linalg.eig(np.matmul(matal.T, matal))[0])) \
                * 1 / np.max(np.real(2 * np.linalg.eig(np.matmul(matal2.T, matal2))[0]))
    a = np.matrix(a)
    y = np.matrix(a.T)
    x = np.matrix(y)
    s = 1
    # b = np.matrix(np.zeros(signal.shape))
    tmatal = 2 * t * matal.T
    with alive_bar(iterk, force_tty=True) as bar:
        for kw in range(iterk):
            if onedim:
                c = y - tmatal * (matal * y - signal)
            else:  # for 2D mode
                b = lt_lt(y, matal, matal2)
                c = y - 2 * t * ilt_ilt(b - signal, matal, matal2)
            x1 = np.abs(c) - lambdal * t
            x1 = (x1 + np.abs(x1)) / 2
            x1 = np.multiply(np.sign(c), x1)
            x1 = (x1 + np.abs(x1)) / 2
            s1 = (1 + np.sqrt((1 + 4 * s ** 2))) / 2
            y = x1 + (s - 1) / s1 * (x1 - x)
            s = s1
            x = np.matrix(x1)
            bar()
    return x
