import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from iznetwork import *

################### parameters for the network ###################

SCALE_FACTOR_CONNECTIONS={"excitatory-excitatory": 17,
                           "excitatory-inhibitory": 50,
                           "inhibitory-excitatory": 2,
                           "inhibitory-inhibitory": 1}
NUM_EXCITATORY_MODULES=8
EXCITATORY_MODULE_SIZE=100
INHIBITORY_MODULE_SIZE=200
EDGES_FOR_EXCITATORY_MODULE=1000
FOCAL_CONNECTIONS=4
TOTAL_N = NUM_EXCITATORY_MODULES * EXCITATORY_MODULE_SIZE + INHIBITORY_MODULE_SIZE
EXCITATORY_MAX_DELAY=20
# izhikevich model parameters
a_excitatory, b_excitatory, c_excitatory, d_excitatory = 0.02, 0.2, -65, 8
a_inhibitory, b_inhibitory, c_inhibitory, d_inhibitory = 0.02, 0.25, -65, 2

################### building the network ###################

def build_basic_connectivity_matrix():
    """
    build a connectivity matrix for the dynamical modular network described in the assignment, before rewiring
    1. randomly assign edges for each excitatory module
    2. connect inhibitory neurons "all to all"
    3. focal connections from excitatory neurons to inhibitory (4:1)
    :return: connectivity matrix A with shape (n, n)
    """

    n_excitatory = NUM_EXCITATORY_MODULES * EXCITATORY_MODULE_SIZE
    connectivity_A = np.zeros((TOTAL_N, TOTAL_N))

    # randomly assign edges for each excitatory module
    for c in range(NUM_EXCITATORY_MODULES):
        for edge in range(EDGES_FOR_EXCITATORY_MODULE):
            i = np.random.choice(range(c * EXCITATORY_MODULE_SIZE, (c + 1) * EXCITATORY_MODULE_SIZE))
            j = np.random.choice(range(c * EXCITATORY_MODULE_SIZE, (c + 1) * EXCITATORY_MODULE_SIZE))
            while j == i or connectivity_A[i][j] == 1:  # this is an invalid edge / already exists
                i = np.random.choice(range(c * EXCITATORY_MODULE_SIZE, (c + 1) * EXCITATORY_MODULE_SIZE))
                j = np.random.choice(range(c * EXCITATORY_MODULE_SIZE, (c + 1) * EXCITATORY_MODULE_SIZE))
            connectivity_A[i][j] = 1

    # connect inhibitory neurons "all to all"
    for inhibitory_n in range(n_excitatory, TOTAL_N):
        connectivity_A[inhibitory_n] = [0 if i == inhibitory_n else 1 for i in range(TOTAL_N)]

    # focal connections from excitatory neurons to inhibitory (4:1)
    j = n_excitatory
    for i in range(0, n_excitatory, FOCAL_CONNECTIONS):
        for x in range(FOCAL_CONNECTIONS):  # add 4 connections
            connectivity_A[i + x][j] = 1
        j += 1

    return connectivity_A

def rewire_excitatory_neurons(connectivity_A, rewiring_prob_p):
    """
    rewire the excitatory neurons with probability p (in place)
    :param connectivity_A: connectivity matrix
    :param rewiring_prob_p: rewiring probability
    :return: rewired connectivity matrix
    """
    # rewire the excitatory
    for c in range(NUM_EXCITATORY_MODULES):
        other_communities = [x for x in range(TOTAL_N - INHIBITORY_MODULE_SIZE)
                             if x not in range(c * EXCITATORY_MODULE_SIZE, (c + 1) * EXCITATORY_MODULE_SIZE)]
        for i in range(c * EXCITATORY_MODULE_SIZE, (c + 1) * EXCITATORY_MODULE_SIZE):
            for j in range(c * EXCITATORY_MODULE_SIZE, (c + 1) * EXCITATORY_MODULE_SIZE):
                if connectivity_A[i][j] == 1 and np.random.rand() < rewiring_prob_p:
                    h = np.random.choice(other_communities)
                    while connectivity_A[i][h] == 1 or h == i:
                        h = np.random.choice(other_communities)
                    connectivity_A[i][j] = 0
                    connectivity_A[i][h] = 1
    return connectivity_A

def build_weights_matrix(connectivity_A):
    """
    build the weighted connectivity matrix W based on the parameters in the assignment
    :param connectivity_A: connectivity matrix
    :return: weighted connectivity matrix W
    """
    n_excitatory = NUM_EXCITATORY_MODULES * EXCITATORY_MODULE_SIZE
    weighted_connectivity_W = copy.deepcopy(connectivity_A)
    # excitatory - excitatory
    weighted_connectivity_W[:n_excitatory, :n_excitatory] *= SCALE_FACTOR_CONNECTIONS["excitatory-excitatory"]
    # excitatory - inhibitory
    weighted_connectivity_W[:n_excitatory, n_excitatory:] *= (
            np.random.uniform(0, 1, (n_excitatory, INHIBITORY_MODULE_SIZE)) *
            SCALE_FACTOR_CONNECTIONS["excitatory-inhibitory"])
    # inhibitory - excitatory
    weighted_connectivity_W[n_excitatory:, :n_excitatory] *= (
            np.random.uniform(-1, 0, (INHIBITORY_MODULE_SIZE, n_excitatory)) *
            SCALE_FACTOR_CONNECTIONS["inhibitory-excitatory"])
    # inhibitory - inhibitory
    weighted_connectivity_W[n_excitatory:, n_excitatory:] *= (
            np.random.uniform(-1, 0, (INHIBITORY_MODULE_SIZE, INHIBITORY_MODULE_SIZE)) *
            SCALE_FACTOR_CONNECTIONS["inhibitory-inhibitory"])
    return weighted_connectivity_W

def build_delay_matrix():
    """
    build the delay matrix D based on the parameters in the assignment
    :return: delay matrix D
    """
    n_excitatory = NUM_EXCITATORY_MODULES * EXCITATORY_MODULE_SIZE
    delay_matrix = np.ones((TOTAL_N, TOTAL_N))
    delay_matrix[:n_excitatory, :n_excitatory] *= np.random.randint(1, EXCITATORY_MAX_DELAY+1,
                                                                    (n_excitatory, n_excitatory))
    return delay_matrix.astype(int)

def set_heterogeneous_params(net):
    """
    set the heterogeneous parameters for the neurons in the network (in place)
    :param net: network object (IzNetwork)
    :return: the network with the heterogeneous parameters
    """
    # building heterogeneous params for excitatory
    r_excitatory = np.random.rand(NUM_EXCITATORY_MODULES * EXCITATORY_MODULE_SIZE)
    n_excitatory = NUM_EXCITATORY_MODULES * EXCITATORY_MODULE_SIZE
    a_hetero_excitatory_array = np.array([a_excitatory] * n_excitatory)
    b_hetero_excitatory_array = np.array([b_excitatory] * n_excitatory)
    c_hetero_excitatory_array = np.array(c_excitatory + 15 * (r_excitatory ** 2))
    d_hetero_excitatory_array = np.array(d_excitatory - 6 * (r_excitatory ** 2))
    # building heterogeneous params for inhibitory
    r_inhibitory = np.random.rand(INHIBITORY_MODULE_SIZE)
    a_hetero_inhibitory_array = a_inhibitory + 0.08 * r_inhibitory
    b_hetero_inhibitory_array = b_inhibitory - 0.05 * r_inhibitory
    c_hetero_inhibitory_array = np.array([c_inhibitory] * INHIBITORY_MODULE_SIZE)
    d_hetero_inhibitory_array = np.array([d_inhibitory] * INHIBITORY_MODULE_SIZE)

    net.setParameters(a=np.concat([a_hetero_excitatory_array, a_hetero_inhibitory_array]),
                      b=np.concat([b_hetero_excitatory_array, b_hetero_inhibitory_array]),
                      c=np.concat([c_hetero_excitatory_array, c_hetero_inhibitory_array]),
                      d=np.concat([d_hetero_excitatory_array, d_hetero_inhibitory_array]))
    return net

def build_network(rewiring_prob_p=0.0):
    """
    build the network using the parameters in the assignment
    :param rewiring_prob_p: rewiring probability
    :return: network, connectivity matrix (A), weighted connectivity matrix (W), delay matrix (D)
    """
    connectivity_A = build_basic_connectivity_matrix()
    connectivity_A = rewire_excitatory_neurons(connectivity_A, rewiring_prob_p)
    weighted_connectivity_W = build_weights_matrix(connectivity_A)
    delay_matrix_D = build_delay_matrix()
    net = IzNetwork(connectivity_A.shape[0], int(np.max(delay_matrix_D)))
    net.setWeights(weighted_connectivity_W)
    net.setDelays(delay_matrix_D)
    net = set_heterogeneous_params(net)
    return net, connectivity_A, weighted_connectivity_W, delay_matrix_D

################### running the experiment ###################

def run_experiment(net, T=1000, pois_distribution_param=0.01, output_dir_path='outputs',
                   rewiring_prob_p=0.0, const_current=15):
    """
    run the dynamical experiment for the network - update the network for T time steps and plot the results
    :param net: network object (IzNetwork)
    :param T: number of time steps
    :param pois_distribution_param: poisson distribution parameter
    :param output_dir_path: output directory path - where to save the plots
    :param rewiring_prob_p: rewiring probability (only used for the plots titles) 
    :param const_current: constant current
    """
    number_of_excitatory = NUM_EXCITATORY_MODULES * EXCITATORY_MODULE_SIZE
    V = np.zeros((T, TOTAL_N))
    for t in range(T):
        current = (const_current * np.random.poisson(pois_distribution_param,  TOTAL_N))
        net.setCurrent(current)
        net.update()
        V[t, :], _ = net.getState()
    raster_plot(V[:, :number_of_excitatory], output_dir_path, p=rewiring_prob_p)
    plot_mean_firing_rate(V, output_dir_path, p=rewiring_prob_p)


################### plotting helpers ###################

def raster_plot(V:np.array, output_dir_path:str, threshold=29.9, p=0.0):
    """
    save a raster plot corresponding to the V matrix given, in the output directory
    :param V: membrane potential matrix - shape (T, N)
    :param output_dir_path: string, output directory path
    :param threshold: float, threshold that defines when membrane potential is considered a spike
    :param p: float, rewiring probability (only used for the plot title)
    :return: None
    """
    t_spikes, neuron_spikes = np.where(V > threshold)
    plt.figure(figsize=(8, 4))
    plt.scatter(t_spikes, neuron_spikes, s=4)
    plt.gca().invert_yaxis()
    plt.title(f'Raster plot for p={p:.1f}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron number')
    plt.savefig(os.path.join(output_dir_path, f'Raster_plot_p={p:.1f}.pdf'))
    plt.close()

def plot_mean_firing_rate(V:np.array, output_dir:str, threshold=29.9, p=0.0, window_size=50, step_size=20):
    """
    plot the mean firing rate for all communities over time.
    :param V: membrane potential matrix - shape (T, N)
    :param output_dir: string, output directory path
    :param threshold: float, threshold that defines when membrane potential is considered a spike
    :param p: float, rewiring probability (only used for the plot title)
    :param window_size: int
    :param step_size: int
    :return: None
    """
    T = V.shape[0]
    mean_firing_rate = np.zeros((8, 50))
    spikes = (V > threshold).astype(int)
    plt.figure(figsize=(14,6))
    pad_amount = window_size - step_size
    spikes = np.pad(spikes, ((0, pad_amount), (0, 0)), 'constant', constant_values=0)
    for c in range(NUM_EXCITATORY_MODULES):
        window = 0
        for t in range(0, T, step_size):
            mean_firing_rate[c][window] = float(np.sum(spikes[t:t+window_size,
                                             c*EXCITATORY_MODULE_SIZE:
                                             (c+1)*EXCITATORY_MODULE_SIZE]))/window_size
            window+=1
        plt.plot(range(0, T, step_size), mean_firing_rate[c,:], label=f"community {c+1}")
    plt.legend()
    plt.xlabel("Time (ms)")
    plt.ylabel("Mean firing rate (Hz)")
    plt.title(f"Mean firing rate of each community over time, for net with rewiring probability p={p:.1f}")
    plt.savefig(os.path.join(output_dir, "mean_firing_rate.pdf"))
    plt.close()

def plot_connectivity_matrix(connectivity_A:np.array, output_dir:str, p=0.0):
    """
    plot the connectivity matrix (A)
    :param connectivity_A: connectivity matrix
    :param output_dir: string, output directory path
    :param p: float, rewiring probability (only used for the title)
    :return: None
    """
    plt.imshow(connectivity_A, cmap='viridis', interpolation='none')
    plt.title(f'Connectivity matrix for p={p:.1f}')
    plt.savefig(os.path.join(output_dir, f'Connectivity_matrix_p={p:.1f}.pdf'))
    plt.close()

def plot_weights_matrix(weights_connectivity_W:np.array, output_dir:str, p=0.0):
    """
    plot the weights matrix (W)
    :param weights_connectivity_W: the weights matrix (W)
    :param output_dir: string, output directory path
    :param p: float, rewiring probability (only used for the title)
    """
    max_w = np.max(weights_connectivity_W)
    plt.imshow(weights_connectivity_W, cmap='seismic', interpolation='none', vmin=-abs(max_w), vmax=abs(max_w))
    plt.title(f'Weighted connectivity matrix for p={p:.1f}')
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, f'Weighted_connectivity_matrix_p={p:.1f}.pdf'))
    plt.close()


################### main ###################

def main():
    """
    build the network and run the experiment for each rewiring probability p
    save the results in the outputs folder of this project directory
    """
    for p in np.arange(0, 0.6, 0.1):
        # output directory
        current_dir = os.path.join('outputs', f'p_{p:.1f}')
        if not os.path.exists(current_dir):
            os.makedirs(current_dir)

        # building the network
        net, connectivity_A, weighted_connectivity_W, delay_mat_D = build_network(rewiring_prob_p=p)

        # plots of the connectivity and the weights matrices
        plot_connectivity_matrix(connectivity_A, current_dir, p=p)
        plot_weights_matrix(weighted_connectivity_W, current_dir, p=p)

        # run the dynamical experiment
        run_experiment(net, output_dir_path=current_dir, rewiring_prob_p=p)


if __name__ == '__main__':
    main()


