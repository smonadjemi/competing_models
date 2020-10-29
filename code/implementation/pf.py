import pathlib
import numpy as np
from random import seed, getstate, setstate
from re import search

def pf_boardrooms(
        data,
        ui_data,
        dimensions,
        d_dimensions,
        c_dimentions,
        sigma_params,
        d_params,
        d_mixture,
        p_flip,
        num_particles,
        filename=None,
        writeparticles=False
):
    """

    :return:
    """
    # sorted dot ids for each time step
    inds = None

    # Capture random state to reset later and set to 0
    rstate, nprstate = setRandomState()

    num_dots = data.shape[0]

    eps = 2 ** -52  # Defined to match Matlab's default eps

    # min and max value of each attribute, needed for random sampling
    c_domains = {atr: (min(data[atr]), max(data[atr])) for atr in c_dimentions}
    d_domains = {atr: {'categories': list(np.unique(data[atr])), 'proportions': list(len(data[data[atr] == c])/num_dots for c in np.unique(data[atr]))} for atr in d_dimensions}


    #num_types = dots[..., -1].max()

    # Compute log N(x; μ, σ²)
    lognormpdf = lambda x, mu, sigma: -0.5 * (
            ((x - mu) / sigma) ** 2 + np.log(2 * np.pi)
    ) - np.log(sigma)

    # Load session data
    #session_clicks = np.loadtxt(filename, delimiter=",", dtype=int)
    num_clicks = ui_data.size
    # Normalize dot locations to [0, 1] (need to copy due to Python pass-by-reference)
    #dots_norm = dots.copy()
    #dots_norm[..., 0:2] = (dots_norm[..., 0:2] - dots_norm[..., 0:2].min(axis=0)) / (
    #        dots_norm[..., 0:2].max(axis=0) - dots_norm[..., 0:2].min(axis=0)
    #)

    result = np.zeros(num_clicks - 1, dtype=int)

    # Find distribution of type counts
    #type_counts = np.bincount(dots_norm[..., 2].astype(int))[1:]

    # Initialize particles and the pi-vectors randomly
    particles = np.array([tuple(np.random.random()*(c_domains[atr][1] - c_domains[atr][0])+(c_domains[atr][0]) if atr in c_dimentions else np.random.choice(d_domains[atr]['categories'], p=d_domains[atr]['proportions']) for atr in dimensions) for pnum in range(num_particles)],
                         dtype=[(dim, data.dtype[dim]) for dim in dimensions])

    particles_pi = np.array([tuple(np.random.random(len(dimensions))) for nump in range(num_particles)],
                             dtype=[(dim, 'float') for dim in dimensions])



    if writeparticles:
        matches = search(r"(\d+)\/(\d+)", filename)
        location = f"../data/particles/task{matches[1]}/session{matches[2]}"
        pathlib.Path(location).mkdir(parents=True, exist_ok=True)
        np.savetxt(
            f"{location}/particles0.csv",
            particles,
            delimiter=",",
            header=','.join(dimensions),
            comments="",
        )

        matches = search(r"(\d+)\/(\d+)", filename)
        location = f"../data/particles/task{matches[1]}/session{matches[2]}"
        pathlib.Path(location).mkdir(parents=True, exist_ok=True)
        np.savetxt(
            f"{location}/particles_pi_0.csv",
            particles_pi,
            delimiter=",",
            header=','.join(dimensions),
            comments="",
        )

    # Particle filtering
    for i in range(num_clicks):
        # Diffuse particles
        diffuseParticlesBoardrooms(
            particles,
            particles_pi,
            d_params,
            d_mixture,
            num_particles,
            p_flip,
            d_domains,
            c_domains,
            dimensions,
            c_dimentions,
            d_dimensions,
            eps
        )

        '''
        if writeparticles:
            np.savetxt(
                f"{location}/particles{i}_d.csv",
                particles,
                delimiter=",",
                header="x,y,type,bias",
                comments="",
            )
        '''

        # Predict next click for each particle
        click_probabilities = getClickProbabilitiesBoardrooms(
            lognormpdf, particles, particles_pi, data, sigma_params, c_dimentions, d_dimensions
        )



        current_click_index = np.where(data == ui_data[i])[0][0]

        # Weight particles by evidence
        weights = click_probabilities[..., current_click_index]

        # Resample particles
        ind = np.random.choice(
            num_particles, num_particles, p=weights / np.sum(weights)
        )
        particles = particles[ind, ...]
        if writeparticles:
            np.savetxt(
                f"{location}/particles{i+1}.csv",
                particles,
                delimiter=",",
                header="x,y,type,bias",
                comments="",
            )

        # Record result of click prediction
        marginal_click_probabilities = np.mean(click_probabilities, axis=0)
        if writeparticles:
            np.savetxt(
                f"{location}/dot_weights{i+1}.csv",
                marginal_click_probabilities,
                delimiter=",",
            )
        ind = np.argsort(-marginal_click_probabilities)
        if inds is None:
            inds = ind
        else:
            inds = np.vstack((inds, ind))
        if i > 0:
            # shayan: index of user click in the sorted array of points
            result[i - 1] = np.nonzero(ind == current_click_index)[0]

    # Reset the states
    setstate(rstate)
    np.random.set_state(nprstate)

    return result, inds



def diffuseParticlesBoardrooms(
    particles,
    particles_pi,
    d_params,
    d_mixture,
    num_particles,
    p_flip,
    d_domains,
    c_domains,
    dimensions,
    c_dimensions,
    d_dimensions,
    eps,
):
    # Diffusion on locations
    """
    particles[..., 0:2] = particles[..., 0:2] + location_d * np.random.normal(
        size=particles[..., 0:2].shape
    )
    """

    # Diffusion on continuous attributes
    for attr in c_dimensions:
        # multiplying by the range of attributes bc I am not normalizing dimensions to be (0,1)
        particles[attr] = particles[attr] + d_params[attr] * np.random.normal(size=particles[attr].shape) * (c_domains[attr][1] - c_domains[attr][0])

    # Diffusion on mixture weights
    """
    particles[..., 3] = particles[..., 3] + mixture_d * np.random.normal(
        size=particles[..., 3].shape
    )
    """

    # Diffusion on matrix weights
    for attr in dimensions:
        particles_pi[attr] = particles_pi[attr] + d_mixture * np.random.normal(size=particles_pi[attr].shape)
        particles_pi[attr] = np.clip(particles_pi[attr], eps, 1 - eps)

    # "Discrete diffusion" on types
    """
    flip_ind = np.random.random(num_particles) < flip_p
    particles[flip_ind, 2] = np.random.choice(
        np.arange(1, num_types + 1),
        size=np.count_nonzero(flip_ind),
        p=type_counts / num_dots,
    )
    """

    # Diffusion on discrete attributes
    for attr in d_dimensions:
        flip_ind = np.random.random(num_particles) < p_flip
        particles[attr][flip_ind] = np.random.choice(d_domains[attr]['categories'], size=np.count_nonzero(flip_ind), p=d_domains[attr]['proportions'])

    # Clip values to boundary
    # particles[..., (0, 1, 3)] = np.clip(particles[..., (0, 1, 3)], eps, 1 - eps)


def getClickProbabilitiesBoardrooms(lognormpdf, particles, particles_pi, data, sigma_params, c_dimensions, d_dimensions):
    # Predict next click for each particle

    # These have to be calculated separately before adding, I don't know why
    """
    lnpdf0 = lognormpdf(particles[..., 0:1], dots_norm[..., 0:1].T, location_sigma)
    lnpdf1 = lognormpdf(particles[..., 1:2], dots_norm[..., 1:2].T, location_sigma)

    log_loc_prob = lnpdf0 + lnpdf1
    log_loc_prob -= np.max(log_loc_prob, axis=1)[..., np.newaxis]

    loc_prob = np.exp(log_loc_prob)
    loc_prob = loc_prob / np.sum(loc_prob, axis=1)[..., np.newaxis]
    """

    # convert structures arrays to ndarrays
    # x.view(np.float64).reshape(x.shape + (-1,))

    # len(particles) by len(data) matrix to hold probabilities of each particle given each dot was last click
    pdfs = {attr: np.array([[1.0 for i in range(len(data))] for j in range(len(particles))]) for attr in d_dimensions + c_dimensions}
    for attr in c_dimensions:
        lnpdf = np.array([[lognormpdf(particles[j][attr], data[i][attr], sigma_params[attr]) for i in range(len(data))] for j in range(len(particles))])
        lnpdf -= np.max(lnpdf, axis=1)[..., np.newaxis]
        pdf = np.exp(lnpdf)
        pdfs[attr] = pdf


    for attr in d_dimensions:

        #type_prob = dots_norm[..., 2:3].T == particles[..., 2:3]
        #type_prob = type_prob / np.sum(type_prob, axis=1)[..., np.newaxis]
        prob = particles[attr][..., None] == data[attr]
        prob = prob / np.sum(prob, axis=1)[..., np.newaxis]
        pdfs[attr] = prob

    # click_probabilities[j, ...] = particles[j, 3] * loc_prob + (1 - particles[j, 3]) * type_prob

    return sum([particles_pi[attr].reshape(-1, 1)*pdfs[attr] for attr in d_dimensions + c_dimensions])
    #return particles[..., 3:4] * loc_prob + (1 - particles[..., 3:4]) * type_prob



def pf(
    dots,
    location_sigma,
    location_d,
    mixture_d,
    flip_p,
    num_particles,
    filename,
    writeparticles=False,
):
    """The particle filtering algorithm

    `dots` -- the crime dots

    `location_sigma` -- p(click | particle) = N(click; particle, σ²I)

    `location_d` -- size of diffusion step for location

    `mixture_d` -- size of diffusion step for mixture weights

    `flip_p` -- probability of type flipping in diffusion step

    `num_particles` -- number of particles to use

    `filename` -- session filename
    """

    # sorted dot ids for each time step
    inds = None

    # Capture random state to reset later and set to 0
    rstate, nprstate = setRandomState()

    num_dots = dots.shape[0]
    num_types = dots[..., -1].max()
    eps = 2 ** -52  # Defined to match Matlab's default eps

    # Compute log N(x; μ, σ²)
    lognormpdf = lambda x, mu, sigma: -0.5 * (
        ((x - mu) / sigma) ** 2 + np.log(2 * np.pi)
    ) - np.log(sigma)

    # Load session data
    session_clicks = np.loadtxt(filename, delimiter=",", dtype=int)
    num_clicks = session_clicks.size
    # Normalize dot locations to [0, 1] (need to copy due to Python pass-by-reference)
    dots_norm = dots.copy()
    dots_norm[..., 0:2] = (dots_norm[..., 0:2] - dots_norm[..., 0:2].min(axis=0)) / (
        dots_norm[..., 0:2].max(axis=0) - dots_norm[..., 0:2].min(axis=0)
    )

    result = np.zeros(num_clicks - 1, dtype=int)

    # Find distribution of type counts
    type_counts = np.bincount(dots_norm[..., 2].astype(int))[1:]

    # Particles will be [x, y, type, π]
    particles = np.zeros((num_particles, 4))

    # Initialize particles randomly
    particles[..., (0, 1, 3)] = np.random.random(particles[..., (0, 1, 3)].shape)
    particles[..., 2] = np.random.choice(
        np.arange(1, num_types + 1),
        size=particles[..., 2].size,
        p=type_counts / num_dots,
    )

    if writeparticles:
        matches = search(r"(\d+)\/(\d+)", filename)
        location = f"../data/particles/task{matches[1]}/session{matches[2]}"
        pathlib.Path(location).mkdir(parents=True, exist_ok=True)
        np.savetxt(
            f"{location}/particles0.csv",
            particles,
            delimiter=",",
            header="x,y,type,bias",
            comments="",
        )

    # Particle filtering
    for i in range(num_clicks):
        # Diffuse particles
        diffuseParticles(
            particles,
            location_d,
            mixture_d,
            num_particles,
            flip_p,
            num_types,
            type_counts,
            num_dots,
            eps,
        )
        if writeparticles:
            np.savetxt(
                f"{location}/particles{i}_d.csv",
                particles,
                delimiter=",",
                header="x,y,type,bias",
                comments="",
            )

        # Predict next click for each particle
        click_probabilities = getClickProbabilities(
            lognormpdf, particles, dots_norm, location_sigma
        )

        # Weight particles by evidence
        weights = click_probabilities[..., session_clicks[i]]
        # Resample particles
        ind = np.random.choice(
            num_particles, num_particles, p=weights / np.sum(weights)
        )
        particles = particles[ind, ...]
        if writeparticles:
            np.savetxt(
                f"{location}/particles{i+1}.csv",
                particles,
                delimiter=",",
                header="x,y,type,bias",
                comments="",
            )

        # Record result of click prediction
        marginal_click_probabilities = np.mean(click_probabilities, axis=0)
        if writeparticles:
            np.savetxt(
                f"{location}/dot_weights{i+1}.csv",
                marginal_click_probabilities,
                delimiter=",",
            )
        ind = np.argsort(-marginal_click_probabilities)
        if inds is None:
            inds = ind
        else:
            inds = np.vstack((inds, ind))
        if i > 0:
            # shayan: index of user click in the sorted array of points
            result[i - 1] = np.nonzero(ind == session_clicks[i])[0]

    # Reset the states
    setstate(rstate)
    np.random.set_state(nprstate)

    return result, inds


def getClickProbabilities(lognormpdf, particles, dots_norm, location_sigma):
    """
    returns a num_particles x num_dots matrix, giving the probability of the particles given the last click was each of the dots.
    :param lognormpdf:
    :param particles:
    :param dots_norm:
    :param location_sigma:
    :return:
    """
    # Predict next click for each particle

    # These have to be calculated separately before adding, I don't know why
    lnpdf0 = lognormpdf(particles[..., 0:1], dots_norm[..., 0:1].T, location_sigma)
    lnpdf1 = lognormpdf(particles[..., 1:2], dots_norm[..., 1:2].T, location_sigma)

    log_loc_prob = lnpdf0 + lnpdf1
    log_loc_prob -= np.max(log_loc_prob, axis=1)[..., np.newaxis]

    loc_prob = np.exp(log_loc_prob)
    loc_prob = loc_prob / np.sum(loc_prob, axis=1)[..., np.newaxis]

    type_prob = dots_norm[..., 2:3].T == particles[..., 2:3]
    type_prob = type_prob / np.sum(type_prob, axis=1)[..., np.newaxis]

    # click_probabilities[j, ...] = particles[j, 3] * loc_prob + (1 - particles[j, 3]) * type_prob
    return particles[..., 3:4] * loc_prob + (1 - particles[..., 3:4]) * type_prob


def diffuseParticles(
    particles,
    location_d,
    mixture_d,
    num_particles,
    flip_p,
    num_types,
    type_counts,
    num_dots,
    eps,
):
    # Diffusion on locations
    particles[..., 0:2] = particles[..., 0:2] + location_d * np.random.normal(
        size=particles[..., 0:2].shape
    )

    # Diffusion on mixture weights
    particles[..., 3] = particles[..., 3] + mixture_d * np.random.normal(
        size=particles[..., 3].shape
    )

    # "Discrete diffusion" on types
    flip_ind = np.random.random(num_particles) < flip_p
    particles[flip_ind, 2] = np.random.choice(
        np.arange(1, num_types + 1),
        size=np.count_nonzero(flip_ind),
        p=type_counts / num_dots,
    )

    # Clip values to boundary
    particles[..., (0, 1, 3)] = np.clip(particles[..., (0, 1, 3)], eps, 1 - eps)


def setRandomState():
    rstate = getstate()
    seed(0)
    nprstate = np.random.get_state()
    np.random.seed(0)
    return rstate, nprstate
