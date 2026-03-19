# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

# -----------------------
# [C/Python] dependencies
# -----------------------
from libc.math cimport log, exp, pow, isnan
from libc.stdint cimport uint64_t

cimport cython
from cython.parallel import prange
from cython.parallel cimport threadid
cimport numpy as cnp

import numpy as np

# --------------------
# [C] LCG random number generator (per-thread, nogil-safe)
# --------------------

cdef uint64_t lcg_rand(uint64_t state) nogil:
    """Advance LCG state and return new state (PCG-style multiplier)."""
    return state * 6364136223846793005UL + 1442695040888963407UL


# --------------------
# [C] CSR user-item membership check
# --------------------

cdef bint item_in_user(int u, int j, int[::1] ui_ptr, int[::1] ui_idx) nogil:
    """Check if item j is in user u's observed items (CSR format, linear scan)."""
    cdef int k
    for k in range(ui_ptr[u], ui_ptr[u + 1]):
        if ui_idx[k] == j:
            return True
    return False


# --------------------
# [C] FM scoring helper
# --------------------

cdef float compute_ui_utility(
    int F,
    int P,
    int Q,
    float[::1] x_uf,
    float[::1] x_if,
    float w_i,
    float[::1] w_if,
    float[::1] v_u,
    float[::1] v_i,
    float[:, ::1] v_uf,
    float[:, ::1] v_if,
    int x_uf_any,
    int x_if_any
) nogil:

    cdef int f, p, q
    cdef float res = w_i

    for f in range(F):
        # user * item: np.dot(v_u[u], v_i[i])
        res += v_u[f] * v_i[f]

    if x_uf_any:
        for p in range(P):
            if x_uf[p] == 0.0:
                continue
            for f in range(F):
                # user-features * item: np.dot(x_uf[u], np.dot(v_uf, v_i[i]))
                res += x_uf[p] * (v_uf[p, f] * v_i[f])

    if x_if_any:
        for q in range(Q):
            if x_if[q] == 0.0:
                continue
            # item-features: np.dot(x_if[i], w_if)
            res += x_if[q] * w_if[q]
            for f in range(F):
                # item-features * user: np.dot(x_if[i], np.dot(v_if, v_u[u]))
                res += x_if[q] * (v_if[q, f] * v_u[f])

    return res

# -------------------------
# [Python] helper functions
# -------------------------

def assert_finite(w_i, w_if, v_u, v_i, v_uf, v_if):
    """assert all model weights are finite"""

    assert np.isfinite(np.sum(w_i)), "item weights [w_i] are not finite - try decreasing feature/sample_weight magnitudes"
    assert np.isfinite(np.sum(w_if)), "item feature weights [w_if] are not finite - try decreasing feature/sample_weight magnitudes"
    assert np.isfinite(np.sum(v_u)), "user factors [v_u] are not finite - try decreasing feature/sample_weight magnitudes"
    assert np.isfinite(np.sum(v_i)), "item factors [v_i] are not finite - try decreasing feature/sample_weight magnitudes"
    assert np.isfinite(np.sum(v_uf)), "user-feature factors [v_uf] are not finite - try decreasing feature/sample_weight magnitudes"
    assert np.isfinite(np.sum(v_if)), "item-feature factors [v_if] are not finite - try decreasing feature/sample_weight magnitudes"


def reg_penalty(alpha, beta, w_i, w_if, v_u, v_i, v_uf, v_if):
    """calculate the total regularization penalty for all model weights"""

    penalty = 0.0
    penalty += np.sum(alpha * np.square(w_i))
    penalty += np.sum(alpha * np.square(v_u))
    penalty += np.sum(alpha * np.square(v_i))
    penalty += np.sum(beta * np.square(w_if))
    penalty += np.sum(beta * np.square(v_uf))
    penalty += np.sum(beta * np.square(v_if))
    return penalty

# --------------------------------
# [RankFM] core modeling functions
# --------------------------------

def _fit(
    int[:, ::1] interactions,
    float[::1] sample_weight,
    int[::1] ui_ptr,           # CSR: user-items row pointers [U+1]
    int[::1] ui_idx,           # CSR: user-items column indices [total_interactions]
    float[:, ::1] x_uf,
    float[:, ::1] x_if,
    float[::1] w_i,
    float[::1] w_if,
    float[:, ::1] v_u,
    float[:, ::1] v_i,
    float[:, ::1] v_uf,
    float[:, ::1] v_if,
    float alpha,
    float beta,
    float learning_rate,
    str learning_schedule,
    float learning_exponent,
    int max_samples,
    int epochs,
    bint verbose,
    int n_threads,
):
    """
    Hogwild!-style parallel SGD training.

    Training samples within each epoch are processed in parallel across n_threads
    threads using OpenMP. Weight updates are applied without locks (Hogwild!), which
    is theoretically sound for sparse problems and works well in practice.
    """

    #############################
    ### VARIABLE DECLARATIONS ###
    #############################

    # constants
    cdef float MARGIN = 1.0

    # matrix shapes/indicators
    cdef int N, U, I, F, P, Q
    cdef int x_uf_any, x_if_any

    # loop iterators/indices
    cdef int r, u, i, j, f, p, q
    cdef int epoch, row, sampled, tid

    # epoch-specific learning rate
    cdef float eta

    # sample weights and (ui, uj) utility scores
    cdef float sw, ut_ui, ut_uj

    # WARP sampling variables
    cdef int min_index
    cdef float pairwise_utility, min_pairwise_utility, multiplier

    # loss function derivatives wrt model weights
    cdef float d_outer
    cdef float d_reg_a = 2.0 * alpha
    cdef float d_reg_b = 2.0 * beta
    cdef float d_w_i = 1.0
    cdef float d_w_j = -1.0
    cdef float d_w_if, d_v_i, d_v_j, d_v_u, d_v_uf, d_v_if

    # per-thread LCG RNG state
    cdef uint64_t rng_state

    #######################################
    ### PYTHON SET-UP PRIOR TO TRAINING ###
    #######################################

    # calculate matrix shapes
    N = interactions.shape[0]
    U = v_u.shape[0]
    I = v_i.shape[0]
    P = v_uf.shape[0]
    Q = v_if.shape[0]
    F = v_u.shape[1]

    # determine whether any user-features/item-features were supplied
    x_uf_any = int(np.asarray(x_uf).any())
    x_if_any = int(np.asarray(x_if).any())

    # shuffle index for each epoch
    shuffle_index = np.arange(N, dtype=np.int32)
    cdef int[::1] shuffle_index_mv = shuffle_index

    # per-thread RNG seeds (one per thread, initialized with different values)
    seeds = (np.arange(1, n_threads + 1, dtype=np.uint64) * 6364136223846793005)
    cdef uint64_t[::1] seeds_mv = seeds

    # per-thread log-likelihood accumulators (avoid race conditions on sum)
    ll_thread = np.zeros(n_threads, dtype=np.float64)
    cdef double[::1] ll_thread_mv = ll_thread

    ################################
    ### MAIN TRAINING EPOCH LOOP ###
    ################################

    for epoch in range(epochs):

        if learning_schedule == 'constant':
            eta = learning_rate
        elif learning_schedule == 'invscaling':
            eta = learning_rate / pow(epoch + 1, learning_exponent)
        else:
            raise ValueError('unknown [learning_schedule]')

        np.random.shuffle(shuffle_index)
        ll_thread[:] = 0.0

        # Hogwild! parallel SGD: process training samples concurrently.
        # Weight updates are racy but theoretically convergent for sparse problems.
        for r in prange(N, schedule='dynamic', nogil=True, num_threads=n_threads):

            # identify current thread and get this sample's training row
            tid = threadid()
            row = shuffle_index_mv[r]
            u = interactions[row, 0]
            i = interactions[row, 1]
            sw = sample_weight[row]

            # compute the utility score of the observed (u, i) pair
            ut_ui = compute_ui_utility(F, P, Q, x_uf[u], x_if[i], w_i[i], w_if, v_u[u], v_i[i], v_uf, v_if, x_uf_any, x_if_any)

            # WARP/BPR negative sampling loop for the (u, i) pair
            # ----------------------------------------------------------
            # Use per-thread LCG RNG instead of the global MT RNG so
            # this block is safe to run concurrently without locks.

            min_index = -1
            min_pairwise_utility = 1e6

            for sampled in range(1, max_samples + 1):

                # sample an unobserved item using this thread's LCG RNG
                while True:
                    seeds_mv[tid] = lcg_rand(seeds_mv[tid])
                    j = <int>(seeds_mv[tid] % <uint64_t>I)
                    if not item_in_user(u, j, ui_ptr, ui_idx):
                        break

                # compute utility of the unobserved (u, j) pair
                ut_uj = compute_ui_utility(F, P, Q, x_uf[u], x_if[j], w_i[j], w_if, v_u[u], v_i[j], v_uf, v_if, x_uf_any, x_if_any)
                pairwise_utility = ut_ui - ut_uj

                if pairwise_utility < min_pairwise_utility:
                    min_index = j
                    min_pairwise_utility = pairwise_utility

                if pairwise_utility < MARGIN:
                    break

            # finalize negative item and loss multiplier
            j = min_index
            pairwise_utility = min_pairwise_utility
            multiplier = log((I - 1) / sampled) / log(I)

            # accumulate per-thread log-likelihood (thread-safe: each thread has its own slot)
            ll_thread_mv[tid] += log(1.0 / (1.0 + exp(-pairwise_utility)))

            # gradient step model weight updates (Hogwild!: racy writes to shared arrays)
            # ---------------------------------------------------------------------------

            # outer derivative [d_LL / d_g(pu)]
            d_outer = 1.0 / (exp(pairwise_utility) + 1.0)

            # update [item] scalar weights
            w_i[i] += eta * (sw * multiplier * (d_outer * d_w_i) - (d_reg_a * w_i[i]))
            w_i[j] += eta * (sw * multiplier * (d_outer * d_w_j) - (d_reg_a * w_i[j]))

            # update [item-feature] scalar weights
            if x_if_any:
                for q in range(Q):
                    d_w_if = x_if[i, q] - x_if[j, q]
                    w_if[q] += eta * (sw * multiplier * (d_outer * d_w_if) - (d_reg_b * w_if[q]))

            # update all [factor] weights
            for f in range(F):

                # base derivatives wrt [user-factors] and [item-factors]
                d_v_u = v_i[i, f] - v_i[j, f]
                d_v_i = v_u[u, f]
                d_v_j = -v_u[u, f]

                # add [user-features] contribution to [item-factor] derivatives
                if x_uf_any:
                    for p in range(P):
                        d_v_i += v_uf[p, f] * x_uf[u, p]
                        d_v_j -= v_uf[p, f] * x_uf[u, p]

                # add [item-features] contribution to [user-factor] derivatives
                if x_if_any:
                    for q in range(Q):
                        d_v_u += v_if[q, f] * (x_if[i, q] - x_if[j, q])

                # update [user-factor] and [item-factor] weights
                v_u[u, f] += eta * (sw * multiplier * (d_outer * d_v_u) - (d_reg_a * v_u[u, f]))
                v_i[i, f] += eta * (sw * multiplier * (d_outer * d_v_i) - (d_reg_a * v_i[i, f]))
                v_i[j, f] += eta * (sw * multiplier * (d_outer * d_v_j) - (d_reg_a * v_i[j, f]))

                # update [user-feature-factor] weights
                if x_uf_any:
                    for p in range(P):
                        if x_uf[u, p] == 0.0:
                            continue
                        d_v_uf = x_uf[u, p] * (v_i[i, f] - v_i[j, f])
                        v_uf[p, f] += eta * (sw * multiplier * (d_outer * d_v_uf) - (d_reg_b * v_uf[p, f]))

                # update [item-feature-factor] weights
                if x_if_any:
                    for q in range(Q):
                        if x_if[i, q] - x_if[j, q] == 0.0:
                            continue
                        d_v_if = (x_if[i, q] - x_if[j, q]) * v_u[u, f]
                        v_if[q, f] += eta * (sw * multiplier * (d_outer * d_v_if) - (d_reg_b * v_if[q, f]))

        # [end epoch]: assert all model weights are finite
        assert_finite(w_i, w_if, v_u, v_i, v_uf, v_if)

        # report the penalized log-likelihood for this training epoch
        if verbose:
            log_likelihood = float(np.sum(ll_thread))
            penalty = reg_penalty(alpha, beta, w_i, w_if, v_u, v_i, v_uf, v_if)
            log_likelihood = round(log_likelihood - penalty, 2)
            print("\ntraining epoch:", epoch)
            print("log likelihood:", log_likelihood)


def _predict(
    float[:, ::1] pairs,
    float[:, ::1] x_uf,
    float[:, ::1] x_if,
    float[::1] w_i,
    float[::1] w_if,
    float[:, ::1] v_u,
    float[:, ::1] v_i,
    float[:, ::1] v_uf,
    float[:, ::1] v_if,
    int n_threads,
):
    """Score user-item pairs in parallel using n_threads OpenMP threads."""

    cdef int N = pairs.shape[0]
    cdef int P = v_uf.shape[0]
    cdef int Q = v_if.shape[0]
    cdef int F = v_u.shape[1]
    cdef int x_uf_any = int(np.asarray(x_uf).any())
    cdef int x_if_any = int(np.asarray(x_if).any())

    cdef int row, u, i
    cdef float u_flt, i_flt
    cdef float nan_value = float('nan')

    scores_np = np.empty(N, dtype=np.float32)
    cdef float[::1] scores = scores_np

    for row in prange(N, nogil=True, num_threads=n_threads):
        u_flt = pairs[row, 0]
        i_flt = pairs[row, 1]

        if isnan(u_flt) or isnan(i_flt):
            scores[row] = nan_value
        else:
            u = <int>u_flt
            i = <int>i_flt
            scores[row] = compute_ui_utility(
                F, P, Q,
                x_uf[u], x_if[i],
                w_i[i], w_if,
                v_u[u], v_i[i],
                v_uf, v_if,
                x_uf_any, x_if_any
            )

    return scores_np


def _recommend(
    float[::1] users,
    dict user_items,
    int n_items,
    bint filter_previous,
    float[:, ::1] x_uf,
    float[:, ::1] x_if,
    float[::1] w_i,
    float[::1] w_if,
    float[:, ::1] v_u,
    float[:, ::1] v_i,
    float[:, ::1] v_uf,
    float[:, ::1] v_if,
    int n_threads,
):
    """
    Score all items for a batch of users in parallel, then select top-N.

    Parallelism is at the user level (outer prange over users), with each thread
    computing scores for all items of its assigned user sequentially.
    The 2D score buffer [U x I] allows each thread to write to its own row
    with no synchronization needed.

    Post-processing (argsort, filter, top-N selection) is done sequentially
    since it involves Python objects (numpy, dict).
    """

    # declare variables
    cdef int U, I, P, Q, F
    cdef int x_uf_any, x_if_any
    cdef int row, u, i, s
    cdef float u_flt

    # calculate matrix shapes
    U = users.shape[0]
    I = w_i.shape[0]
    P = v_uf.shape[0]
    Q = v_if.shape[0]
    F = v_u.shape[1]

    # determine whether any user-features/item-features were supplied
    x_uf_any = int(np.asarray(x_uf).any())
    x_if_any = int(np.asarray(x_if).any())

    # output recommendations matrix [U x n_items]
    rec_items = np.empty((U, n_items), dtype=np.float32)

    # [U x I] score buffer: each user gets its own row, enabling lock-free parallel writes
    all_scores = np.empty((U, I), dtype=np.float32)
    cdef float[:, ::1] all_scores_mv = all_scores

    # parallel scoring: outer loop over users, inner loop over items per user
    for row in prange(U, schedule='dynamic', nogil=True, num_threads=n_threads):
        u_flt = users[row]
        if not isnan(u_flt):
            u = <int>u_flt
            for i in range(I):
                all_scores_mv[row, i] = compute_ui_utility(
                    F, P, Q,
                    x_uf[u], x_if[i],
                    w_i[i], w_if,
                    v_u[u], v_i[i],
                    v_uf, v_if,
                    x_uf_any, x_if_any
                )

    # sequential post-processing: sort and select top-N for each user
    for row in range(U):
        u_flt = users[row]
        if isnan(u_flt):
            # unknown user: return NaN recommendations
            rec_items[row] = np.full(n_items, np.nan, dtype=np.float32)
        else:
            u = <int>u_flt
            ranked_items = np.argsort(all_scores[row])[::-1]
            selected_items = np.empty(n_items, dtype=np.float32)

            # select top-N items, optionally filtering previously observed items
            s = 0
            for i in range(I):
                if filter_previous and ranked_items[i] in user_items[u]:
                    continue
                selected_items[s] = ranked_items[i]
                s += 1
                if s == n_items:
                    break

            rec_items[row] = selected_items

    return rec_items
