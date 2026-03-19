"""
rankfm main modeling class
"""

import numpy as np
import pandas as pd

from rankfmc._rankfm import _fit, _predict, _recommend
from rankfmc.utils import get_data


class RankFM:
    """Factorization Machines for Ranking Problems with Implicit Feedback Data"""

    def __init__(
        self,
        factors=10,
        loss='bpr',
        max_samples=10,
        alpha=0.01,
        beta=0.1,
        sigma=0.1,
        learning_rate=0.1,
        learning_schedule='constant',
        learning_exponent=0.25,
        n_jobs=1,
    ):
        """store hyperparameters and initialize internal model state

        :param factors: latent factor rank
        :param loss: optimization/loss function to use for training: ['bpr', 'warp']
        :param max_samples: maximum number of negative samples to draw for WARP loss
        :param alpha: L2 regularization penalty on [user, item] model weights
        :param beta: L2 regularization penalty on [user-feature, item-feature] model weights
        :param sigma: standard deviation to use for random initialization of factor weights
        :param learning_rate: initial learning rate for gradient step updates
        :param learning_schedule: schedule for adjusting learning rates by training epoch: ['constant', 'invscaling']
        :param learning_exponent: exponent applied to epoch number to adjust learning rate: scaling = 1 / pow(epoch + 1, learning_exponent)
        :param n_jobs: number of parallel OpenMP threads for training, prediction, and recommendation (default=1)
        :return: None
        """

        # validate user input
        assert isinstance(factors, int) and factors >= 1, "[factors] must be a positive integer"
        assert isinstance(loss, str) and loss in ('bpr', 'warp'), "[loss] must be in ('bpr', 'warp')"
        assert isinstance(max_samples, int) and max_samples > 0, "[max_samples] must be a positive integer"
        assert isinstance(alpha, float) and alpha > 0.0, "[alpha] must be a positive float"
        assert isinstance(beta, float) and beta > 0.0, "[beta] must be a positive float"
        assert isinstance(sigma, float) and sigma > 0.0, "[sigma] must be a positive float"
        assert isinstance(learning_rate, float) and learning_rate > 0.0, "[learning_rate] must be a positive float"
        assert isinstance(learning_schedule, str) and learning_schedule in ('constant', 'invscaling'), "[learning_schedule] must be in ('constant', 'invscaling')"
        assert isinstance(learning_exponent, float) and learning_exponent > 0.0, "[learning_exponent] must be a positive float"
        assert isinstance(n_jobs, int) and n_jobs >= 1, "[n_jobs] must be a positive integer"

        # store model hyperparameters
        self.factors = factors
        self.loss = loss
        self.max_samples = max_samples
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.learning_schedule = learning_schedule
        self.learning_exponent = learning_exponent
        self.n_jobs = n_jobs

        # set/clear initial model state
        self._reset_state()


    # --------------------------------
    # begin private method definitions
    # --------------------------------


    def _reset_state(self):
        """initialize or reset internal model state"""

        # [ID, IDX] arrays
        self.user_id = None
        self.item_id = None
        self.user_idx = None
        self.item_idx = None

        # [ID <-> IDX] mappings
        self.index_to_user = None
        self.index_to_item = None
        self.user_to_index = None
        self.item_to_index = None

        # set of observed items for each user
        self.user_items = None

        # [user, item] features
        self.x_uf = None
        self.x_if = None

        # [item, item-feature] scalar weights
        self.w_i = None
        self.w_if = None

        # [user, item, user-feature, item-feature] latent factors
        self.v_u = None
        self.v_i = None
        self.v_uf = None
        self.v_if = None

        # internal model state indicator
        self.is_fit = False


    def _init_all(self, interactions, user_features=None, item_features=None, sample_weight=None):
        """index the interaction data and user/item features and initialize model weights

        :param interactions: dataframe of observed user/item interactions: [user_id, item_id]
        :param user_features: dataframe of user metadata features: [user_id, uf_1, ..., uf_n]
        :param item_features: dataframe of item metadata features: [item_id, if_1, ..., if_n]
        :param sample_weight: vector of importance weights for each observed interaction
        :return: None
        """

        assert isinstance(interactions, (np.ndarray, pd.DataFrame)), "[interactions] must be np.ndarray or pd.dataframe"
        assert interactions.shape[1] == 2, "[interactions] should be: [user_id, item_id]"

        # save unique arrays of users/items in terms of original identifiers
        interactions_df = pd.DataFrame(get_data(interactions), columns=['user_id', 'item_id'])
        self.user_id = pd.Series(np.sort(np.unique(interactions_df['user_id'])))
        self.item_id = pd.Series(np.sort(np.unique(interactions_df['item_id'])))

        # create zero-based index to identifier mappings
        self.index_to_user = self.user_id
        self.index_to_item = self.item_id

        # create reverse mappings from identifiers to zero-based index positions
        self.user_to_index = pd.Series(data=self.index_to_user.index, index=self.index_to_user.values)
        self.item_to_index = pd.Series(data=self.index_to_item.index, index=self.index_to_item.values)

        # store unique values of user/item indexes and observed interactions for each user
        self.user_idx = np.arange(len(self.user_id), dtype=np.int32)
        self.item_idx = np.arange(len(self.item_id), dtype=np.int32)

        # map the interactions to internal index positions
        interactions, sample_weight = self._init_interactions(interactions, sample_weight)

        # map the user/item features to internal index positions
        self._init_features(user_features, item_features)

        # initialize the model weights after the user/item/feature dimensions have been established
        self._init_weights(user_features, item_features)
        return interactions, sample_weight


    def _init_interactions(self, interactions, sample_weight):
        """map new interaction data to existing internal user/item indexes

        :param interactions: dataframe of observed user/item interactions: [user_id, item_id]
        :param sample_weight: vector of importance weights for each observed interaction
        :return: None
        """

        assert isinstance(interactions, (np.ndarray, pd.DataFrame)), "[interactions] must be np.ndarray or pd.dataframe"
        assert interactions.shape[1] == 2, "[interactions] should be: [user_id, item_id]"

        # map the raw user/item identifiers to internal zero-based index positions
        # NOTE: any user/item pairs not found in the existing indexes will be dropped
        interactions = pd.DataFrame(get_data(interactions).copy(), columns=['user_id', 'item_id'])
        interactions['user_id'] = interactions['user_id'].map(self.user_to_index).astype(np.int32)
        interactions['item_id'] = interactions['item_id'].map(self.item_to_index).astype(np.int32)
        interactions = interactions.rename({'user_id': 'user_idx', 'item_id': 'item_idx'}, axis=1).dropna()

        # store the sample weights internally or generate a vector of ones if not given
        if sample_weight is not None:
            assert isinstance(sample_weight, (np.ndarray, pd.Series)), "[sample_weight] must be np.ndarray or pd.series"
            assert sample_weight.ndim == 1, "[sample_weight] must a vector (ndim=1)"
            assert len(sample_weight) == len(interactions), "[sample_weight] must have the same length as [interactions]"
            sample_weight = np.ascontiguousarray(get_data(sample_weight), dtype=np.float32)
        else:
            sample_weight = np.ones(len(interactions), dtype=np.float32)

        # create a dictionary containing the set of observed items for each user
        # NOTE: if the model has been previously fit extend rather than replace the itemset for each user

        if self.is_fit:
            new_user_items = interactions.groupby('user_idx')['item_idx'].apply(set).to_dict()
            self.user_items = {user: np.array(list(set(self.user_items[user]) | set(new_user_items[user])), dtype=np.int32) for user in self.user_items.keys()}
        else:
            self.user_items = interactions.groupby('user_idx')['item_idx'].unique().apply(np.array, dtype=np.int32).to_dict()

        # format the interactions data as a c-contiguous integer array for cython use
        interactions = np.ascontiguousarray(interactions, dtype=np.int32)
        return interactions, sample_weight



    def _init_features(self, user_features=None, item_features=None):
        """initialize the user/item features given existing internal user/item indexes

        :param user_features: dataframe of user metadata features: [user_id, uf_1, ... , uf_n]
        :param item_features: dataframe of item metadata features: [item_id, if_1, ... , if_n]
        :return: None
        """

        # store the user features as a ndarray [UxP] row-ordered by user index position
        if user_features is not None:
            x_uf = pd.DataFrame(user_features.copy())
            x_uf = x_uf.set_index(x_uf.columns[0])
            x_uf.index = x_uf.index.map(self.user_to_index)
            if np.array_equal(sorted(x_uf.index.values), self.user_idx):
                self.x_uf = np.ascontiguousarray(x_uf.sort_index(), dtype=np.float32)
            else:
                raise KeyError('the users in [user_features] do not match the users in [interactions], user id may be missing or duplicated in [user_features]')
        else:
            self.x_uf = np.zeros([len(self.user_idx), 1], dtype=np.float32)

        # store the item features as a ndarray [IxQ] row-ordered by item index position
        if item_features is not None:
            x_if = pd.DataFrame(item_features.copy())
            x_if = x_if.set_index(x_if.columns[0])
            x_if.index = x_if.index.map(self.item_to_index)
            if np.array_equal(sorted(x_if.index.values), self.item_idx):
                self.x_if = np.ascontiguousarray(x_if.sort_index(), dtype=np.float32)
            else:
                raise KeyError('the items in [item_features] do not match the items in [interactions], item id may be missing or duplicated in [item_features]')
        else:
            self.x_if = np.zeros([len(self.item_idx), 1], dtype=np.float32)


    def _init_weights(self, user_features=None, item_features=None):
        """initialize model weights given user/item and user-feature/item-feature indexes/shapes

        :param user_features: dataframe of user metadata features: [user_id, uf_1, ... , uf_n]
        :param item_features: dataframe of item metadata features: [item_id, if_1, ... , if_n]
        :return: None
        """

        # initialize scalar weights as ndarrays of zeros
        self.w_i = np.zeros(len(self.item_idx)).astype(np.float32)
        self.w_if = np.zeros(self.x_if.shape[1]).astype(np.float32)

        # initialize latent factors by drawing random samples from a normal distribution
        self.v_u = np.random.normal(loc=0, scale=self.sigma, size=(len(self.user_idx), self.factors)).astype(np.float32)
        self.v_i = np.random.normal(loc=0, scale=self.sigma, size=(len(self.item_idx), self.factors)).astype(np.float32)

        # randomly initialize user feature factors if user features were supplied
        # NOTE: set all user feature factor weights to zero to prevent random scoring influence otherwise
        if user_features is not None:
            scale = (self.alpha / self.beta) * self.sigma
            self.v_uf = np.random.normal(loc=0, scale=scale, size=[self.x_uf.shape[1], self.factors]).astype(np.float32)
        else:
            self.v_uf = np.zeros([self.x_uf.shape[1], self.factors], dtype=np.float32)

        # randomly initialize item feature factors if item features were supplied
        # NOTE: set all item feature factor weights to zero to prevent random scoring influence otherwise
        if item_features is not None:
            scale = (self.alpha / self.beta) * self.sigma
            self.v_if = np.random.normal(loc=0, scale=scale, size=[self.x_if.shape[1], self.factors]).astype(np.float32)
        else:
            self.v_if = np.zeros([self.x_if.shape[1], self.factors], dtype=np.float32)


    def _build_ui_csr(self):
        """Convert user_items dict to CSR (Compressed Sparse Row) format.

        The resulting arrays are used by the Cython training loop to perform
        nogil-compatible membership checks during negative sampling.

        :return: (ui_ptr, ui_idx) — CSR row pointers and column indices as int32 arrays
        """
        n_users = len(self.user_idx)
        ptr = np.zeros(n_users + 1, dtype=np.int32)

        for u in range(n_users):
            n = len(self.user_items[u]) if u in self.user_items else 0
            ptr[u + 1] = ptr[u] + n

        total = int(ptr[n_users])
        idx = np.empty(total, dtype=np.int32)

        for u in range(n_users):
            start = int(ptr[u])
            end = int(ptr[u + 1])
            if start < end:
                idx[start:end] = self.user_items[u]

        return (
            np.ascontiguousarray(ptr, dtype=np.int32),
            np.ascontiguousarray(idx, dtype=np.int32),
        )


    # -------------------------------
    # begin public method definitions
    # -------------------------------


    def fit(self, interactions, user_features=None, item_features=None, sample_weight=None, epochs=1, verbose=False):
        """clear previous model state and learn new model weights using the input data

        :param interactions: dataframe of observed user/item interactions: [user_id, item_id]
        :param user_features: dataframe of user metadata features: [user_id, uf_1, ... , uf_n]
        :param item_features: dataframe of item metadata features: [item_id, if_1, ... , if_n]
        :param sample_weight: vector of importance weights for each observed interaction
        :param epochs: number of training epochs (full passes through observed interactions)
        :param verbose: whether to print epoch number and log-likelihood during training
        :return: self
        """

        self._reset_state()
        self.fit_partial(interactions, user_features, item_features, sample_weight, epochs, verbose)
        return self


    def fit_partial(self, interactions, user_features=None, item_features=None, sample_weight=None, epochs=1, verbose=False):
        """learn or update model weights using the input data and resuming from the current model state

        :param interactions: dataframe of observed user/item interactions: [user_id, item_id]
        :param user_features: dataframe of user metadata features: [user_id, uf_1, ... , uf_n]
        :param item_features: dataframe of item metadata features: [item_id, if_1, ... , if_n]
        :param sample_weight: vector of importance weights for each observed interaction
        :param epochs: number of training epochs (full passes through observed interactions)
        :param verbose: whether to print epoch number and log-likelihood during training
        :return: self
        """

        assert isinstance(epochs, int) and epochs >= 1, "[epochs] must be a positive integer"
        assert isinstance(verbose, bool), "[verbose] must be a boolean value"

        if self.is_fit:
            interactions, sample_weight = self._init_interactions(interactions, sample_weight)
            self._init_features(user_features, item_features)
        else:
            interactions, sample_weight = self._init_all(interactions, user_features, item_features, sample_weight)

        # determine the number of negative samples to draw depending on the loss function
        # NOTE: if [loss == 'bpr'] -> [max_samples == 1] and [multiplier ~= 1] for all updates
        # NOTE: the [multiplier] is scaled by total number of items so it's always [0, 1]

        if self.loss == 'bpr':
            max_samples = 1
        elif self.loss == 'warp':
            max_samples = self.max_samples
        else:
            raise ValueError('[loss] function not recognized')

        # build CSR representation of user_items for nogil negative sampling in _fit
        ui_ptr, ui_idx = self._build_ui_csr()

        # NOTE: the cython private _fit() method updates the model weights in-place via typed memoryviews
        # NOTE: therefore there's nothing returned explicitly by either method

        _fit(
            interactions,
            sample_weight,
            ui_ptr,
            ui_idx,
            self.x_uf,
            self.x_if,
            self.w_i,
            self.w_if,
            self.v_u,
            self.v_i,
            self.v_uf,
            self.v_if,
            self.alpha,
            self.beta,
            self.learning_rate,
            self.learning_schedule,
            self.learning_exponent,
            max_samples,
            epochs,
            verbose,
            self.n_jobs,
        )

        self.is_fit = True
        return self


    def predict(self, pairs, cold_start='nan'):
        """calculate the predicted pointwise utilities for all (user, item) pairs

        :param pairs: dataframe of [user, item] pairs to score
        :param cold_start: whether to generate missing values ('nan') or drop ('drop') user/item pairs not found in training data
        :return: np.array of real-valued model scores
        """

        assert isinstance(pairs, (np.ndarray, pd.DataFrame)), "[pairs] must be np.ndarray or pd.dataframe"
        assert pairs.shape[1] == 2, "[pairs] should be: [user_id, item_id]"
        assert self.is_fit, "you must fit the model prior to generating predictions"

        pred_pairs = pd.DataFrame(get_data(pairs).copy(), columns=['user_id', 'item_id'])
        pred_pairs['user_id'] = pred_pairs['user_id'].map(self.user_to_index)
        pred_pairs['item_id'] = pred_pairs['item_id'].map(self.item_to_index)
        pred_pairs = np.ascontiguousarray(pred_pairs, dtype=np.float32)

        scores = _predict(
            pred_pairs,
            self.x_uf,
            self.x_if,
            self.w_i,
            self.w_if,
            self.v_u,
            self.v_i,
            self.v_uf,
            self.v_if,
            self.n_jobs,
        )

        if cold_start == 'nan':
            return scores
        elif cold_start == 'drop':
            return scores[~np.isnan(scores)]
        else:
            raise ValueError("param [cold_start] must be set to either 'nan' or 'drop'")


    def _score_batch(self, user_indices):
        """Score all items for a batch of users using vectorized BLAS matrix multiplication.

        Uses numpy's highly-optimized BLAS matmul (OpenBLAS / MKL) rather than a manual
        Cython loop, giving much better throughput through SIMD and multi-threaded BLAS.

        :param user_indices: int32 array of zero-based user indices, shape [B]
        :return: float32 score matrix of shape [B, I]
        """
        v_u_batch = self.v_u[user_indices]                  # [B, F]
        scores = v_u_batch @ self.v_i.T + self.w_i          # [B, I]  (w_i broadcast over B)

        if np.any(self.x_uf):
            uf_embed = self.x_uf[user_indices] @ self.v_uf  # [B, P] @ [P, F] = [B, F]
            scores += uf_embed @ self.v_i.T                  # [B, I]

        if np.any(self.x_if):
            scores += self.x_if @ self.w_if                  # [I] broadcast over B
            if_embed = self.x_if @ self.v_if                 # [I, F]
            scores += v_u_batch @ if_embed.T                 # [B, I]

        return scores.astype(np.float32)                     # ensure float32 output


    def recommend(self, users, n_items=10, filter_previous=False, cold_start='nan', batch_size=512):
        """calculate the topN items for each user

        Scoring is done via vectorized BLAS matrix multiplication (_score_batch) which
        is significantly faster than scalar Cython loops. Top-N selection uses
        np.argpartition (O(I)) instead of a full argsort (O(I log I)).

        :param users: iterable of user identifiers for which to generate recommendations
        :param n_items: number of recommended items to generate for each user
        :param filter_previous: remove observed training items from generated recommendations
        :param cold_start: whether to generate missing values ('nan') or drop ('drop') users not found in training data
        :param batch_size: number of users to score per batch (controls peak memory:
               batch_size × I × 4 bytes; default 512)
        :return: pandas dataframe where the index values are user identifiers and the columns are recommended items
        """

        assert getattr(users, '__iter__', False), "[users] must be an iterable (e.g. list, array, series)"
        assert self.is_fit, "you must fit the model prior to generating recommendations"
        assert isinstance(batch_size, int) and batch_size >= 1, "[batch_size] must be a positive integer"

        users_series = pd.Series(users)
        user_idx_float = np.asarray(users_series.map(self.user_to_index), dtype=np.float64)
        n_users = len(user_idx_float)
        n_all_items = len(self.item_idx)

        rec_float = np.full((n_users, n_items), np.nan, dtype=np.float32)

        for start in range(0, n_users, batch_size):
            end = min(start + batch_size, n_users)
            batch_float = user_idx_float[start:end]          # float64 with NaN for unknowns
            valid_mask = ~np.isnan(batch_float)
            valid_int = batch_float[valid_mask].astype(np.int32)

            if len(valid_int) == 0:
                continue

            # Vectorized scoring: [V, I] via BLAS matmul
            scores = self._score_batch(valid_int)            # [V, I] float32

            if not filter_previous:
                # Fast top-N: argpartition O(n_all_items) + small sort O(n log n)
                if n_all_items <= n_items:
                    top_n = np.argsort(-scores, axis=1)
                else:
                    top_n_unsorted = np.argpartition(-scores, n_items, axis=1)[:, :n_items]
                    top_n_scores   = np.take_along_axis(scores, top_n_unsorted, axis=1)
                    sort_order     = np.argsort(-top_n_scores, axis=1)
                    top_n = np.take_along_axis(top_n_unsorted, sort_order, axis=1)
                rec_float[start:end][valid_mask] = top_n.astype(np.float32)

            else:
                # Per-user loop required to filter previously seen items
                v = 0
                for b_idx in range(end - start):
                    if not valid_mask[b_idx]:
                        continue
                    u = int(valid_int[v])
                    user_scores = scores[v]               # [I]
                    v += 1

                    seen = self.user_items.get(u, np.array([], dtype=np.int32))
                    n_seen = len(seen)

                    # Fetch enough candidates to survive filtering
                    k = min(n_items + n_seen, n_all_items)
                    if k < n_all_items:
                        top_k = np.argpartition(-user_scores, k)[:k]
                        top_k = top_k[np.argsort(-user_scores[top_k])]
                    else:
                        top_k = np.argsort(-user_scores)

                    seen_set = set(seen.tolist())
                    selected = [idx for idx in top_k if idx not in seen_set][:n_items]
                    n_sel = len(selected)
                    rec_float[start + b_idx, :n_sel] = np.array(selected, dtype=np.float32)

        rec_items = pd.DataFrame(rec_float, index=users).apply(lambda c: c.map(self.index_to_item))

        if cold_start == 'nan':
            return rec_items
        elif cold_start == 'drop':
            return rec_items.dropna(how='any')
        else:
            raise ValueError("param [cold_start] must be set to either 'nan' or 'drop'")


    def similar_items(self, item_id, n_items=10):
        """find the most similar items wrt latent factor space representation

        :param item_id: item to search
        :param n_items: number of similar items to return
        :return: np.array of topN most similar items wrt latent factor representations
        """

        assert item_id in self.item_id.values, "you must select an [item_id] present in the training data"
        assert self.is_fit, "you must fit the model prior to generating similarities"

        try:
            item_idx = self.item_to_index.loc[item_id]
        except (KeyError, TypeError):
            raise("item_id={} not found in training data".format(item_id))

        # calculate item latent representations in F dimensional factor space
        lr_item = self.v_i[item_idx] + np.dot(self.v_if.T, self.x_if[item_idx])
        lr_all_items = self.v_i + np.dot(self.x_if, self.v_if)

        # calculate the most similar N items excluding the search item
        similarities = pd.Series(np.dot(lr_all_items, lr_item)).drop(item_idx).sort_values(ascending=False)[:n_items]
        most_similar = pd.Series(similarities.index).map(self.index_to_item).values
        return most_similar


    def similar_users(self, user_id, n_users=10):
        """find the most similar users wrt latent factor space representation

        :param user_id: user to search
        :param n_users: number of similar users to return
        :return: np.array of topN most similar users wrt latent factor representations
        """

        assert user_id in self.user_id.values, "you must select an [user_id] present in the training data"
        assert self.is_fit, "you must fit the model prior to generating similarities"

        try:
            user_idx = self.user_to_index.loc[user_id]
        except (KeyError, TypeError):
            raise("user_id={} not found in training data".format(user_id))

        # calculate user latent representations in F dimensional factor space
        lr_user = self.v_u[user_idx] + np.dot(self.v_uf.T, self.x_uf[user_idx])
        lr_all_users = self.v_u + np.dot(self.x_uf, self.v_uf)

        # calculate the most similar N users excluding the search user
        similarities = pd.Series(np.dot(lr_all_users, lr_user)).drop(user_idx).sort_values(ascending=False)[:n_users]
        most_similar = pd.Series(similarities.index).map(self.index_to_user).values
        return most_similar
