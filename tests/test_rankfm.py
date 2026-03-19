"""
RankFM comprehensive unit tests.
Covers: constructor validation, fit, fit_partial, predict, recommend,
        similar_items, similar_users, and pre-fit guard checks.
"""

import pytest
import numpy as np
import pandas as pd

from rankfmc import RankFM


# =============================================================================
# Shared fixtures
# =============================================================================

@pytest.fixture
def intx_int():
    """Small integer interaction DataFrame (3 users x 6 items)."""
    return pd.DataFrame([
        (1, 1), (1, 3), (1, 5),
        (2, 1), (2, 2), (2, 6),
        (3, 3), (3, 6), (3, 4),
    ], columns=['user_id', 'item_id'], dtype=np.int32)


@pytest.fixture
def uf_good():
    """Valid user features (numeric, includes id column)."""
    return pd.DataFrame([
        (1, 0, 1, 5, 3.14),
        (2, 1, 0, 6, 2.72),
        (3, 0, 0, 4, 1.62),
    ], columns=['user_id', 'bin_1', 'bin_2', 'int', 'cnt'])


@pytest.fixture
def if_good():
    """Valid item features (numeric, includes id column)."""
    return pd.DataFrame([
        (1, 0, 1, 5, 3.14),
        (2, 1, 0, 6, 2.72),
        (3, 0, 0, 4, 1.62),
        (4, 1, 1, 3, 1.05),
        (5, 1, 0, 6, 0.33),
        (6, 0, 0, 0, 0.00),
    ], columns=['item_id', 'bin_1', 'bin_2', 'int', 'cnt'])


@pytest.fixture
def fitted_model(intx_int):
    """Pre-fitted RankFM model (factors=2, 2 epochs)."""
    model = RankFM(factors=2)
    model.fit(intx_int, epochs=2)
    return model


# --- module-level data (needed for @pytest.mark.parametrize) ---

_intx_pd_int = pd.DataFrame([
    (1, 1), (1, 3), (1, 5),
    (2, 1), (2, 2), (2, 6),
    (3, 3), (3, 6), (3, 4),
], columns=['user_id', 'item_id'], dtype=np.int32)

_intx_pd_str = pd.DataFrame([
    ('X', 'A'), ('X', 'C'), ('X', 'E'),
    ('Y', 'A'), ('Y', 'B'), ('Y', 'F'),
    ('Z', 'C'), ('Z', 'F'), ('Z', 'D'),
], columns=['user_id', 'item_id'])

_intx_np = np.array([
    (1, 1), (1, 3), (1, 5),
    (2, 1), (2, 2), (2, 6),
    (3, 3), (3, 6), (3, 4),
])

_uf_pd = pd.DataFrame([
    (1, 0, 1, 5, 3.14),
    (2, 1, 0, 6, 2.72),
    (3, 0, 0, 4, 1.62),
], columns=['user_id', 'bin_1', 'bin_2', 'int', 'cnt'])

_uf_np = np.array([
    (1, 0, 1, 5, 3.14),
    (2, 1, 0, 6, 2.72),
    (3, 0, 0, 4, 1.62),
])

_if_pd = pd.DataFrame([
    (1, 0, 1, 5, 3.14),
    (2, 1, 0, 6, 2.72),
    (3, 0, 0, 4, 1.62),
    (4, 1, 1, 3, 1.05),
    (5, 1, 0, 6, 0.33),
    (6, 0, 0, 0, 0.00),
], columns=['item_id', 'bin_1', 'bin_2', 'int', 'cnt'])

_if_np = np.array([
    (1, 0, 1, 5, 3.14),
    (2, 1, 0, 6, 2.72),
    (3, 0, 0, 4, 1.62),
    (4, 1, 1, 3, 1.05),
    (5, 1, 0, 6, 0.33),
    (6, 0, 0, 0, 0.00),
])

_train_users = np.array([1, 2, 3])
_valid_users = np.array([1, 2, 4, 5])

_intx_disjoint = pd.DataFrame([
    (1, 1), (1, 3), (1, 5),
    (2, 1), (2, 2), (2, 7),   # item 7 unknown
    (4, 3), (4, 7), (4, 4),   # user 4 unknown
], columns=['user_id', 'item_id'], dtype=np.int32)


# =============================================================================
# 1. Constructor / Hyperparameter validation
# =============================================================================

@pytest.mark.parametrize("kwargs", [
    {"factors": 0},
    {"factors": -1},
    {"loss": "unknown"},
    {"max_samples": 0},
    {"alpha": -0.1},
    {"alpha": 0.0},
    {"beta": -0.1},
    {"sigma": 0.0},
    {"learning_rate": 0.0},
    {"learning_schedule": "cyclic"},
    {"learning_exponent": 0.0},
    {"n_jobs": 0},
])
def test_constructor_invalid(kwargs):
    """Invalid hyperparameters should raise AssertionError."""
    with pytest.raises(AssertionError):
        RankFM(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {},
    {"factors": 10, "loss": "bpr"},
    {"factors": 10, "loss": "warp", "max_samples": 5},
    {"learning_schedule": "constant"},
    {"learning_schedule": "invscaling", "learning_exponent": 0.5},
    {"n_jobs": 2},
])
def test_constructor_valid(kwargs):
    """Valid hyperparameters should construct without error and not be fit."""
    model = RankFM(**kwargs)
    assert not model.is_fit


# =============================================================================
# 2. fit — good inputs
# =============================================================================

@pytest.mark.parametrize("interactions,user_features,item_features", [
    (_intx_pd_int, None,    None),     # integer IDs, pd.DataFrame
    (_intx_pd_str, None,    None),     # string IDs, pd.DataFrame
    (_intx_np,     None,    None),     # numpy array interactions
    (_intx_pd_int, _uf_pd,  None),     # with user features (DataFrame)
    (_intx_pd_int, None,    _if_pd),   # with item features (DataFrame)
    (_intx_pd_int, _uf_pd,  _if_pd),   # with both features (DataFrame)
    (_intx_pd_int, _uf_np,  _if_np),   # with both features (ndarray)
])
def test_fit_good(interactions, user_features, item_features):
    """Model fits successfully on all valid input format combinations."""
    model = RankFM(factors=2)
    model.fit(interactions, user_features, item_features, epochs=2, verbose=True)
    assert model.is_fit


@pytest.mark.parametrize("loss", ["bpr", "warp"])
def test_fit_both_loss_functions(intx_int, loss):
    """Both BPR and WARP loss functions complete training and index users/items."""
    model = RankFM(factors=2, loss=loss)
    model.fit(intx_int, epochs=3)
    assert model.is_fit
    assert len(model.user_id) == 3
    assert len(model.item_id) == 6


def test_fit_resets_state(intx_int):
    """Calling fit() a second time should clear previous weights."""
    model = RankFM(factors=2)
    model.fit(intx_int, epochs=2)
    w_first = model.w_i.copy()
    model.fit(intx_int, epochs=2)
    # State must be re-initialised (weights differ from first run due to new random seed)
    assert model.is_fit
    assert model.w_i.shape == w_first.shape


# =============================================================================
# 3. fit — bad inputs
# =============================================================================

def test_fit_bad_rating_col():
    """3-column interaction DataFrame (with ratings) should raise AssertionError."""
    intx = pd.DataFrame([
        (1, 1, 5), (1, 3, 2), (1, 5, 3),
        (2, 1, 2), (2, 2, 1), (2, 6, 4),
        (3, 3, 3), (3, 6, 4), (3, 4, 5),
    ], columns=['user_id', 'item_id', 'rating'], dtype=np.int32)
    with pytest.raises(AssertionError):
        RankFM(factors=2).fit(intx)


def test_fit_bad_uf_no_id(intx_int):
    """user_features without an id column should raise KeyError."""
    uf = pd.DataFrame([
        (0, 1, 5, 3.14),
        (1, 0, 6, 2.72),
        (0, 0, 4, 1.62),
    ], columns=['bin_1', 'bin_2', 'int', 'cnt'])
    with pytest.raises(KeyError):
        RankFM(factors=2).fit(intx_int, user_features=uf)


def test_fit_bad_uf_str_cols(intx_int):
    """user_features with string-valued columns should raise ValueError."""
    uf = pd.DataFrame([
        (1, 0, 1, "A", 3.14),
        (2, 1, 0, "B", 2.72),
        (3, 0, 0, "C", 1.62),
    ], columns=['user_id', 'bin_1', 'bin_2', 'str', 'cnt'])
    with pytest.raises(ValueError):
        RankFM(factors=2).fit(intx_int, user_features=uf)


def test_fit_bad_if_no_id(intx_int):
    """item_features without an id column should raise KeyError."""
    if_ = pd.DataFrame([
        (0, 1, 5, 3.14),
        (1, 0, 6, 2.72),
        (0, 0, 4, 1.62),
        (1, 1, 3, 1.05),
        (1, 0, 6, 0.33),
        (0, 0, 0, 0.00),
    ], columns=['bin_1', 'bin_2', 'int', 'cnt'])
    with pytest.raises(KeyError):
        RankFM(factors=2).fit(intx_int, item_features=if_)


def test_fit_bad_if_str_cols(intx_int):
    """item_features with string-valued columns should raise ValueError."""
    if_ = pd.DataFrame([
        (1, 0, 1, "A", 3.14),
        (2, 1, 0, "B", 2.72),
        (3, 0, 0, "C", 1.62),
        (4, 1, 1, "A", 1.05),
        (5, 1, 0, "F", 0.33),
        (6, 0, 0, "G", 0.00),
    ], columns=['item_id', 'bin_1', 'bin_2', 'str', 'cnt'])
    with pytest.raises(ValueError):
        RankFM(factors=2).fit(intx_int, item_features=if_)


# =============================================================================
# 4. fit_partial
# =============================================================================

def test_fit_partial_updates_weights(intx_int):
    """fit_partial should update model weights without resetting state."""
    model = RankFM(factors=2)
    model.fit(intx_int, epochs=2)
    w_before = model.w_i.copy()
    model.fit_partial(intx_int, epochs=2)
    assert model.is_fit
    assert not np.array_equal(w_before, model.w_i)


def test_fit_partial_preserves_user_items(intx_int):
    """fit_partial should not shrink the user_items mapping for known users."""
    model = RankFM(factors=2)
    model.fit(intx_int, epochs=1)
    items_before = {u: set(v.tolist()) for u, v in model.user_items.items()}

    model.fit_partial(intx_int, epochs=1)
    assert model.is_fit
    for u, items_set in items_before.items():
        assert set(model.user_items[u].tolist()) >= items_set


def test_fit_partial_sample_weight(intx_int):
    """fit_partial should accept a sample_weight vector without error."""
    model = RankFM(factors=2)
    model.fit(intx_int, epochs=1)
    weights = np.ones(len(intx_int), dtype=np.float32)
    model.fit_partial(intx_int, sample_weight=weights, epochs=1)
    assert model.is_fit


# =============================================================================
# 5. predict
# =============================================================================

def test_predict_train_pairs(intx_int, fitted_model):
    """Predict on all training pairs: correct shape, dtype, no NaN."""
    scores = fitted_model.predict(intx_int)
    assert scores.shape == (9,)
    assert scores.dtype == np.float32
    assert np.sum(np.isnan(scores)) == 0


def test_predict_cold_start_nan(fitted_model):
    """Pairs with unknown user/item using cold_start='nan': 4 NaN values."""
    scores = fitted_model.predict(_intx_disjoint, cold_start='nan')
    assert scores.shape == (9,)
    assert scores.dtype == np.float32
    assert np.sum(np.isnan(scores)) == 4


def test_predict_cold_start_drop(fitted_model):
    """Pairs with unknown user/item using cold_start='drop': 5 known pairs."""
    scores = fitted_model.predict(_intx_disjoint, cold_start='drop')
    assert scores.shape == (5,)
    assert scores.dtype == np.float32
    assert np.sum(np.isnan(scores)) == 0


def test_predict_cold_start_all_unknown(fitted_model):
    """All-unknown pairs with cold_start='drop' → empty result."""
    pairs = pd.DataFrame({'user_id': [99, 100], 'item_id': [99, 100]})
    scores = fitted_model.predict(pairs, cold_start='drop')
    assert scores.shape == (0,)


def test_predict_cold_start_invalid(intx_int, fitted_model):
    """Invalid cold_start value should raise ValueError."""
    with pytest.raises(ValueError):
        fitted_model.predict(intx_int, cold_start='invalid')


def test_predict_before_fit(intx_int):
    """predict() before fit() should raise AssertionError."""
    with pytest.raises(AssertionError):
        RankFM(factors=2).predict(intx_int)


# =============================================================================
# 6. recommend
# =============================================================================

def test_recommend_train_users(intx_int, fitted_model):
    """Recommend for known training users: type, shape, index, valid items."""
    recs = fitted_model.recommend(_train_users, n_items=3)
    assert isinstance(recs, pd.DataFrame)
    assert recs.shape == (3, 3)
    assert np.array_equal(recs.index.values, _train_users)
    assert recs.isin(intx_int['item_id'].values).all().all()


def test_recommend_filter_previous(intx_int, fitted_model):
    """filter_previous=True must not include any already-seen (user, item) pairs."""
    recs = fitted_model.recommend(_train_users, n_items=3, filter_previous=True)
    assert isinstance(recs, pd.DataFrame)
    assert recs.shape == (3, 3)
    assert recs.isin(intx_int['item_id'].values).all().all()

    recs_long = recs.stack().reset_index().drop('level_1', axis=1)
    recs_long.columns = ['user_id', 'item_id']
    overlap = pd.merge(intx_int, recs_long, on=['user_id', 'item_id'], how='inner')
    assert overlap.empty


def test_recommend_unseen_users_nan(intx_int, fitted_model):
    """Unseen users with cold_start='nan': row present but all NaN."""
    recs = fitted_model.recommend(_valid_users, n_items=3, cold_start='nan')
    assert isinstance(recs, pd.DataFrame)
    assert recs.shape == (4, 3)
    assert np.array_equal(sorted(recs.index.values), sorted(_valid_users))

    new_users = list(set(_valid_users.tolist()) - set(_train_users.tolist()))
    assert recs.loc[new_users].isnull().all().all()
    assert recs.dropna().isin(intx_int['item_id'].values).all().all()


def test_recommend_unseen_users_drop(intx_int, fitted_model):
    """Unseen users with cold_start='drop': only known users returned."""
    recs = fitted_model.recommend(_valid_users, n_items=3, cold_start='drop')
    assert recs.shape == (2, 3)
    assert np.isin(recs.index.values, _valid_users).all()

    same_users = sorted(set(_valid_users.tolist()) & set(_train_users.tolist()))
    assert np.array_equal(sorted(recs.index.values), same_users)


def test_recommend_all_unseen_drop(fitted_model):
    """All-unknown users with cold_start='drop' → empty DataFrame."""
    recs = fitted_model.recommend([99, 100], n_items=3, cold_start='drop')
    assert recs.empty


def test_recommend_batch_size_consistent(fitted_model):
    """batch_size should not change the output shape or values."""
    recs_default = fitted_model.recommend(_train_users, n_items=3)
    recs_batched = fitted_model.recommend(_train_users, n_items=3, batch_size=1)
    assert recs_default.shape == recs_batched.shape
    assert (recs_default.values == recs_batched.values).all()


def test_recommend_cold_start_invalid(fitted_model):
    """Invalid cold_start should raise ValueError."""
    with pytest.raises(ValueError):
        fitted_model.recommend(_train_users, n_items=3, cold_start='invalid')


def test_recommend_before_fit(intx_int):
    """recommend() before fit() should raise AssertionError."""
    with pytest.raises(AssertionError):
        RankFM(factors=2).recommend(_train_users)


# =============================================================================
# 7. similar_items
# =============================================================================

def test_similar_items_good(intx_int, fitted_model):
    """similar_items: correct shape, all in training set, excludes query item."""
    similar = fitted_model.similar_items(1, n_items=3)
    assert isinstance(similar, np.ndarray)
    assert similar.shape == (3,)
    assert np.isin(similar, intx_int['item_id'].unique()).all()
    assert 1 not in similar


def test_similar_items_unknown(fitted_model):
    """similar_items for an unknown item should raise AssertionError."""
    with pytest.raises(AssertionError):
        fitted_model.similar_items(99, n_items=3)


# =============================================================================
# 8. similar_users
# =============================================================================

def test_similar_users_good(intx_int, fitted_model):
    """similar_users: correct shape, all in training set, excludes query user."""
    similar = fitted_model.similar_users(1, n_users=2)
    assert isinstance(similar, np.ndarray)
    assert similar.shape == (2,)
    assert np.isin(similar, intx_int['user_id'].unique()).all()
    assert 1 not in similar


def test_similar_users_unknown(fitted_model):
    """similar_users for an unknown user should raise AssertionError."""
    with pytest.raises(AssertionError):
        fitted_model.similar_users(9, n_users=1)
