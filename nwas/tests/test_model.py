import edward as ed
import numpy
import nwas
import pytest
import scipy.special
import tensorflow as tf

from nwas.model import *

@pytest.fixture
def tf_session():
    session = tf.Session()
    yield session
    session.close()

def test_spikeslab():
    p = 100
    theta = SpikeSlab(logodds=tf.zeros([p, 1]),
                      loc=tf.zeros([p, 1]),
                      scale=tf.zeros([p, 1]))

def test_spikeslab_zero_mean(tf_session):
    p = 100
    theta = SpikeSlab(logodds=tf.zeros([p, 1]),
                      loc=tf.zeros([p, 1]),
                      scale=tf.zeros([p, 1]))
    mean = tf_session.run(theta.mean())
    assert mean.shape == (p, 1)
    assert numpy.isclose(mean, 0).all()

def test_spikeslab_nonzero_mean(tf_session):
    p = 100
    for _ in range(50):
        loc = numpy.random.normal(size=p).astype('float32')
        theta = SpikeSlab(logodds=tf.fill([p, 1], 10.0),
                          loc=tf.convert_to_tensor(loc),
                          scale=tf.ones([p, 1]))
        mean = tf_session.run(theta.mean())
        assert numpy.isclose(mean, scipy.special.expit(10) * loc).all()

def test_spikeslab_variance(tf_session):
    p = 100
    for _ in range(50):
        loc = numpy.random.normal(size=(p, 1)).astype('float32')
        theta = SpikeSlab(logodds=tf.zeros([p, 1]),
                          loc=tf.convert_to_tensor(loc),
                          scale=tf.zeros([p, 1]))
        var = tf_session.run(theta.variance())
        assert var.shape == (p, 1)
        assert numpy.isclose(var, 0.5 / numpy.log(2) + .25 * loc * loc).all()

def test_spikeslab_scalar_hyperparams():
    p = 100
    theta = SpikeSlab(logodds=tf.zeros([1]),
                      loc=tf.zeros([p, 1]),
                      scale=tf.zeros([1]))

def test_spikeslab_zero_mean_scalar_hyperparams(tf_session):
    p = 100
    theta = SpikeSlab(logodds=tf.constant(10.0),
                      loc=tf.zeros([p, 1]),
                      scale=tf.zeros([1]))
    mean = tf_session.run(theta.mean())
    assert mean.shape == (p, 1)
    assert numpy.isclose(mean, 0).all()

def test_spikeslab_nonzero_mean_scalar_hyperparams(tf_session):
    p = 100
    for _ in range(50):
        loc = numpy.random.normal(size=p).astype('float32').reshape(-1, 1)
        theta = SpikeSlab(logodds=tf.constant(10.0),
                          loc=tf.convert_to_tensor(loc),
                          scale=tf.zeros([1]))
        mean = tf_session.run(theta.mean())
        assert mean.shape == (p, 1)
        assert numpy.isclose(mean, scipy.special.expit(10) * loc).all()

def test_spikeslab_var_scalar_hyperparams(tf_session):
    p = 100
    for _ in range(50):
        loc = numpy.random.normal(size=p).astype('float32').reshape(-1, 1)
        theta = SpikeSlab(logodds=tf.constant(0.0),
                          loc=tf.convert_to_tensor(loc),
                          scale=tf.zeros([1]))
        var = tf_session.run(theta.variance())
        assert var.shape == (p, 1)
        assert numpy.isclose(var, 0.5 / numpy.log(2) + 0.25 * loc * loc).all()

def test_spikeslab_kl(tf_session):
    p = 100
    loc = numpy.random.normal(size=p).astype('float32').reshape(-1, 1)
    p_theta = SpikeSlab(logodds=tf.constant(-2.0),
                        loc=tf.zeros([1]),
                        scale=tf.ones([1]))
    q_theta = SpikeSlab(logodds=tf.zeros([p, 1]),
                        loc=tf.convert_to_tensor(loc),
                        scale=tf.zeros([p, 1]))
    tf_session.run(tf.contrib.distributions.kl_divergence(q_theta, p_theta))

def test_spikeslab_variable():
    p = 100
    theta = SpikeSlab(
        logodds=tf.Variable(tf.zeros([p, 1])),
        loc=tf.Variable(tf.random_normal([p, 1])),
        scale=tf.Variable(tf.zeros([p, 1]))
    )

def test_spikeslab_edward_copy():
    p_theta = SpikeSlab(logodds=tf.constant(0.1),
                        loc=tf.zeros([1]),
                        scale=tf.zeros([1]))
    copy = ed.util.copy(p_theta)

def test_geneticvalue():
    n = 500
    p = 100
    m = 10
    x_ref = numpy.zeros((n, p)).astype('float32')
    theta = SpikeSlab(logodds=tf.constant(0.1),
                      loc=tf.zeros([p, m]),
                      scale=tf.constant(0.0),
    )
    GeneticValue(x=x_ref, theta=theta)

def test_geneticvalue_variable():
    n = 500
    p = 100
    m = 10
    x_ref = numpy.zeros((n, p)).astype('float32')
    q_theta = SpikeSlab(
        logodds=tf.Variable(tf.zeros([p, m])),
        loc=tf.Variable(tf.zeros([p, m])),
        scale=tf.Variable(tf.zeros([p, m])))
    GeneticValue(x=x_ref, theta=q_theta)

def test_geneticvalue_shape():
    n = 500
    p = 100
    m = 10
    x_ref = numpy.zeros((n, p)).astype('float32')
    q_theta = SpikeSlab(
        logodds=tf.Variable(tf.zeros([p, m])),
        loc=tf.Variable(tf.zeros([p, m])),
        scale=tf.Variable(tf.zeros([p, m])))
    q_eta_ref_m = GeneticValue(x=x_ref, theta=q_theta)
    assert q_eta_ref_m.shape == (n, m)

def test_geneticvalue_edward_impl():
    n = 500
    p = 100
    m = 10
    x_ref = numpy.zeros((n, p)).astype('float32')
    q_theta = SpikeSlab(
        logodds=tf.Variable(tf.zeros([p, m])),
        loc=tf.Variable(tf.zeros([p, m])),
        scale=tf.Variable(tf.zeros([p, m])))
    q_eta_ref_m = GeneticValue(x=x_ref, theta=q_theta)

def test_geneticvalue_edward_copy():
    n = 500
    p = 100
    m = 10
    x_ref = numpy.zeros((n, p)).astype('float32')
    q_theta = SpikeSlab(
        logodds=tf.Variable(tf.zeros([p, m])),
        loc=tf.Variable(tf.zeros([p, m])),
        scale=tf.Variable(tf.zeros([p, m])))
    q_eta_ref_m = GeneticValue(x=x_ref, theta=q_theta)
    copy = ed.util.copy(q_eta_ref_m)
