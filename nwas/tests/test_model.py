import numpy
import nwas
import pytest
import tensorflow as tf

@pytest.fixture
def tf_session():
    session = tf.Session()
    yield session
    session.close()

def test_spikeslab():
    p = 100
    theta = nwas.model.SpikeSlab(alpha=tf.ones([p]),
                                 beta=tf.zeros([p]),
                                 gamma=tf.ones([p]))

def test_spikeslab_zero_mean(tf_session):
    p = 100
    theta = nwas.model.SpikeSlab(alpha=tf.ones([p]),
                                 beta=tf.zeros([p]),
                                 gamma=tf.ones([p]))
    mean = tf_session.run(theta.mean())
    assert numpy.isclose(mean, 0).all()

def test_spikeslab_nonzero_mean(tf_session):
    p = 100
    for _ in range(50):
        beta = numpy.random.normal(size=p).astype('float32')
        theta = nwas.model.SpikeSlab(alpha=tf.ones([p]),
                                     beta=tf.convert_to_tensor(beta),
                                     gamma=tf.ones([p]))
        mean = tf_session.run(theta.mean())
        assert numpy.isclose(mean, beta).all()

def test_spikeslab_variance(tf_session):
    p = 100
    for _ in range(50):
        alpha = tf.sigmoid(tf.zeros([p]))
        beta = tf.zeros([p])
        gamma = tf.ones([p])
        theta = nwas.model.SpikeSlab(alpha=alpha,
                                     beta=beta,
                                     gamma=gamma)
        var = tf_session.run(theta.variance())
        assert numpy.isclose(var, 0.5).all()

def test_spikeslab_scalar_hyperparams():
    p = 100
    theta = nwas.model.SpikeSlab(alpha=tf.sigmoid(tf.zeros([1])),
                                 beta=tf.zeros([p]),
                                 gamma=tf.nn.softplus(tf.zeros([1])))

def test_spikeslab_zero_mean_scalar_hyperparams(tf_session):
    p = 100
    theta = nwas.model.SpikeSlab(alpha=tf.constant(0.1),
                                 beta=tf.zeros([p]),
                                 gamma=tf.ones([1]))
    mean = tf_session.run(theta.mean())
    assert numpy.isclose(mean, 0).all()

def test_spikeslab_nonzero_mean_scalar_hyperparams(tf_session):
    p = 100
    for _ in range(50):
        beta = numpy.random.normal(size=p).astype('float32')
        theta = nwas.model.SpikeSlab(alpha=tf.constant(0.1),
                                     beta=tf.convert_to_tensor(beta),
                                     gamma=tf.ones([1]))
        mean = tf_session.run(theta.mean())
        assert numpy.isclose(mean, 0.1 * beta).all()

def test_spikeslab_var_scalar_hyperparams(tf_session):
    p = 100
    for _ in range(50):
        beta = numpy.random.normal(size=p).astype('float32')
        theta = nwas.model.SpikeSlab(alpha=tf.constant(0.1),
                                     beta=tf.convert_to_tensor(beta),
                                     gamma=tf.ones([1]))
        var = tf_session.run(theta.variance())
        assert numpy.isclose(var, .1 + .09 * beta * beta).all()

def test_spikeslab_kl(tf_session):
    p = 100
    beta = numpy.random.normal(size=p).astype('float32')
    p_theta = nwas.model.SpikeSlab(alpha=tf.constant(0.1),
                                   beta=tf.zeros([1]),
                                   gamma=tf.ones([1]))
    q_theta = nwas.model.SpikeSlab(alpha=tf.ones([p]),
                                   beta=tf.convert_to_tensor(beta),
                                   gamma=tf.ones([p]))
    tf_session.run(tf.contrib.distributions.kl_divergence(q_theta, p_theta))
