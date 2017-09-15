import edward as ed
import numpy
import nwas
import pytest
import tensorflow as tf

from nwas.models import *

@pytest.fixture
def tf_session():
    session = tf.Session()
    yield session
    session.close()

def test_localreparameterization():
    LocalReparameterization(
        ed.models.Normal(loc=tf.ones([10]), scale=tf.ones([1])))

def test_localreparameterization_Variable():
    LocalReparameterization(
        ed.models.Normal(loc=tf.Variable(tf.zeros([10])),
                         scale=tf.Variable(tf.zeros([10]))))

def test_localreparameterization_shape():
    eta = LocalReparameterization(
        ed.models.Normal(loc=tf.Variable(tf.zeros([10, 1])),
                         scale=tf.Variable(tf.zeros([10, 1]))))
    assert eta.shape == (10, 1)

def test_localreparameterization_mean(tf_session):
    loc = numpy.random.normal(size=(10, 1)).astype('float32')
    eta = LocalReparameterization(
        ed.models.Normal(loc=loc,
                         scale=tf.ones([1])))
    assert numpy.isclose(tf_session.run(eta.mean()),loc).all()

def test_localreparameterization_kl():
    eta = LocalReparameterization(
        ed.models.Normal(loc=tf.random_normal([10]),
                         scale=tf.ones([10])))
    q_eta = LocalReparameterization(
        ed.models.Normal(loc=tf.Variable(tf.random_normal([10])),
                         scale=tf.Variable(tf.random_normal([10]))))
    assert tf.contrib.distributions.kl_divergence(eta, q_eta) == 0

def test_localreparameterization_sample():
    eta = LocalReparameterization(
        ed.models.Normal(loc=tf.random_normal([10]),
                         scale=tf.ones([10])))
    eta.value()

def test_localreparameterization_edward_copy(tf_session):
    eta = LocalReparameterization(
        ed.models.Normal(loc=tf.random_normal([10]),
                         scale=tf.ones([10])))
    val = tf_session.run(ed.util.copy(eta))
