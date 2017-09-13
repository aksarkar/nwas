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
