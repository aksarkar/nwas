"""Approximate inference for models using SGVB

Author: Abhishek Sarkar <aksarkar@uchicago.edu>

"""
import numpy as np
import tensorflow as tf

def normal_llik(y, mean, prec):
  return -.5 * (-tf.log(prec) + tf.square(y - mean) * prec)

def normal_sample(mean, prec, n=1):
  samples = tf.random_normal([n]) * tf.ones(tf.shape(mean))
  return mean + samples * tf.sqrt(tf.reciprocal(prec))

def kl_normal_normal(mean_a, prec_a, mean_b, prec_b, reduce=True):
  """Rasmussen & Williams eq. A.23 for univariate Gaussians"""
  return .5 * (1 + tf.log(prec_a) - tf.log(prec_b) + prec_b * (tf.square(mean_a - mean_b) + 1 / prec_a))

def kl_bernoulli_bernoulli(p, q, reduce=True):
  """Rasmussen & Williams eq. A.22"""
  return (p * tf.log(p / q) + (1 - p) * tf.log((1 - p) / (1 - q)))

def biased_softplus(x, bias=1e-6):
  return bias + tf.nn.softplus(x)

def sigmoid(x):
  """Sigmoid clipped to float32 resolution

  This is needed because sigmoid(x) = 0 leads to NaN downstream

  """ 
  min_ = np.log(np.finfo('float32').resolution)
  return tf.nn.sigmoid(tf.clip_by_value(x, min_, -min_))

def sgvb(feed_dict, error, kl, opt, num_epochs=1000, learning_rate=1e-2, trace=None, verbose=False):
  elbo = error - tf.add_n(kl)
  opt.append(elbo)
  trace_ = [elbo, error] + kl
  if trace is not None:
    trace_.extend(trace)
  optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
  train = optimizer.minimize(-elbo)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_epochs):
      _, trace_output = sess.run([train, trace_], feed_dict)
      if np.isnan(trace_output[0]):
        raise tf.train.NanLossDuringTrainingError
      if verbose and not i % 100:
        print(i, *trace_output)
    return sess.run(opt, feed_dict)

def gaussian_spike_slab(x, y, stoch_samples=10, **kwargs):
  """Fit linear regression with the spike-and-slab prior using SGVB"""
  n, p = x.shape
  graph = tf.Graph()
  with graph.as_default():
    x_ph = tf.placeholder(tf.float32)
    y_ph = tf.placeholder(tf.float32)

    with tf.variable_scope('q', initializer=tf.zeros_initializer):
      # Residual inverse variance
      resid_prec_mean = tf.get_variable('resid_prec_mean', shape=[1])
      resid_prec_prec = biased_softplus(tf.get_variable('resid_prec_prec', shape=[1]))

      logodds_mean = tf.get_variable('logodds_mean', initializer=tf.constant([-10.]))
      logodds_prec = biased_softplus(tf.get_variable('q_logodds_log_prec', shape=[1]))

      # Effect size inverse variance
      effect_prec_mean = tf.get_variable('effect_prec_mean', shape=[1])
      effect_prec_prec = biased_softplus(tf.get_variable('effect_prec_prec', shape=[1]))

      pip = sigmoid(tf.get_variable('pip', shape=[p, 1]))
      mean = tf.get_variable('effect_mean', shape=[p, 1])
      prec = biased_softplus(tf.get_variable('prec', shape=[p, 1]))

    effect_posterior_mean = pip * mean
    effect_posterior_var = pip / prec + pip * (1 - pip) * tf.square(mean)

    eta_mean = tf.matmul(x_ph, effect_posterior_mean)
    eta_var = tf.matmul(tf.square(x_ph), effect_posterior_var)

    eta = normal_sample(eta_mean, tf.reciprocal(eta_var), stoch_samples)
    resid_prec = biased_softplus(normal_sample(resid_prec_mean, resid_prec_prec, stoch_samples))
    odds = sigmoid(normal_sample(logodds_mean, logodds_prec, stoch_samples))
    effect_prec = tf.exp(normal_sample(effect_prec_mean, effect_prec_prec, stoch_samples))

    error = tf.reduce_mean(tf.reduce_sum(normal_llik(y_ph, eta, resid_prec), axis=1))
    kl = [
      tf.reduce_mean(tf.reduce_sum(kl_bernoulli_bernoulli(pip, odds), axis=1)),
      tf.reduce_mean(tf.reduce_sum(pip * kl_normal_normal(mean, prec, tf.constant(0.), effect_prec), axis=1)),
      tf.reduce_sum(kl_normal_normal(logodds_mean, logodds_prec, tf.constant(0.), tf.constant(1.))),
      tf.reduce_sum(kl_normal_normal(effect_prec_mean, effect_prec_prec, tf.constant(0.), tf.constant(1.))),
    ]

    # GLM coefficient of determination
    R = 1 - tf.reduce_sum(tf.square(y_ph - eta_mean)) / tf.reduce_sum(tf.square(y_ph - tf.reduce_mean(y_ph)))
    opt = [pip, effect_posterior_mean, effect_posterior_var,
           logodds_mean, tf.reciprocal(biased_softplus(logodds_prec)),
           effect_prec_mean,
           tf.reciprocal(biased_softplus(effect_prec_prec))]
    return sgvb({x_ph: x, y_ph: y}, error, kl, opt, trace=[R], **kwargs)

def project_simplex(x):
  """Project x onto the probability simplex
  
  Wang & Carreira-Perpiñán https://arxiv.org/abs/1309.1541
  """
  u = tf.nn.top_k(x, k=x.shape[-1]).values
  cu = tf.cumsum(u, axis=-1)
  j = tf.ones([x.shape[0], 1]) * tf.reshape(tf.cast(tf.range(x.shape[-1]), tf.float32), [1, -1])
  rho = tf.reduce_sum(tf.map_fn(
    lambda val: tf.cast(val[0] + (1 - val[1]) / (1 + val[2]) > 0,
                        tf.float32),
    [u, cu, j], dtype=tf.float32), axis=-1, keep_dims=True)
  lambda_ = (1 - tf.reduce_sum(cu * tf.one_hot(tf.cast(tf.reshape(rho - 1, [-1]), tf.int32), x.shape[-1]), axis=-1, keep_dims=True)) / rho
  return tf.maximum(x + lambda_, tf.constant(1e-8))

def kl_categorical_categorical(logits_p, logits_q):
  return tf.nn.sigmoid(logits_p) * (tf.nn.softplus(-logits_q) - tf.nn.softplus(-logits_p))

def categorical_slab_cov(probs, mean, prec):
  if probs.shape[0] == 1:
    cov = tf.zeros([probs.shape[1], probs.shape[1]])
  else:
    cov = (tf.matmul(probs, probs, transpose_a=True) *
           tf.matmul(mean, mean, transpose_a=True))
  diag = tf.diag(tf.reduce_sum(probs * (tf.square(mean) + tf.reciprocal(prec)),
                               axis=0))
  cov += diag
  return cov

def gaussian_categorical_slab(x, y, l, stoch_samples=10, **kwargs):
  """Fit the model assuming l causal variants"""
  n, p = x.shape
  graph = tf.Graph()
  with graph.as_default():
    x_ph = tf.placeholder(tf.float32)
    y_ph = tf.placeholder(tf.float32)

    with tf.variable_scope('hyperparams', initializer=tf.zeros_initializer):
      # Residual inverse variance
      v_mean = tf.get_variable('v_mean', [1])
      v_prec = biased_softplus(tf.get_variable('v_prec', [1]))
      # Effect size inverse variance
      v_b_mean = tf.get_variable('v_b_mean', [1])
      v_b_prec = biased_softplus(tf.get_variable('v_b_prec', [1]))
    with tf.variable_scope('params', initializer=tf.random_normal_initializer):
      logit_pi = tf.get_variable('pi', [l, p])
      pi = tf.nn.softmax(logit_pi)
      mu = tf.get_variable('mu', [l, p])
      phi = biased_softplus(tf.get_variable('phi', [l, p]))

    b_mean = tf.transpose(tf.reduce_sum(pi * mu, axis=0, keep_dims=True))
    # Important: there is non-zero covariance, but we're not going to use it
    # here
    b_var = tf.transpose(tf.reduce_sum(pi / phi + pi * (1 - pi) * tf.square(mu), axis=0, keep_dims=True))

    # eta = x b'
    eta_mean = tf.matmul(x, b_mean)
    # Important: this is not the same as for the spike-and-slab prior because of Cov(b_j, b_k)!
    # Matrix Cookbook eq. 313
    eta_cov = tf.matmul(tf.matmul(x, categorical_slab_cov(pi, mu, phi)), x, transpose_b=True)
    jitter = tf.constant(1e-6) * tf.reduce_mean(tf.diag_part(eta_cov)) * tf.eye(tf.shape(eta_cov)[0])
    eta_l = tf.cholesky(eta_cov + jitter)
    # [stoch_samples, n]
    eta = tf.transpose(eta_mean) + tf.transpose(tf.matmul(eta_l, tf.random_normal([n, stoch_samples])))
    v = tf.exp(normal_sample(v_mean, v_prec, stoch_samples))
    error = tf.reduce_mean(tf.reduce_sum(normal_llik(y_ph, eta, v), axis=1))

    v_b = tf.exp(normal_sample(v_b_mean, v_b_prec, stoch_samples))
    kl = [
      tf.reduce_sum(kl_normal_normal(v_mean, v_prec, tf.constant(0.), tf.constant(1.))),
      tf.reduce_sum(kl_normal_normal(v_b_mean, v_b_prec, tf.constant(0.), tf.constant(1.))),
      tf.reduce_sum(kl_categorical_categorical(logit_pi, tf.fill([l, p], -tf.log(p - 1.)))),
      tf.reduce_mean(tf.reduce_sum(kl_normal_normal(mu, phi, tf.constant(0.), biased_softplus(v_b_mean)), axis=1)),
    ]

    # GLM coefficient of determination
    R = 1 - tf.reduce_sum(tf.square(y_ph - eta_mean)) / tf.reduce_sum(tf.square(y_ph - tf.reduce_mean(y_ph)))
    opt = [pi, b_mean, b_var, v_mean, v_prec, v_b_mean, v_b_prec]

    return sgvb({x_ph: x, y_ph: y}, error, kl, opt, trace=[R], **kwargs)

if __name__ == '__main__':
  import nwas
  
  with nwas.simulation.simulation(p=1000, pve=0.5, annotation_params=[(10, 1)], seed=0) as s:
    x, y = s.sample_gaussian(n=500)
    x = x.astype('float32')
    y = y.reshape(-1, 1).astype('float32')
