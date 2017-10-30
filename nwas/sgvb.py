"""Approximate inference for models using SGVB

Author: Abhishek Sarkar <aksarkar@uchicago.edu>

"""
import numpy as np
import tensorflow as tf

def normal_llik(y, mean, prec):
  return -.5 * (-tf.log(prec) + tf.square(y - mean) * prec)

def kl_bernoulli_bernoulli(p, q, reduce=True):
  """Rasmussen & Williams eq. A.22"""
  return (p * tf.log(p / q) + (1 - p) * tf.log((1 - p) / (1 - q)))

def kl_normal_normal(mean_a, prec_a, mean_b, prec_b, reduce=True):
  """Rasmussen & Williams eq. A.23 for univariate Gaussians"""
  return .5 * (1 + tf.log(prec_a) - tf.log(prec_b) + prec_b * (tf.square(mean_a - mean_b) + 1 / prec_a))

def biased_softplus(x, bias=1e-6):
  return bias + tf.nn.softplus(x)

def sigmoid(x):
  """Sigmoid clipped to float32 resolution

  This is needed because sigmoid(x) = 0 leads to NaN downstream

  """ 
  min_ = np.log(np.finfo('float32').resolution)
  return tf.nn.sigmoid(tf.clip_by_value(x, min_, -min_))

def gaussian_spike_slab(x, y, num_epochs=1000, learning_rate=1e-2, stoch_samples=10, verbose=False):
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
      odds = sigmoid(logodds_mean)

      # Effect size inverse variance
      effect_prec_mean = tf.get_variable('effect_prec_mean', shape=[1])
      effect_prec_prec = biased_softplus(tf.get_variable('effect_prec_prec', shape=[1]))
      effect_prec = biased_softplus(effect_prec_mean)

      pip = sigmoid(tf.get_variable('pip', shape=[p, 1]))
      mean = tf.get_variable('effect_mean', shape=[p, 1])
      prec = biased_softplus(tf.get_variable('prec', shape=[p, 1]))

    effect_posterior_mean = pip * mean
    effect_posterior_var = pip / prec + pip * (1 - pip) * tf.square(mean)

    eta_mean = tf.matmul(x_ph, effect_posterior_mean)
    eta_std = tf.sqrt(tf.matmul(tf.square(x_ph), effect_posterior_var))
    noise = tf.random_normal([stoch_samples, 2])
    eta = eta_mean + noise[:,0] * eta_std
    phi = biased_softplus(effect_prec_mean + noise[:,1] * tf.sqrt(tf.reciprocal(effect_prec_prec)))

    llik = tf.reduce_mean(tf.reduce_sum(normal_llik(y_ph, eta, phi), axis=0))
    kl_terms = [
      tf.reduce_sum(kl_bernoulli_bernoulli(pip, odds)),
      tf.reduce_sum(pip * kl_normal_normal(mean, prec, tf.constant(0.), effect_prec)),
      tf.reduce_sum(kl_normal_normal(logodds_mean, logodds_prec, tf.constant(0.), tf.constant(1.))),
      tf.reduce_sum(kl_normal_normal(effect_prec_mean, effect_prec_prec, tf.constant(0.), tf.constant(1.))),
    ]
    elbo = llik - tf.add_n(kl_terms)

    # GLM coefficient of determination
    R = 1 - tf.reduce_sum(tf.square(y_ph - eta_mean)) / tf.reduce_sum(tf.square(y_ph - tf.reduce_mean(y_ph)))

    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(-elbo)
    trace = [elbo, llik, R] + kl_terms
    opt = [pip, effect_posterior_mean, effect_posterior_var, odds, effect_prec, elbo]

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for i in range(num_epochs):
        _, trace_output = sess.run([train, trace], feed_dict={x_ph: x, y_ph: y})
        if np.isnan(trace_output[0]):
          raise tf.train.NanLossDuringTrainingError
        if verbose and not i % 100:
          print(i, *trace_output)
      return sess.run(opt, feed_dict={x_ph: x, y_ph: y})

def logit(x):
  min_ = np.log(np.finfo('float32').resolution)
  return tf.clip_by_value(tf.log(x) - tf.log1p(-x), min_, -min_)

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

def gaussian_categorical_slab(x, y, l, num_epochs=1000, stoch_samples=10, learning_rate=1e-2):
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
      pi = sigmoid(logit_pi)
      mu = tf.get_variable('mu', [l, p])
      phi = biased_softplus(tf.get_variable('phi', [l, p]))

    # TODO: priors appear only in the KL terms which is tricky for
    # reading/understanding, but bayesflow/Edward don't DTRT
    kl_terms = [
      tf.reduce_sum(kl_normal_normal(v_mean, v_prec, tf.constant(0.), tf.constant(1.))),
      tf.reduce_sum(kl_normal_normal(v_b_mean, v_b_prec, tf.constant(0.), tf.constant(1.))),
      tf.reduce_sum(kl_categorical_categorical(logit_pi, tf.fill([l, p], -tf.log(p - 1.)))),
      tf.reduce_sum(kl_normal_normal(mu, phi, tf.constant(0.), biased_softplus(v_b_mean))),
    ]

    b_mean = tf.transpose(tf.reduce_sum(pi * mu, axis=0, keep_dims=True))
    b_var = tf.transpose(tf.reduce_sum(pi / phi + pi * (1 - pi) * tf.square(mu), axis=0, keep_dims=True))
    eta_mean = tf.matmul(x, b_mean)
    eta_var = tf.matmul(tf.square(x), b_var)

    noise = tf.random_normal([stoch_samples, 2])
    eta_samples = eta_mean + noise[:,0] * tf.sqrt(eta_var)
    v_samples = biased_softplus(v_mean + noise[:,1] * tf.sqrt(tf.reciprocal(v_prec)))

    llik = tf.reduce_mean(tf.reduce_sum(normal_llik(y_ph, eta_samples, v_samples), axis=1))
    elbo = llik - tf.add_n(kl_terms)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(-elbo)

    with tf.control_dependencies([train]):
      pi = project_simplex(pi)
      train = tf.assign(logit_pi, tf.log(pi) - tf.log1p(-pi))

    # GLM coefficient of determination
    R = 1 - tf.reduce_sum(tf.square(y_ph - eta_mean)) / tf.reduce_sum(tf.square(y_ph - tf.reduce_mean(y_ph)))
    trace = [elbo, llik, R] + kl_terms
    opt = [pi, mu, phi, biased_softplus(v_mean), v_prec, biased_softplus(v_b_mean, bias=1e-3), v_b_prec]

  with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_epochs):
      _, trace_output = sess.run([train, trace], feed_dict={x_ph: x, y_ph: y})
      if np.isnan(trace_output).any():
        raise tf.train.NanLossDuringTrainingError
      if (np.array(trace_output[-4:]) <= 0).any():
        import pdb; pdb.set_trace()
      if not i % 100:
        print(i, *trace_output)
    return sess.run(opt, feed_dict={x_ph: x, y_ph: y})
