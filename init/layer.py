import tensorflow as tf

def linear(input, output_dim, scope = None, name = None, stddev = 1.0, lamb = 0.001):
	with tf.variable_scope(scope or "linear"):
		w = tf.get_variable(
				name = "w",
				shape = [input.get_shape()[1], output_dim], 
				initializer = tf.random_normal_initializer(stddev = stddev)
			)
		b = tf.get_variable(
				name = "b",
				shape = [output_dim],
				initializer = tf.constant_initializer(0.0)
			)
		tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lamb)(w))

	return tf.add(tf.matmul(input, w), b, name = name)

def cnn2d(input, filter_shape, strid_shape, padding, scope = None, name = None, stddev = 1.0, lamb = 0.001):
	with tf.variable_scope(scope or "cnn"):
		w = tf.get_variable(
				name = "w",
				shape = filter_shape,
				initializer = tf.random_normal_initializer(stddev = stddev)
			)
		b = tf.get_variable(
				name = "b",
				shape = filter_shape[-1],
				initializer = tf.constant_initializer(0.0)
			)
		tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lamb)(w))
	return tf.add(
				tf.nn.conv2d(
					input = input, 
					filter = w,
					strides = strid_shape,
					padding = padding
				),
				b,
				name = name
			)
def decnn2d(input, filter_shape, output_shape, strid_shape, padding, scope = None, name = None, stddev = 1.0):
	with tf.variable_scope(scope or "decnn"):
		w = tf.get_variable(
				name = "w",
				shape = filter_shape,
				initializer = tf.random_normal_initializer(stddev = stddev)
			)
		b = tf.get_variable(
				name = "b",
				shape = filter_shape[-2],
				initializer = tf.constant_initializer(0.0)
			)

	return tf.add(
				tf.nn.conv2d_transpose(
					value = input,
					filter = w,
					output_shape = output_shape,
					strides = strid_shape,
				),
				b,
				name = name
			)

def bn_layer(x, is_training, scope = None, name = None, moving_decay = 0.9, eps = 1e-5):
	param_shape = x.get_shape()[-1]
	with tf.variable_scope(scope or "BatchNorm"):
		gamma = tf.get_variable(
					name = "gamma",
					shape = param_shape,
					initializer = tf.constant_initializer(1)
				)
		beta = tf.get_variable(
					name = "beat",
					shape = param_shape,
					initializer = tf.constant_initializer(0)
				)
		axis = list(range(len(x.get_shape()) - 1))
		batch_mean, batch_var = tf.nn.moments(
				x, 
				axis, 
				name = "moments"
			)		
		ema = tf.train.ExponentialMovingAverage(
					decay = moving_decay,
					name = "ExponentialMovingAverage"
				)
		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		mean, var = tf.cond(
						pred = tf.equal(is_training, True), 
						true_fn = mean_var_with_update, 
						false_fn = lambda : (ema.average(batch_mean), ema.average(batch_var))
					)
	return tf.nn.batch_normalization(
				x = x, 
				mean = mean, 
				variance = var, 
				offset = beta, 
				scale = gamma, 
				variance_epsilon = eps, 
				name = name
			)

def clip(var_list):
	clip_ops = []
	for var in var_list:
		clip_bounds = [-.01, .01]
		clip_ops.append(
				tf.assign(
					var, 
					tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
				)
			)
	clip_disc_weights = tf.group(*clip_ops)
	return clip_disc_weights


def LeakyReLU(x, alpha = 0.2):
	return tf.maximum(alpha*x, x)

def optimizer(loss, var_list, gan_type = "wgan-gp", learning_rate = 0.01):

	if gan_type == "wgan":
		optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(
			loss,
			var_list = var_list
		)

	if gan_type == "wgan-gp":
		optimizer = tf.train.AdamOptimizer(
					learning_rate, beta1 = 0.5, beta2 = 0.9
				).minimize(
						loss, var_list = var_list
					)

	return optimizer