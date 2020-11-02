import tensorflow as tf
from .layer import linear, bn_layer, decnn2d, clip, optimizer

def Generator(noise, is_training, batch_size):
	l_1 = linear(
				input = noise, 
				output_dim = 4*6*128, 
				scope = "l_1", 
				name = "l1", 
			)
	bn_1 = bn_layer(
				x = l_1,
				is_training = is_training,
				scope = "bn_1",
				name = "bn1"
			)

	bn_1 = tf.reshape(tf.nn.relu(bn_1), [-1, 6, 4, 128])

	dec_2 = decnn2d(
				input = bn_1, 
				filter_shape = [5, 5, 64, 128], 
				output_shape = [batch_size, 11, 8, 64], 
				strid_shape = [1, 2, 2, 1], 
				padding = "SAME", 
				scope = "decnn_2", 
				name = "decnn2",
			)

	bn_2 = bn_layer(
				x = dec_2,
				is_training = is_training,
				scope = "bn_2",
				name = "bn2"
			)
	dec_3 = decnn2d(
				input = tf.nn.relu(bn_2), 
				filter_shape = [3, 3, 32, 64], 
				output_shape = [batch_size, 22, 8, 32], 
				strid_shape = [1, 2, 1, 1], 
				padding = "SAME", 
				scope = "decnn_3", 
				name = "decnn3",
			)
	bn_3 = bn_layer(
				x = dec_3,
				is_training = is_training,
				scope = "bn_3",
				name = "bn3"
			)
	dec_4 = decnn2d(
				input = tf.nn.relu(bn_3), 
				filter_shape = [3, 3, 16, 32], 
				output_shape = [batch_size, 43, 8, 16], 
				strid_shape = [1, 2, 1, 1], 
				padding = "SAME", 
				scope = "decnn_4", 
				name = "decnn4",
			)
	bn_4 = bn_layer(
				x = dec_4,
				is_training = is_training,
				scope = "bn_4",
				name = "bn4"
			)
	dec_5 = decnn2d(
				input = tf.nn.relu(bn_4), 
				filter_shape = [3, 3, 1, 16], 
				output_shape = [batch_size, 85, 8, 1], 
				strid_shape = [1, 2, 1, 1], 
				padding = "SAME", 
				scope = "decnn_5", 
				name = "decnn5",
			)

	output = tf.nn.sigmoid(dec_5, name = "output")

	return output

def Discriminator(inputs, one_hot_fea, matrixs, atom2vec_len, batch_size, is_training):
	inputs = tf.squeeze(inputs)
	atom_num = one_hot_fea.get_shape()[0]
	atom2vec = tf.nn.tanh(
				linear(one_hot_fea, atom2vec_len, scope = "atom2vec", name = "atom2vec_")
			)
	atom2vec_ = tf.reshape(tf.tile(atom2vec, [batch_size, 1]), [batch_size, atom_num, atom2vec_len])
	inputs = tf.transpose(inputs, [0, 2, 1])
	material_fea = tf.reduce_sum(tf.multiply(tf.matmul(inputs, atom2vec_), matrixs), axis = 1)
	l_1 = linear(material_fea, 128, scope = "1", name = "1_")
	l_2 = linear(tf.nn.relu(l_1), 64, scope = "2", name = "2_")
	l_3 = linear(tf.nn.relu(l_2), 32, scope = "3", name = "3_")
	output = linear(tf.nn.relu(l_3), 1, scope = "output", name = "output_")

	return atom2vec, output


class GAN(object):
	def __init__(self, args, one_hot_fea, is_train = True):
		
		self.is_train = is_train
		self.args = args
		matrixs = []
		for i in range(self.args.max_idx):
			matrix = tf.zeros((1, args.atom2vec_len), dtype = tf.float32, name = None) + i + 1
			matrixs.append(matrix)
		matrixs = tf.concat(matrixs, axis = 0)

		self.matrixs = tf.reshape(
						tf.tile(matrixs, [self.args.batch_size, 1]), 
						[self.args.batch_size, self.args.max_idx, args.atom2vec_len]
					)

		self.one_hot_fea = tf.convert_to_tensor(one_hot_fea, dtype = tf.float32)

		self.real_data = tf.placeholder(
							tf.float32, 
							shape = (self.args.batch_size, self.args.atom_num, self.args.max_idx, 1), 
							name = "real_data"
						)
		self.random = tf.placeholder(
							tf.float32,
							shape = (self.args.batch_size, self.args.random_len),
							name = "random"
						)
		self.build(self.is_train)
		
	def build(self, is_train = True):

		with tf.variable_scope("G"):
			self.fake_data = Generator(
									self.random, 
									is_training = is_train,
									batch_size = self.args.batch_size
								)

		with tf.variable_scope("D"):
			self.atom2vec, self.disc_real = Discriminator(
												self.real_data,
												self.one_hot_fea,
												self.matrixs,
												self.args.atom2vec_len,
												self.args.batch_size,
												is_training = is_train
											)

		with tf.variable_scope("D", reuse = tf.AUTO_REUSE):
			_, self.disc_fake = Discriminator(
								self.fake_data,
								self.one_hot_fea,
								self.matrixs,
								self.args.atom2vec_len,
								self.args.batch_size,
								is_training = is_train
							)

		self.gen_cost = -tf.reduce_mean(self.disc_fake)
		self.disc_cost = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(self.disc_real)

		if self.args.gan_type == "wgan-gp":
			epsilon = tf.random_uniform(
						shape = [self.args.batch_size, 1, 1, 1], minval = 0., maxval = 1.
					)

			interpolated_image = self.real_data + epsilon * (self.fake_data - self.real_data)

			with tf.variable_scope("D", reuse = tf.AUTO_REUSE):
				d_interpolated = Discriminator(
									interpolated_image,
									self.one_hot_fea,
									self.matrixs,
									self.args.atom2vec_len,
									self.args.batch_size,
									is_training = is_train
								)
			grad_d_interpolated = tf.gradients(
											d_interpolated, [interpolated_image]
										)[0]
			slopes = tf.sqrt(
						1e-8 + tf.reduce_sum(
									tf.square(grad_d_interpolated), 
									axis = [1, 2, 3]
								)
					)
			gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

			self.disc_cost += self.args.gamma * gradient_penalty
			tf.summary.scalar("loss/gradient_penalty", gradient_penalty)

		vars = tf.trainable_variables()

		self.g_params = [v for v in vars if v.name.startswith("G/")]
		self.d_params = [v for v in vars if v.name.startswith("D/")]

		if self.args.gan_type == "wgan":
			self.wclip = clip(self.d_params)
	
		self.opt_g = optimizer(
						self.gen_cost, self.g_params, learning_rate = self.args.g_lr
					)
		self.opt_d = optimizer(
						self.disc_cost, self.d_params, learning_rate = self.args.d_lr
					)

def Fcc(inputs, is_train):
    l_1 = linear(inputs, 128, scope = "1", name = "1_")
    bn_1 = bn_layer(l_1, is_train, "bn_1", "bn1")
    l_2 = linear(bn_1, 128, scope = "2", name = "2_")
    bn_2 = bn_layer(l_2, is_train, "bn_2", "bn2")
    l_3 = linear(tf.nn.relu(bn_2), 64, scope = "3", name = "3_")
    bn_3 = bn_layer(l_3, is_train, "bn_3", "bn3")
    l_4 = linear(tf.nn.relu(bn_3), 64, scope = "4", name = "4_")
    bn_4 = bn_layer(l_4, is_train, "bn_4", "bn4")
    l_5 = linear(tf.nn.relu(bn_4), 32, scope = "5", name = "5_")
    bn_5 = bn_layer(l_5, is_train, "bn_5", "bn5")
    l_6 = linear(tf.nn.relu(bn_5), 32, scope = "6", name = "6_")
    bn_6 = bn_layer(l_6, is_train, "bn_6", "bn6")
    output = linear(tf.nn.relu(bn_6), 1, scope = "output", name = "output_")
    return output

class DNN(object):
	def __init__(self, args, is_train):
		self.is_train = is_train
		self.args = args
		self.x = tf.placeholder(
					tf.float32, 
					shape = (self.args.batch_size, self.args.fea_len), 
					name = "x"
				)
		self.y = tf.placeholder(
					tf.float32, 
					shape = (self.args.batch_size, 1), 
					name = "y"
				)
		self.lr = tf.placeholder(
					tf.float32,
					name = "lr"
				)
		self.build(self.is_train)

	def build(self, is_train):
		regularizer = tf.contrib.layers.l2_regularizer(0.001)
		with tf.variable_scope("F"):
			self.pre = Fcc(self.x, is_train)
		self.mae_loss = tf.reduce_mean(tf.square(tf.subtract(self.pre, self.y))) 
		tf.add_to_collection('losses', self.mae_loss)
		self.regularizer_loss = tf.add_n(tf.get_collection("losses"))
		vars = tf.trainable_variables()
		self.params = [v for v in vars if v.name.startswith("F/")]

		self.opt = optimizer(self.mae_loss, self.params, learning_rate = self.lr)