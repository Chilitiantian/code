import numpy as np
import tensorflow as tf
from init.load_data import Dataset, Subset, One_hot_fea, Evalution
from init.config import parse_args
from init.model import GAN

def train_discriminator(sess, model, random, real_data):
    feed_dict = {model.random : random, model.real_data : real_data}
    d_loss, _ = sess.run([model.disc_cost, model.opt_d], feed_dict = feed_dict)	
    return d_loss

def train_generator(sess, model, random):
    feed_dict = {model.random : random}
    g_loss, _ = sess.run([model.gen_cost, model.opt_g], feed_dict = feed_dict)	
    return g_loss

def test_discriminator(sess, model, random, real_data):
    feed_dict = {model.random : random, model.real_data : real_data}
    d_loss = sess.run(model.disc_cost, feed_dict = feed_dict)
    return d_loss

def main(args):	
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
    one_hot_fea = One_hot_fea(args.Periodic_table)
    data = Dataset(args.data_path, args.fea_type, args.Periodic_table)
    train, test = Subset(data, args.per)
    args.atom_num, args.max_idx = data[0][1].shape[1], data[0][1].shape[-1]
    net = GAN(args, one_hot_fea, is_train = True)
    saver = tf.train.Saver()
    model_path = "check_point/" + args.database + "/model.ckpt"
    best_disc_loss = 1e3
    file = open("d_loss.csv", "w")
    with tf.Session(
            config = tf.ConfigProto(gpu_options = gpu_options)
        ) as sess:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        # saver.restore(sess, model_path)
        ev = Evalution()
        while train.epochs != args.epochs:
            
            for i in range(args.disc_iters):
                random = np.random.normal(0, 1, [args.batch_size, args.random_len])
                real_data = data[train.next_batch(args.batch_size)][1][:,:,:,np.newaxis]
                d_loss = train_discriminator(sess, net, random, real_data)

            random = np.random.normal(0, 1, [args.batch_size, args.random_len])
            g_loss = train_generator(sess, net, random)

            if train.loop_time % 10 == 0:
                ev = Evalution()
                while not test.end_one_epoch:
                    random = np.random.normal(0, 1, [args.batch_size, args.random_len])
                    real_data = data[test.next_batch(args.batch_size)][1][:,:,:,np.newaxis]
                    d_loss = test_discriminator(sess, net, random, real_data)
                    ev.updata(d_loss)
                test.end_one_epoch = False
                print(train.loop_time, ev.total_loss)
                print(train.loop_time, ev.total_loss, file = file)
                # print(train.epochs, "g_loss = ", g_loss, "d_loss = ", d_loss, "test_d_loss = ", ev.total_loss)
                # if ev.total_loss < best_disc_loss:
                #     print("*"*20 + " save model " + "*"*20)
                #     best_disc_loss = ev.total_loss
                #     saver.save(sess, model_path)

if __name__ == "__main__":
    main(parse_args())
