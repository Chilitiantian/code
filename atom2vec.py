import numpy as np
import tensorflow as tf
from init.load_data import Dataset, Subset, One_hot_fea, Evalution
from init.config import parse_args
from init.model import GAN

def get_atom_vec(sess, model, random, real_data):
    feed_dict = {model.random : random, model.real_data : real_data}
    atom2vec = sess.run(model.atom2vec, feed_dict = feed_dict)
    return atom2vec

def main(args):

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
    one_hot_fea = One_hot_fea(args.Periodic_table)
    data = Dataset(args.data_path, args.fea_type, args.Periodic_table)
    args.atom_num, args.max_idx = data[0][1].shape[1], data[0][1].shape[-1]

    net = GAN(args, one_hot_fea, is_train = False)
    saver = tf.train.Saver()
    model_path = "check_point/" + args.database + "/model.ckpt"
    atom2vec_path = "atom2vec/" + args.database + "/atom2vec.csv"
    with tf.Session(
            config = tf.ConfigProto(gpu_options = gpu_options)
        ) as sess:
        saver.restore(sess, model_path)
        random = np.random.normal(0, 1, [args.batch_size, args.random_len])
        real_data = data[:args.batch_size][1][:,:,:,np.newaxis]
        atomfea = get_atom_vec(sess, net, random, real_data)
    atoms = np.loadtxt(args.Periodic_table, dtype = str)[:, np.newaxis]
    atom2vec = np.concatenate([atoms, atomfea], axis = 1)
    print(atom2vec)    
    np.savetxt(atom2vec_path, atom2vec, fmt = "%s", delimiter = ",")

if __name__ == "__main__":
    main(parse_args())
