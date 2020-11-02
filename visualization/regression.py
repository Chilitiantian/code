import sys
sys.path.append("../")
import numpy as np
import tensorflow as tf
from init.load_data import Dataset, Subset, One_hot_fea, Evalution
from init.config import parse_args
from init.model import DNN
from init.evalution import Ev_DL
from init.lr_decay import Piecewise_constant

def train_net(sess, model, x, y, lr):
    feed_dict = {model.x:x, model.y:y, model.lr:lr}
    loss, _ = sess.run([model.mae_loss, model.opt], feed_dict = feed_dict)
    return loss

def pre(sess, model, x):
    feed_dict = {model.x:x}
    pre = sess.run(model.pre, feed_dict = feed_dict)
    return pre

def main(args):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
    data = Dataset(args.data_path, args.fea_type, args.Periodic_table, "formulas", "formation_energies")
    args.fea_len = data[0][1].shape[-1]
    # print(args.fea_len)
    # exit()
    train, test = Subset(data, args.per)
    net = DNN(args, True)
    best_r = np.zeros([3])
    saver = tf.train.Saver()
    model_path = "../check_point/FCC/model.ckpt"
    boundaries = [30, 110, 200]
    lr = Piecewise_constant(boundaries, args.fcc_lr)
    with tf.Session(
            config = tf.ConfigProto(gpu_options = gpu_options)
        ) as sess:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()        
        while train.epochs != args.epochs:
            train_data = data[train.next_batch(args.batch_size)]
            train_x, train_y = train_data[1], train_data[-1]
            train_net(sess, net, train_x, train_y[:,np.newaxis], lr.updata(train.epochs))
            if train.loop_time % 20 == 0:
                ev = Ev_DL()
                while not test.end_one_epoch:
                    test_data = data[test.next_batch(args.batch_size)]
                    test_x, test_y = test_data[1], test_data[-1]
                    test_pre = pre(sess, net, test_x)
                    ev.updata(test_y, test_pre)
                test.end_one_epoch = False
                print(train.epochs, ev(), best_r)
                if ev.R() > best_r[-1]:
                    print("*"*20 + " save model " + "*"*20)
                    best_r = ev()
                    #saver.save(sess, model_path)
        all_test, all_pre = [], []
        while not test.end_one_epoch:
            test_data = data[test.next_batch(args.batch_size)]
            test_x, test_y = test_data[1], test_data[-1]
            test_pre = pre(sess, net, test_x)
            print(test_y.shape, test_pre.shape)
            exit()
            all_test.append(test_y)
            all_pre.append(test_pre)
        np.concatenate([all_test, all_pre], axis = 1)

if __name__ == "__main__":
    main(parse_args())   
