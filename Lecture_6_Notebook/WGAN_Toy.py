import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
tf.disable_v2_behavior()


class mydata:
    def __init__(self, mode, real_dim):
        self.mode = mode
        self.real_dim = real_dim

    def next_batch(self, batchsize):
        if self.mode == "uni":
            x = np.random.uniform(-0.7, 0.7, (self.real_dim, batchsize))
        elif self.mode == "1d":
            z = np.random.uniform(-2, 2, (1, batchsize))
            x = []
            for i in range(self.real_dim):
                x.append(np.sin(z + i))
            x = np.concatenate(x, axis=0)
        else:
            print("Not Implemented")
        return x


def feed_NN(X, W, b):
    A = X
    L = len(W)
    for i in range(L - 1):
        A = tf.nn.relu(tf.add(tf.matmul(W[i], A), b[i]))
    return tf.add(tf.matmul(W[-1], A), b[-1])


def discriminator(x, W, b):
    y = feed_NN(x, W, b)
    return y


def generator(z, W, b):
    y = feed_NN(z, W, b)
    return y


def runner(mode, z_dim):
    savedir = mode + str(z_dim)
    try:
        os.stat(savedir)
    except:
        os.mkdir(savedir)

    tf.set_random_seed(1)
    np.random.seed(1)

    real_dim = 10
    lamda = 0.1
    batch_size = 256

    Data = mydata(mode, real_dim)
    np.savetxt(f"{mode}.csv", Data.next_batch(1000).transpose(), delimiter=",")

    tf.reset_default_graph()

    layer_dims = [real_dim] + 8 * [64] + [1]
    L = len(layer_dims)
    D_W = [
        tf.get_variable(
            "D_W" + str(l),
            [layer_dims[l], layer_dims[l - 1]],
            initializer=tf.keras.initializers.glorot_normal(),
        )
        for l in range(1, L)
    ]
    D_b = [
        tf.get_variable(
            "D_b" + str(l), [layer_dims[l], 1], initializer=tf.zeros_initializer()
        )
        for l in range(1, L)
    ]

    layer_dims = [z_dim] + 8 * [64] + [real_dim]
    L = len(layer_dims)
    G_W = [
        tf.get_variable(
            "G_W" + str(l),
            [layer_dims[l], layer_dims[l - 1]],
            initializer=tf.keras.initializers.glorot_normal(),
        )
        for l in range(1, L)
    ]
    G_b = [
        tf.get_variable(
            "G_b" + str(l), [layer_dims[l], 1], initializer=tf.zeros_initializer()
        )
        for l in range(1, L)
    ]

    z_disc = tf.placeholder(tf.float32, [z_dim, None])
    e_unif = tf.placeholder(tf.float32, [1, None])

    real_data = tf.placeholder(tf.float32, [real_dim, None])
    fake_data = generator(z_disc, G_W, G_b)

    disc_real = discriminator(real_data, D_W, D_b)
    disc_fake = discriminator(fake_data, D_W, D_b)

    D_vars = D_W + D_b
    G_vars = G_W + G_b

    interpolates = e_unif * real_data + (1 - e_unif) * fake_data
    disc_interpolates = discriminator(interpolates, D_W, D_b)
    gradients = tf.gradients(disc_interpolates, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[0]))
    gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)

    # GAN
    # disc_loss = - tf.reduce_mean(tf.log(1 - disc_fake)) - tf.reduce_mean(tf.log(disc_real))
    # WGAN
    disc_loss = (
        tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real) + lamda * gradient_penalty
    )

    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, beta1=0.5, beta2=0.9
    ).minimize(disc_loss, var_list=D_vars)

    # Generator for training G_vars
    z_g = tf.placeholder(tf.float32, [z_dim, None])
    generated_data = generator(z_g, G_W, G_b)

    D_g = discriminator(generated_data, D_W, D_b)
    # GAN
    # gen_loss = tf.reduce_mean(tf.log(1 - D_g))
    # WGAN
    gen_loss = -tf.reduce_mean(D_g)

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, beta1=0.5, beta2=0.9
    ).minimize(gen_loss, var_list=G_vars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    step = 0

    iterations = 50001
    for i in range(iterations):
        if step % 100 == 0:
            plotbatch = Data.next_batch(1000)
            fig = plt.figure()
            plt.scatter(
                plotbatch[0, :],
                plotbatch[1, :],
                marker="o",
                color="k",
                label="Real Data",
                s=2,
            )
            z_plt = np.random.normal(0, 1, size=[z_dim, 1000])
            G_plt = sess.run(generated_data, feed_dict={z_g: z_plt})
            plt.scatter(
                G_plt[0, :],
                G_plt[1, :],
                marker="o",
                color="r",
                label="Generated Data",
                s=2,
            )
            plt.xlim((-1.5, 1.5))
            plt.ylim((-1.5, 1.5))
            plt.gca().set_aspect("equal", adjustable="box")
            plt.title("Step: " + str(step))
            plt.legend()
            fig.savefig(savedir + "/" + str(step) + ".png", dpi=fig.dpi)
            np.savetxt(
                savedir + "/" + str(step) + ".csv", G_plt.transpose(), delimiter=","
            )

        for _ in range(5):
            z_disc_data = np.random.normal(0, 1, size=[z_dim, batch_size])
            e_unif_data = np.random.uniform(0, 1, size=[1, batch_size])
            real_data_data = Data.next_batch(batch_size)

            sess.run(
                disc_train_op,
                feed_dict={
                    z_disc: z_disc_data,
                    e_unif: e_unif_data,
                    real_data: real_data_data,
                },
            )

        z_g_data = np.random.normal(0, 1, size=[z_dim, batch_size])
        sess.run(gen_train_op, feed_dict={z_g: z_g_data})
        step += 1


runner("uni", 10)
runner("uni", 1)
runner("1d", 10)
runner("1d", 1)
