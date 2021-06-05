# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import tensorflow as tf
import numpy as np


class PhysicsInformedNeuralNetwork:
    def __init__(self, layers, x_0, y_0, t_0, u_0, v_0, x_bc, y_bc, t_bc, u_bc, v_bc, x, y, t):
        self.layers = layers
        self.weights, self.biases = PhysicsInformedNeuralNetwork.initialize_nn(layers)

        self.x_0_tf = tf.keras.Input(dtype=tf.float32, shape=x_0.shape)
        self.y_0_tf = tf.keras.Input(dtype=tf.float32, shape=y_0.shape)
        self.t_0_tf = tf.keras.Input(dtype=tf.float32, shape=t_0.shape)
        self.u_0_tf = tf.keras.Input(dtype=tf.float32, shape=u_0.shape)
        self.v_0_tf = tf.keras.Input(dtype=tf.float32, shape=v_0.shape)

        self.x_bc_tf = tf.keras.Input(dtype=tf.float32, shape=x_bc.shape)
        self.y_bc_tf = tf.keras.Input(dtype=tf.float32, shape=y_bc.shape)
        self.t_bc_tf = tf.keras.Input(dtype=tf.float32, shape=t_bc.shape)
        self.u_bc_tf = tf.keras.Input(dtype=tf.float32, shape=u_bc.shape)
        self.v_bc_tf = tf.keras.Input(dtype=tf.float32, shape=v_bc.shape)

        self.x_tf = tf.keras.Input(dtype=tf.float32, shape=x.shape)
        self.y_tf = tf.keras.Input(dtype=tf.float32, shape=y.shape)
        self.t_tf = tf.keras.Input(dtype=tf.float32, shape=t.shape)

    @staticmethod
    def initialize_nn(layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for layer in range(0, num_layers - 1):
            w = PhysicsInformedNeuralNetwork.xavier_init(size=[layers[layer], layers[layer + 1]])
            b = tf.Variable(tf.zeros([1, layers[layer + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(w)
            biases.append(b)
        return weights, biases

    @staticmethod
    def xavier_init(size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, x, weights, biases):
        num_layers = len(weights) + 1

        h = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0  # normalize input
        for layer in range(0, num_layers - 2):
            w = weights[layer]
            b = biases[layer]
            hh = tf.tanh(tf.add(tf.matmul(h, w), b))
        w = weights[-1]
        b = biases[-1]
        y = tf.add(tf.matmul(hh, w), b)
        return y

    def net_uv(self, x, y, t):
        xx = tf.concat([x, y, t], 1)

        uv = PhysicsInformedNeuralNetwork.neural_net(xx, self.weights, self.biases)
        u = uv[:, 0:1]
        v = uv[:, 1:2]

        u_x = tf.gradients(u, x)[0]
        v_x = tf.gradients(v, x)[0]
        u_y = tf.gradients(u, y)[0]
        v_y = tf.gradients(v, y)[0]

        return u, v, u_x, v_x, u_y, v_y

    def net_f_uv(self, x, y, t):
        u, v, u_x, v_x, u_y, v_y = self.net_uv(x, y, t)

        u_t = tf.gradients(u, t)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]

        v_t = tf.gradients(v, t)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]

        f_u = u_t + 0.5 * (v_xx + v_yy)
        f_v = v_t - 0.5 * (u_xx + u_yy)

        return f_u, f_v


def main():
    t = np.linspace(0, 1, 100)

    x_left = np.linspace(-505, -5, 5000)
    y_left = np.linspace(-5, 5, 100)
    xx_left, yy_left, tt_left = np.meshgrid(x_left[1:], y_left[1:len(y_left) - 1], t[1:])

    x_mid = np.linspace(-5, 5, 100)
    y_mid = np.linspace(-10, 10, 200)
    xx_mid, yy_mid, tt_mid = np.meshgrid(x_mid[1:len(x_mid) - 1], y_mid[1:len(y_mid) - 1], t[1:])

    x_right = np.linspace(5, 405, 4000)
    y_right = np.linspace(-5, 5, 100)
    xx_right, yy_right, tt_right = np.meshgrid(x_right[:len(x_right) - 1], y_right[1:len(y_right) - 1], t[1:])

    xyt = np.stack((np.hstack((xx_left.flatten(), xx_mid.flatten(), xx_right.flatten())),
                    np.hstack((yy_left.flatten(), yy_mid.flatten(), yy_right.flatten())),
                    np.hstack((tt_left.flatten(), tt_mid.flatten(), tt_right.flatten()))))

    # boundary
    xb_1, yb_1, tb_1 = np.meshgrid(x_left, np.array([5]), t)
    xb_2, yb_2, tb_2 = np.meshgrid(np.array([-5]), np.linspace(5, 10, 50), t)
    xb_3, yb_3, tb_3 = np.meshgrid(x_mid, np.array([10]), t)
    xb_4, yb_4, tb_4 = np.meshgrid(np.array([5]), np.linspace(5, 10, 50), t)
    xb_5, yb_5, tb_5 = np.meshgrid(x_right, np.array([5]), t)
    xb_6, yb_6, tb_6 = np.meshgrid(x_right, np.array([-5]), t)
    xb_7, yb_7, tb_7 = np.meshgrid(np.array([5]), np.linspace(-10, -5, 50), t)
    xb_8, yb_8, tb_8 = np.meshgrid(x_mid, np.array([-10]), t)
    xb_9, yb_9, tb_9 = np.meshgrid(np.array([-5]), np.linspace(-10, -5, 50), t)
    xb_10, yb_10, tb_10 = np.meshgrid(x_left, np.array([-5]), t)

    xyt_bc = np.stack((np.hstack((xb_1.flatten(), xb_2.flatten(), xb_3.flatten(), xb_4.flatten(), xb_5.flatten(),
                                  xb_6.flatten(), xb_7.flatten(), xb_8.flatten(), xb_9.flatten(), xb_10.flatten())),
                       np.hstack((yb_1.flatten(), yb_2.flatten(), yb_3.flatten(), yb_4.flatten(), yb_5.flatten(),
                                  yb_6.flatten(), yb_7.flatten(), yb_8.flatten(), yb_9.flatten(), yb_10.flatten())),
                       np.hstack((tb_1.flatten(), tb_2.flatten(), tb_3.flatten(), tb_4.flatten(), tb_5.flatten(),
                                  tb_6.flatten(), tb_7.flatten(), tb_8.flatten(), tb_9.flatten(), tb_10.flatten()))))

    beta = 1
    u0 = np.exp(-yy_left ** 2 / 2) * np.exp(-(xx_left - 355) ** 2 / (2 * 20 ** 2)) * np.sin(beta * xx_left)
    v0 = np.exp(-yy_left ** 2 / 2) * np.exp(-(xx_left - 355) ** 2 / (2 * 20 ** 2)) * np.cos(beta * xx_left)
    u_0 = np.hstack((u0.flatten(), np.zeros(xx_mid.shape).flatten(), np.zeros(xx_right.shape).flatten()))
    v_0 = np.hstack((v0.flatten(), np.zeros(xx_mid.shape).flatten(), np.zeros(xx_right.shape).flatten()))

    N0 = 1000
    idx = np.random.choice(xyt.shape[1], N0, replace=False)

    # initial condition
    x_0 = xyt[0]
    y_0 = xyt[1]
    t_0 = np.zeros(xyt[2].shape)

    # boundary condition
    x_bc = xyt_bc[0]
    y_bc = xyt_bc[1]
    t_bc = xyt_bc[2]
    u_bc = np.zeros(xyt_bc[0].shape)
    v_bc = np.zeros(xyt_bc[0].shape)

    # differential equation
    x = xyt[0]
    y = xyt[1]
    t = xyt[2]

    nn_layer = [2, 100, 100, 100, 100, 2]
    network = PhysicsInformedNeuralNetwork(nn_layer, x_0, y_0, t_0, u_0, v_0, x_bc, y_bc, t_bc, u_bc, v_bc, x, y, t)


if __name__ == "__main__":
    main()
