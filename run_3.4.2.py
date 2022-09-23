# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

import argparse
import os
import shutil
import sys
import time
from typing import List, Any

import matplotlib.pyplot as plt
import numpy as np
import paddle
import paddle.nn as nn
import paddle.static as static

import visual_data
from basic_model_pdpd import DeepModel_single
from SALib.sample import sobol_sequence

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
pi = np.pi


def get_args():
    parser = argparse.ArgumentParser('PINNs for Burgers2', add_help=False)
    parser.add_argument('-f', type=str, default="external")
    parser.add_argument('--net_type', default='pinn_rar', type=str)
    parser.add_argument('--epochs_adam', default=50000, type=int)
    parser.add_argument('--save_freq', default=1000, type=int, help="frequency to save model and image")
    parser.add_argument('--print_freq', default=200, type=int, help="frequency to print loss")
    parser.add_argument('--device', default=True, type=bool, help="use gpu")
    parser.add_argument('--work_name', default='Burgers_2D', type=str, help="save_path")

    parser.add_argument('--Nx_EQs', default=1500, type=int)
    parser.add_argument('--Nt_Val', default=100, type=int)
    parser.add_argument('--Nx_Val', default=256, type=int)
    parser.add_argument('--Nx_Add', default=100000, type=int)
    parser.add_argument('--samp_ids', default=0, type=int)
    parser.add_argument('--g_weight', default=0.0001, type=float)
    return parser.parse_args()


class Net_single(DeepModel_single):
    def __init__(self, planes, active):
        super(Net_single, self).__init__(planes, active=active, data_norm=[0, 0])

        self.pick_1 = paddle.to_tensor([[1, 0], ], dtype=paddle.float32).T
        self.pick_2 = paddle.to_tensor([[0, 1], ], dtype=paddle.float32).T

    def out_transform(self, inn_var, out_var):
        x_in = paddle.matmul(inn_var, self.pick_1)
        t_in = paddle.matmul(inn_var, self.pick_2)
        return out_var * (1 - x_in) * (1 + x_in) * (1 - paddle.exp(-t_in)) - paddle.sin(pi * x_in)

    def equation(self, inn_var):
        out_var = self.forward(inn_var)
        out_var = self.out_transform(inn_var, out_var)
        duda = paddle.incubate.autograd.grad(out_var, inn_var)
        dudx, dudt = duda[:, 0:1], duda[:, 1:2]
        Ddudx = paddle.incubate.autograd.grad(dudx, inn_var)
        d2udx2 = Ddudx[:, 0:1]
        # eqs = 0.
        eqs = dudt + out_var * dudx - 0.01 / pi * d2udx2
        if 'gpinn' in opts.net_type:
            Dd2udx2 = paddle.incubate.autograd.grad(d2udx2, inn_var)
            Ddudt = paddle.incubate.autograd.grad(dudt, inn_var)

            d2udtx, d2udt2 = Ddudt[:, 0:1], Ddudt[:, 1:2]
            d3udx3, d3udx2t = Dd2udx2[:, 0:1], Dd2udx2[:, 1:2]

            g_eqs = [d2udtx + (dudx * dudx + out_var * d2udx2) - 0.01 / pi * d3udx3,
                     d2udt2 + (dudt * dudx + out_var * d2udtx) - 0.01 / pi * d3udx2t, ]
            # g_eqs = paddle.concat(g_eqs, axis=-1)
        else:
            g_eqs = paddle.zeros((2,), dtype=paddle.float32)

        return out_var, eqs, g_eqs


def build(opts, model):
    ## 采样

    EQs_var = paddle.static.data('EQs_var', shape=[opts.Nx_EQs + 10 * opts.samp_ids, 2], dtype='float32')
    EQs_var.stop_gradient = False
    # EQs_tar = paddle.static.data('EQs_tar', shape=[opts.Nx_EQs, 1], dtype='float32')

    Val_var = paddle.static.data('Val_var', shape=[opts.Nx_Val * opts.Nt_Val, 2], dtype='float32')
    Val_var.stop_gradient = False
    Val_tar = paddle.static.data('Val_tar', shape=[opts.Nx_Val * opts.Nt_Val, 1], dtype='float32')

    Add_var = paddle.static.data('Add_var', shape=[opts.Nx_Add, 2], dtype='float32')
    Add_var.stop_gradient = False

    _, eqs, g_eqs = model.equation(EQs_var)
    val, eqs_v, _ = model.equation(Val_var)
    _, eqs_a, _ = model.equation(Add_var)
    eqs_a = paddle.abs(eqs_a)


    EQsLoss = paddle.norm(eqs, p=2) ** 2 / opts.Nx_EQs  # 方程所有计算守恒残差点的损失，不参与训练
    gEQsLoss = paddle.norm(g_eqs[0], p=2) ** 2 / opts.Nx_EQs + paddle.norm(g_eqs[1], p=2) ** 2 / opts.Nx_EQs
    datLoss = paddle.norm(Val_tar - val, p=2) ** 2 / opts.Nt_Val / opts.Nx_Val  # 方程所有计算守恒残差点的损失，不参与训练

    total_loss = EQsLoss + gEQsLoss * opts.g_weight

    scheduler = paddle.optimizer.lr.MultiStepDecay(0.001, [opts.epochs_adam*0.6, opts.epochs_adam*0.8], gamma=0.1)
    optimizer = paddle.optimizer.Adam(scheduler)
    # optimizer = paddle.incubate.optimizer.functional.minimize_lbfgs(func, x0)
    optimizer.minimize(total_loss)
    # optimizer.minimize(EQsLoss)
    #
    return [val, eqs_v, eqs_a], [EQsLoss, gEQsLoss, datLoss, total_loss], scheduler


def gen_testdata():
    data = np.load("Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.concatenate([xx[..., None], tt[..., None]], axis=2)

    return X.astype(np.float32), exact[..., None].astype(np.float32)


def gen_traindata(N, method='uniform'):
    if method == 'uniform':
        Nt = int((N / 2) ** 0.5)
        Nx = 2 * Nt
        x = np.linspace(-1, 1, Nx, endpoint=True)
        t = np.linspace(0, 1, Nt, endpoint=True)
        xx, tt = np.meshgrid(x, t)
    elif method == 'sobol':
        # n = int(N*0.05)
        a = sobol_sequence.sample(N, 2)
        xx = a[:, 0:1] * 2 - 1
        tt = a[:, 1:2]
    else:
        n = int(N * 0.05)
        xx = np.random.random(N) * 2 - 1
        tt = np.random.random(N)
        xx = np.concatenate([xx, np.ones((n, 1)) * -1, np.ones((n, 1)) * 1, np.random.random(n) * 2 - 1])
        tt = np.concatenate([tt, np.random.random(n), np.random.random(n), np.zeros((n, 1)) * 1])

    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    return X.astype(np.float32)


if __name__ == '__main__':


    opts = get_args()
    print("start sampling process {:3d}".format(opts.samp_ids))
    print(opts)

    try:
        import paddle.fluid as fluid
        use_cuda = opts.device
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    except:
        place = None
    # compiled_program = static.CompiledProgram(static.default_main_program())

    paddle.enable_static()

    # opts.Nx_EQs = N
    save_path = opts.net_type + '-Nx_EQs_' + str(opts.Nx_EQs)
    work_path = os.path.join('work', opts.work_name, save_path)
    tran_path = os.path.join('work', opts.work_name, save_path, 'train')
    isCreated = os.path.exists(tran_path)
    if not isCreated:
        os.makedirs(tran_path)

    # 将控制台的结果输出到a.log文件，可以改成a.txt
    sys.stdout = visual_data.Logger(os.path.join(work_path, 'train.log'), sys.stdout)

    train_x = gen_traindata(opts.Nx_EQs, method='sobol')  # 生成监督测点
    add_x = gen_traindata(opts.Nx_Add, method='sobol')
    valid_x, valid_u = gen_testdata()
    valid_x, valid_u = valid_x.reshape((-1, 2)), valid_u.reshape((-1, 1))


    paddle.incubate.autograd.enable_prim()

    planes = [2, ] + [32, ] * 3 + [1, ]
    Net_model = Net_single(planes=planes, active=nn.Tanh())
    [U_pred, R_pred, R_add], Loss, Scheduler = build(opts, Net_model)

    Visual = visual_data.matplotlib_vision('/', field_name=('u',), input_name=('t', 'x'))
    star_time = time.time()
    log_loss = []
    start_epoch = 0

    exe = static.Executor(place)
    exe.run(static.default_startup_program())
    prog = static.default_main_program()


    if opts.samp_ids > 0:
        params = paddle.load(os.path.join(work_path, 'latest_model.pdparams'))
        prog.set_state_dict(params)
        train_log = paddle.load(os.path.join(work_path, 'train_log.pth'))
        log_loss, train_x = train_log['log_loss'], train_log['train_x']
        temp = np.concatenate([train_x, add_x[:10]], axis=0)
        [add_r, ] = exe.run(prog, feed={'EQs_var': temp, 'Add_var': add_x, 'Val_var': valid_x, 'Val_tar': valid_u},
                            fetch_list=[R_add, ])

        ids_x = np.argsort(-add_r.squeeze(), axis=0)[:10]
        train_x = np.concatenate([train_x, add_x[ids_x]], axis=0)

        plt.figure(200, figsize=(12, 10))
        plt.clf()
        plt.scatter(train_x[:-10, 0], train_x[:-10, 1], c='b')
        plt.scatter(train_x[-10:, 0], train_x[-10:, 1], c='r')
        plt.xlabel("x")
        plt.ylabel("t")
        plt.tight_layout()
        plt.savefig(os.path.join(tran_path, 'add_points' + str(opts.samp_ids) + '.svg'))

    for epoch in range(start_epoch, 1+opts.epochs_adam):
        ## 采样

        Scheduler.step()
        exe.run(prog, feed={'EQs_var': train_x, 'Add_var': add_x, 'Val_var': valid_x, 'Val_tar': valid_u},
                fetch_list=[Loss[-1]])

        if epoch > 0 and epoch % opts.print_freq == 0:
            all_items = exe.run(prog, feed={'EQs_var': train_x, 'Add_var': add_x, 'Val_var': valid_x, 'Val_tar': valid_u},
                                fetch_list=[[U_pred, R_pred] + Loss[:-1]])

            x_true = valid_x.reshape((opts.Nt_Val, opts.Nx_Val, 2))
            u_true = valid_u.reshape((opts.Nt_Val, opts.Nx_Val, 1))
            u_pred = all_items[0].reshape((opts.Nt_Val, opts.Nx_Val, 1))
            r_pred = all_items[1].reshape((opts.Nt_Val, opts.Nx_Val, 1))
            loss_items = all_items[2:]

            log_loss.append(np.array(loss_items).squeeze())
            # print(loss_items[1:])

            print('iter: {:6d}, lr: {:.1e}, cost: {:.2f}, val_loss: {:.2e}, EQs_loss: {:.2e}, Grad_loss: {:.2e}'.
                  format(epoch, Scheduler.get_lr(), time.time() - star_time, float(loss_items[-1]),
                         float(loss_items[0]), float(loss_items[1])))
            star_time = time.time()

        if epoch > 0 and epoch % opts.save_freq == 0:

            # print(np.array(par_pred).shape)
            plt.figure(100, figsize=(12, 10))
            plt.clf()
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, -1], 'dat_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 0], 'eqs_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 1], 'geqs_loss')
            plt.rcParams['font.size'] = 20
            plt.tight_layout()
            plt.savefig(os.path.join(tran_path, 'log_loss.svg'))

            plt.figure(1, figsize=(25, 8))
            plt.clf()
            Visual.plot_fields_ms(u_true, u_pred, x_true, )
            plt.xlabel("x")
            plt.ylabel("t")
            plt.tight_layout()
            plt.savefig(os.path.join(tran_path, 'pred_u' + str(opts.samp_ids) + '.jpg'))

            err = u_pred - u_true
            plt.figure(2, figsize=(12, 10))
            plt.clf()
            plt.pcolormesh(x_true[..., 0], x_true[..., 1], err[..., 0], cmap='viridis',
                           shading='gouraud', antialiased=True, snap=True)
            plt.xlabel("x")
            plt.ylabel("t")
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=20)  # 设置色标刻度字体大小
            plt.tight_layout()
            plt.savefig(os.path.join(tran_path, 'err_u' + str(opts.samp_ids) + '.jpg'))

            eqs = np.abs(r_pred)
            plt.figure(3, figsize=(12, 10))
            plt.clf()
            plt.pcolormesh(x_true[..., 0], x_true[..., 1], r_pred[..., 0], cmap='viridis', shading='gouraud',
                           antialiased=True, snap=True)
            plt.xlabel("x")
            plt.ylabel("t")
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=20)  # 设置色标刻度字体大小
            plt.tight_layout()
            plt.savefig(os.path.join(tran_path, 'err_eqs' + str(opts.samp_ids) + '.jpg'))

            star_time = time.time()

            paddle.save({'log_loss': log_loss, 'train_x': train_x}, os.path.join(work_path, 'train_log.pth'))
            paddle.save({'valid_x': valid_x, 'valid_u': valid_u,
                         'u_pred': u_pred, 'r_pred': r_pred, },
                        os.path.join(work_path, 'out_res' + str(opts.samp_ids) + '.pth'), )

    paddle.save(prog.state_dict(), os.path.join(work_path, 'latest_model.pdparams'), )
    time.sleep(3)
    shutil.move(os.path.join(work_path, 'train.log'), tran_path)
