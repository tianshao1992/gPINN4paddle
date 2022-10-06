# Paddle 论文复现挑战赛第七期
—— 科学计算方向 51

原文：[Gradient-enhanced physics-informed neural networks for forward and inverse PDE problems](https://doi.org/10.1016/j.cma.2022.114823)

参考：[GPINN](https://github.com/lu-group/gpinn)

- 近年来，大量文献研究表明PINN通过将PDE残差嵌入神经网络的损失函数中可以成功解决各类正向和反向问题。但是，即使采用在求解域充分的采样点，其准确度仍有限。因此，该文章提出了一种具有梯度增强的物理知识融合网络GPINNs，同时提高训练效率和网络准确性。

- 在GPINNs的实现过程中，作者将PDE残差的梯度信息作为一个加权损失函数补充训练。相对于传统的PINNs，作者处理的技巧主要包括了两点（GPINNs——Gradient-enhanced physics-informed neural networks）：

  - 在损失函数设计，GPINNs引入了PDE残差对各个自变量的梯度残差损失，并采用加权损失函数。
  - 所有问题中均引入了[hard constraints](https://epubs.siam.org/doi/10.1137/21M1397908)约束神经网络输出，使得边界条件和初始条件天然满足，降低了神经网络优化的难度。
  - 在训练过程中，先以大量的残差样本点作为初始训练集，在训练过程中，通过不断寻找PDE残差较大的区域进行重采样进行补充训练集——即类似主动学习的策略对PINNs进行训练。


## 2.代码说明

1. [gPINNs4paddle AI studio](https://aistudio.baidu.com/aistudio/projectdetail/4493662)相关运行结果

  - 脚本文件包括

    - run_3.2.1.py、run_3.2.2.py——实际对应原文中3.2部分代码。
    - run_3.3.1.py、run_3.3.2.py——实际对应原文中3.3.1部分代码。
    - run_3.4.1.py、run_3.4.2.py——实际对应原文中3.4部分代码。
    - 其中每个脚本都进行了如下的参数设置：

    ```python
    def get_args():
        parser = argparse.ArgumentParser('PINNs for Poisson2D', add_help=False)
        parser.add_argument('-f', type=str, default="external")
        parser.add_argument('--net_type', default='gpinn', type=str)
        parser.add_argument('--epochs_adam', default=20000, type=int)
        parser.add_argument('--save_freq', default=5000, type=int, help="frequency to save model and image")
        parser.add_argument('--print_freq', default=1000, type=int, help="frequency to print loss")
        parser.add_argument('--device', default=True, type=bool, help="use gpu")
        parser.add_argument('--work_name', default='Poisson-2D', type=str, help="save_path")
    
        parser.add_argument('--Nx_EQs', default=100, type=int)
        parser.add_argument('--Nt_Val', default=100, type=int)
        parser.add_argument('--Nx_Val', default=100, type=int)
        parser.add_argument('--g_weight', default=0.01, type=float)
        return parser.parse_args()
    
    ```

    其中，net_type可供选择采用gPINN模式或pinn模式；

    epochs_adam为采用adam优化器进行训练的总步数；

    save_freq、print_freq为打印残差以及输出中间可视化结果的频率；

    device为是否使用gpu并行；

    work_name为保存路径。

    Nx_EQs为训练PDE残差的采样点个数；

    Nt_Val、Nx_Val分别指代采用验证数据时在时间t（如果存在）以及x方向的采样点个数（均匀网格）；

    g_weight为gradient ehanced的残差权重，即下式的$w_{g_i}$。
    $
    \mathcal{L}=w_f \mathcal{L}_f+w_b \mathcal{L}_b+w_i \mathcal{L}_i+\sum_{i=1}^d w_{g_i} \mathcal{L}_{g_i}\left(\boldsymbol{\theta} ; \mathcal{T}_{g_i}\right)
    $
    其中，$L_f、L_b、L_i$为PDE残差、边界条件、初始条件损失，权重均默认为1，后两者由于采用了hard constraints因此均为0。

  - **fig文件夹**中为原始论文结果相关图片以及作者复现所整理的对应结果，**work文件夹**中为训练过程及中间结果

    - Poisson-1D为3.2.1中算例对应结果文件存放位置，用于绘制Figure2。
    - Poisson-2D为3.2.2中算例对应结果文件存放位置，用于绘制Figure3。
    - Brinkman-Forchheimer-1为3.3.1中算例对应结果文件存放位置，用于绘制Figure 6，反问题中预测一个参数。
    - Brinkman-Forchheimer-2为3.3.1中算例对应结果文件存放位置，用于绘制Figure 7，反问题中预测二个参数。
    - Buergers-2D为3.4中算力对应结果文件存放位置，用于绘制Figure 10、11、12。

  - basic_model.py 中为实现的全连接神经网络结构

  - process_data.py 中为求解域离散点采样方法

  - visual_data.py 为训练过程的数据可视化

## 3.环境依赖

  > numpy == 1.22.3 \
  > paddlepaddle-gpu develop \
  > matplotlib==3.5.1 \
  > seaborn==0.11.2 

## 4.复现结果

### 4.1 正向问题 

1).原始方程为1-D Poisson方程：

$
-\Delta u=\sum_{i=1}^4 i \sin (i x)+8 \sin (8 x), \quad x \in[0, \pi]
$

方程解析解为

$
u(x)=x+\sum_{i=1}^4 \frac{\sin (i x)}{i}+\frac{\sin (8 x)}{8}
$

损失函数为

$
\mathcal{L}=\mathcal{L}_f+w \mathcal{L}_g
$

1D Possion （对应文章 3.2.1 Figure 2）详细代码见run_3.2.1.py 以及train_3.2.1.sh 设置不同的traning points——Nx_EQs以及权重w。（误差带为运行10次求取均值以及方差绘制，以下类似。）
下表详细展示了采用GPINN对1D Possion问题的预测效果，以及不同权重、不同训练点数量对于GPINN的影响。其中，左侧为本次复现结果，而右侧为论文结果。需要指出，复现的结果中PINNs以及gPINNs均较原始论文更好，其中GPINNs 权重为1.0时效果尤其明显。



|      |  复现  | 论文 |
| :--: | :------------: | :------: |
| Figure 2 A |![](https://ai-studio-static-online.cdn.bcebos.com/2b6d12bb61a942669cb6e3e1d7646468f4d0be21ea094ca5a924959bea60f617)|![](https://ai-studio-static-online.cdn.bcebos.com/e388ebec43184e6cb753531f8ffda619ce1606d38c67417da736ec73bffa4c56)|
| Figure 2 B |![](https://ai-studio-static-online.cdn.bcebos.com/1d49151e46f743718ed6a79f8358e56a5fdbaf289c85446dac21ad87d03ee128)|![](https://ai-studio-static-online.cdn.bcebos.com/5dc2d060cd3b4cf1aedf6eaa7314756361899eaa54b34244b89b60a0ff1c8e7c)|
| Figure 2 C |![](https://ai-studio-static-online.cdn.bcebos.com/171bd98c445a4af99c3e31b6148eb517f795b2047d8640098d66d291312a8f4c)|![](https://ai-studio-static-online.cdn.bcebos.com/0ff6140f3d65448089e92cc224911942fae9b4cc1147497cb25133fa59ce61d9)|
| Figure 2 D |![](https://ai-studio-static-online.cdn.bcebos.com/28f5843feadc4fd188fcbe643e11899d9859a9b3d40644a3896f0f5d948f8608)|![](https://ai-studio-static-online.cdn.bcebos.com/0fca0a25e6a543d095cb67ebc12fc386386e6fd93bf84ccbb976ce2a7cb0f09f)|
| Figure 2 E |![](https://ai-studio-static-online.cdn.bcebos.com/967d1c6a143c4d83b558a15808216eccd3ea908bd3e944a598c13bf0f68ed72d)|![](https://ai-studio-static-online.cdn.bcebos.com/24692bfdcc6f4fb19898554feaefd111d234386f669e4b4ea13a1ce68e542532)|
| Figure 2 F |![](https://ai-studio-static-online.cdn.bcebos.com/244eeff8c354471092333d57dfb8d2d5fa96706024cd43708a677725252e6c74)|![](https://ai-studio-static-online.cdn.bcebos.com/641292b71d264b98b024261112fa1356296a82eae8e84bd29ed6a2a02159756f)|
| Figure 2 G |![](https://ai-studio-static-online.cdn.bcebos.com/eff151ebeed046d28910628ffda742397a37f0ccdcb9437cb02f10bde8ab2497)|![](https://ai-studio-static-online.cdn.bcebos.com/956db99226804e01aa99a0d871ab395dca79027a48cc4cbb885ad779c48dd960)|

2).原始方程为2-D Poisson方程：

$
\frac{\partial u}{\partial t}=D \frac{\partial^2 u}{\partial x^2}+R(x, t), \quad x \in[-\pi, \pi], t \in[0,1]
$

其中源项：

$
R(x, t)=e^{-t}\left[\frac{3}{2} \sin (2 x)+\frac{8}{3} \sin (3 x)+\frac{15}{4} \sin (4 x)+\frac{63}{8} \sin (8 x)\right]
$

方程解析解为

$
u(x, t)=e^{-t}\left[\sum_{i=1}^4 \frac{\sin (i x)}{i}+\frac{\sin (8 x)}{8}\right]
$

损失函数为

$
\mathcal{L}=\mathcal{L}_f+w \mathcal{L}_{g_x}+w \mathcal{L}_{g_t}
$


2D Possion （对应文章 3.2.2），详细代码见run_3.2.2.py 以及train_3.2.2.sh 设置不同的Nx_EQs以及权重w。


|      |  复现  | 论文 |
| :--: | :------------: | :------: |
| Figure 3 A |![](https://ai-studio-static-online.cdn.bcebos.com/803be944194e4460b4f6918f9d3a67832eca00a218af4ac0bec0601a71f7fa8c)|![](https://ai-studio-static-online.cdn.bcebos.com/91c2d5958e9f4f3095f1653c4ebca2c232e05caf5c05400bab1dd45a744dd46c)|
| Figure 3 B |![](https://ai-studio-static-online.cdn.bcebos.com/5243ea884e754d25818b6b401aa2adaf1971fe72b2674822934d517486f6d4cc)|![](https://ai-studio-static-online.cdn.bcebos.com/491a8266edec44a8a546a010a9ba0363d85e9118d7124de980d542cd6e54d797)|
| Figure 3 C |![](https://ai-studio-static-online.cdn.bcebos.com/2b9921cdfea843cd8fc598c069ab128aeb7642a2a7424efa9b022f0fc19bec9f)|![](https://ai-studio-static-online.cdn.bcebos.com/58e961f4bb1e417f93b9edde171479404f6bd42d67f245d390eb591975270178)|
| Figure 3 D |![](https://ai-studio-static-online.cdn.bcebos.com/497212ad54c64910ab9373b5f585e919466d4a5b3f28467f8713e0d38b204795)|![](https://ai-studio-static-online.cdn.bcebos.com/b29e60792dd3443b8f1f8edc379ac5ba39319f78eb474e8fa3bebb9e7a7191e7)|


### 4.2 反向问题 
原始方程为1-DBrinkman-Forchheimer方程：
$$
-\frac{\nu_e}{\epsilon} \nabla^2 u+\frac{\nu u}{K}=g, \quad x \in[0, H]
$$
解析解为：
$$
u(x)=\frac{g K}{\nu}\left[1-\frac{\cosh \left(r\left(x-\frac{H}{2}\right)\right)}{\cosh \left(\frac{r H}{2}\right)}\right]
$$
此外，本问题中还需识别模型的粘度$\nu_e$以及渗透性$K$。

1).仅预测Brinkman-Forchheimer 模型的粘度$\nu_e$，详细代码见run_3.3.1.py 以及train_3.3.1.sh 设置不同的traning points——Nx_EQs。

需要指出复现结果中，gPINN相对于PINN在训练样本点较少时优势明显，但随着训练样本增多时，gPINN优势并不明显。

|      |  复现  | 论文 |
| :--: | :------------: | :------: |
| Figure 6 A |![](https://ai-studio-static-online.cdn.bcebos.com/fe34ded30fa9401799a77c104a9b8a1b95cb501089dd4c019833fa8aea5b1911#pic_center=100x100)|![](https://ai-studio-static-online.cdn.bcebos.com/002f8297e8d8484c89cd06f57ae4110b611f5703d50e44558bbf3d0ccd6c545c)|
| Figure 6 B |![](https://ai-studio-static-online.cdn.bcebos.com/3988dad934fd42bea1fdf518fc011770f92e263a83f042bc8d23273e48012ae3#pic_center=100x100)|![](https://ai-studio-static-online.cdn.bcebos.com/0229368a271a4d069bacffa8871ad4f0b509ea4c02dc42d397f3b3354d8c6912)|
| Figure 6 C |![](https://ai-studio-static-online.cdn.bcebos.com/dc136d8d6c3d412ba6668b7d21ca923c07007324d45a4df390ea0f9c4d89ac52#pic_center=100x100)|![](https://ai-studio-static-online.cdn.bcebos.com/cc8bbc4a315a42db84af156723999c8a9ed6f8fef44e44328e4a6f9b150dee83)|
| Figure 6 D |![](https://ai-studio-static-online.cdn.bcebos.com/ebc45345c8eb45f39562e917d274e06c19e8e2a06fc147a59b991333ba337153#pic_center=100x100)|![](https://ai-studio-static-online.cdn.bcebos.com/da17a7d7b49c4b459b0a155168c51b95ee5caf83eb3f4d709c125d950f324b8e)|
| Figure 6 E |![](https://ai-studio-static-online.cdn.bcebos.com/7e0cdcc47e804d9abea13d88d623e03cca76fabb4c184005861ee5282319bb68#pic_center=100x100)|![](https://ai-studio-static-online.cdn.bcebos.com/4ca7d3921b3743e0a1f5ddceede2cde4957fd6ce3856414d87b262318bd4257e)|

2).同时预测Brinkman-Forchheimer 模型的粘度$\nu_e$和渗透性$K$。需要指出，本复现结果中，在traning points Nx_EQs取10时，二者的结果差距并不明显。

|      |  复现  | 论文 |
| :--: | :------------: | :------: |
| Figure 7 A |![](https://ai-studio-static-online.cdn.bcebos.com/349571080603401382731c9a697d9092c0ede25319f84d31a72922f51e141370)|![](https://ai-studio-static-online.cdn.bcebos.com/2cb12165a9e84a4d8b780523beb45ae899c03a25fc364ca5a25864c4edcf831f)|
| Figure 7 B |![](https://ai-studio-static-online.cdn.bcebos.com/e367a42c582743678d1e99519a80e69753b32501c09742bfae0e4504d67a3958)|![](https://ai-studio-static-online.cdn.bcebos.com/7e665a375644404bae03e1e294e848d15d7e81cdd4d3498ca9fc8174a66b91f9)|
| Figure 7 C |![](https://ai-studio-static-online.cdn.bcebos.com/3a9fd034a8c843adafbb921a2b808341d1c7e770ab0c4cd88607c8da256e3bb9)|![](https://ai-studio-static-online.cdn.bcebos.com/7e4e32360c5541f4a41b99906edae6d93385e19a51f94e4f89ed7e7de4d4dc63)|

### 4.3 RAR 方法
原始方程为2D-Burgers方程： 

$
\frac{\partial u}{\partial t}+u \frac{\partial u}{\partial x}=\nu \frac{\partial^2 u}{\partial x^2}, \quad x \in[-1,1], t \in[0,1]
$

初始条件、边界条件为：

$
u(x, 0)=-\sin (\pi x), \quad u(-1, t)=u(1, t)=0
$

其数值解作者在论文github中已提供https://github.com/lu-group/gpinn。

1).综合对比，详细代码见run_3.4.1.py 以及train_3.4.1.sh 设置不同的training points——Nx_EQs以及net_type。

|      |  复现  | 论文 |
| :--: | :------------: | :------: |
| 对比 |![](https://ai-studio-static-online.cdn.bcebos.com/47cd5e11c2df49e7adff8631ae812b6bdd1501f77b72402598274c7325201003)|![](https://ai-studio-static-online.cdn.bcebos.com/dd482a4f00ed4f22b4d95fb9065b0f00ac2c6e797bda4d13b9a7584581b3bc80)|

2).PINN with RAR

|残差点数|  残差点图  | 物理场误差 | PDE残差 |
| :--: | :-------: | :------: | :-----: |
| 1500 |![](https://ai-studio-static-online.cdn.bcebos.com/4e01981db4564f20b0447dffa9135d62829880c247444073be4f3cf6b90db219)|![](https://ai-studio-static-online.cdn.bcebos.com/a98828a4a9fd4ae48b38236461a1b0cb81bf2aea3fdb4c64a869494f3d46b21d)|![](https://ai-studio-static-online.cdn.bcebos.com/6dc97dee16ea4134a2371c5128d8c4c6894de4a84958496585bf0dd95e0c1a6a)|
| 1600 |![](https://ai-studio-static-online.cdn.bcebos.com/d5d4c2a315fb41ccb5a5d2d153ab229ec2659571c5fb45ba9a2102c8b0295ea1)|![](https://ai-studio-static-online.cdn.bcebos.com/bf5d99350d414e4ebd095eaae30ea2ace75bf0a78f194cf583d26cffa55df8c4)|![](https://ai-studio-static-online.cdn.bcebos.com/7e77bd59a926477d89ddd987875b728a58fec7d1d9f546168f34de6099ac7e4b)|
| 1700 |![](https://ai-studio-static-online.cdn.bcebos.com/a5c91d6a5b7145f780b3e23e2dd03f6cba9f9a412696444190edbf83938c4ba6)|![](https://ai-studio-static-online.cdn.bcebos.com/78091017da544518a7d8484220a1ef00c9dcde1cb7f34767a5ee5ab822a07d11)|![](https://ai-studio-static-online.cdn.bcebos.com/02696bd7ae464c3aa301b1828362a4f23ec18fe26c5941c38f5ca398da472d93)|
| 1800 |![](https://ai-studio-static-online.cdn.bcebos.com/ba14da6a73f743c2a6aa09ffd94c2809bf51ffbdcaeb4326970bbe6cf02f1ce5)|![](https://ai-studio-static-online.cdn.bcebos.com/cf4e660436714aec8da1c287e2fc69567dad9fa15bc745f58f31dc2ca0d3fadf)|![](https://ai-studio-static-online.cdn.bcebos.com/1d8443f4efbf491986624a1b51d606aae6ea6568511f4330a4e2cb9032c4dfd3)|
| 1900 |![](https://ai-studio-static-online.cdn.bcebos.com/736fcb35ebe648ca9d9a9cc14efe3f0b1bfe32161fde4ee4ac3a6ee482a07d00)|![](https://ai-studio-static-online.cdn.bcebos.com/a99a44ab8da04dfda03505a62571b4f8101c6a3d5cde41fc9ae07027fdf1f291)|![](https://ai-studio-static-online.cdn.bcebos.com/1f4d373157344535990af859725cfad7835d0b90344f45678702864f30f4e1bd)|

论文结果为Fig 11
![](https://ai-studio-static-online.cdn.bcebos.com/86e5bccc13f94df69a056c607a94bd2bb8b66bd31d91407687ace9cc4111b677)


3).GPINN with RAR

|残差点数|  残差点图  | 物理场误差 | PDE残差 |
| :--: | :-------: | :------: | :-----: |
| 1500 |![](https://ai-studio-static-online.cdn.bcebos.com/30a0ce803ad44cd3a5d2d6800cd99d4d77a0f47f9f254949bf2bfc5f899fb0b7)|![](https://ai-studio-static-online.cdn.bcebos.com/f6779261ec1242299cccee22dc9984c9643668bd997d4e659aadcca919e8a765)|![](https://ai-studio-static-online.cdn.bcebos.com/a308096727034e79b9a25980af616fd4d33e25ca1f52421cbbd3fd282e7aaeeb)|
| 1600 |![](https://ai-studio-static-online.cdn.bcebos.com/9690ffd9c9be440a8ae9d892a11511d91401088a4a004104848d76f705f46111)|![](https://ai-studio-static-online.cdn.bcebos.com/c856d6644cc4444499fe939aaa00cae2c7ff27b7cb5e49958fa4095ed4feb680)|![](https://ai-studio-static-online.cdn.bcebos.com/0eb850bf28104177a2646a91bfddd53e5b3650d41e31441eb77c9236bfe05a10)|
| 1700 |![](https://ai-studio-static-online.cdn.bcebos.com/ae6f0cc6c8bd426daad4733e318b4723ed098a972fb7412583d25b326a8fefb1)|![](https://ai-studio-static-online.cdn.bcebos.com/e702832e6380442db8eed13be54f99c89b771b31865e435a86c31916e57e1576)|![](https://ai-studio-static-online.cdn.bcebos.com/9233e0d3a7b14bd1ab48ebe7edf3e057027d1e1256b34e40b5d89fac5b79f221)|
| 1800 |![](https://ai-studio-static-online.cdn.bcebos.com/2d6d08f274e74a8084a3be4b3559a97c2889b257d7114ffba2e4203b879ef96b)|![](https://ai-studio-static-online.cdn.bcebos.com/52f34c66601e46a48c914c0495841bbaba21e083c9b9457c9fbc763298fb34be)|![](https://ai-studio-static-online.cdn.bcebos.com/5b531cb1f09642feb8bd838c907dc9ba40f05412c3cf4d949ae5cf13f4227a39)|
| 1900 | ![](https://ai-studio-static-online.cdn.bcebos.com/eae9fc7a372a492c862d0cb4a0d661bd695c420d46ec45fd9bb92eb56a64b5cd)|![](https://ai-studio-static-online.cdn.bcebos.com/44896176bc5a4b52bf86136c95383a42275509f679db4dc894550635048d8363)|![](https://ai-studio-static-online.cdn.bcebos.com/dd5e94201f654cf795cca5e25af98173ff66ebd6521444418b9377b4b033d734)|

论文结果为Fig 12
![](https://ai-studio-static-online.cdn.bcebos.com/9a04af733dc64bffbabbe1605d679d2e9d2955198c6c4c93b04a6b76c64e6083)


## 6.存在问题

构建偏微分方程残差对自变量的自动微分时，目前Paddle的动态图不支持3阶以上微分，因此需要使用静态图模式，但对论文3.4.1以及3.4.2算例计算有误，代码如下时无法运行（**该方式在其他算例中均可运行**）。

```python
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
            g_eqs = paddle.incubate.autograd.grad(eqs, inn_var)
        else:
            g_eqs = paddle.zeros((2,), dtype=paddle.float32)

        return out_var, eqs, g_eqs
```

因此采用手动微分计算PDE残差对自变量的微分，修正为：

```python
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
            # temp = paddle.incubate.autograd.grad(eqs, inn_var)
            # g_eqs = temp[:, 0:1]
        else:
            g_eqs = paddle.zeros((2,), dtype=paddle.float32)

        return out_var, eqs, g_eqs
```



## 7.模型信息

| 信息          | 说明                                                      |
| ------------- | --------------------------------------------------------- |
| 发布者        | tianshao1992                                              |
| 时间          | 2022.9                                                    |
| 框架版本      | Paddle Develope                                           |
| 应用场景      | 科学计算                                                  |
| 支持硬件      | CPU、GPU                                                  |
| AI studio地址 | https://aistudio.baidu.com/aistudio/projectdetail/4493662 |
