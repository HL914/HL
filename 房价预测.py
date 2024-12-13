import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

# 读取训练和测试数据集
train_data = pd.read_csv(
    r'C:\Users\HL\PycharmProjects\PythonProject\data\house-prices-advanced-regression-techniques\train.csv')
test_data = pd.read_csv(
    r'C:\Users\HL\PycharmProjects\PythonProject\data\house-prices-advanced-regression-techniques\test.csv')

# 打印数据集的形状以确认读取成功
print(train_data.shape)
print(test_data.shape)

# 查看前四行的部分列以了解数据结构
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# 去掉ID列，因为它对预测没有帮助
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 数据预处理
# 获取数值特征（非对象类型）
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index

# 对数值特征进行标准化处理：(x - mean) / std
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std())
)

# 将标准化后的缺失值填充为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 处理离散特征：将类别特征转换为独热编码，并保留NaN值作为单独的一个特征
all_features = pd.get_dummies(all_features, dummy_na=True, dtype=int)

# 转换为张量
n_train = train_data.shape[0]  # 训练集的数据个数
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

# 定义输入特征的数量
in_features = train_features.shape[1]

# 定义损失函数：均方误差损失
loss = nn.MSELoss()


def get_net():
    """定义神经网络模型"""
    net = nn.Sequential(
        nn.Linear(in_features, 128),  # 全连接层，输入到128个神经元
        nn.ReLU(),  # ReLU激活函数
        nn.Linear(128, 1)  # 全连接层，128个神经元到输出层
    )
    return net


def init_weights(m):
    """初始化权重"""
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)  # Xavier均匀分布初始化权重
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # 初始化偏置为0


def log_rmse(net, features, labels):
    """计算对数均方根误差"""
    clipped_preds = torch.clamp(net(features), 1, float('inf'))  # 预测值限制在[1, inf]范围内
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))  # 计算对数均方根误差
    return rmse.item()  # 返回标量值


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    """训练模型"""
    train_ls, test_ls = [], []  # 存储训练和验证的RMSE
    train_iter = d2l.load_array((train_features, train_labels), batch_size)  # 创建数据加载器
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)  # Adam优化器

    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()  # 清零梯度
            l = loss(net(X), y)  # 计算损失
            l.backward()  # 反向传播
            optimizer.step()  # 更新参数

        train_ls.append(log_rmse(net, train_features, train_labels))  # 记录训练RMSE
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))  # 记录验证RMSE

    return train_ls, test_ls


def get_k_fold_data(k, i, X, y):
    """获取第i折的数据"""
    assert k > 1
    fold_size = X.shape[0] // k  # 每折的大小
    X_train, y_train = None, None

    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]

        if j == i:
            X_valid, y_valid = X_part, y_part  # 第i折作为验证集
        elif X_train is None:
            X_train, y_train = X_part, y_part  # 第一次赋值给训练集
        else:
            X_train = torch.cat([X_train, X_part], 0)  # 连接训练集
            y_train = torch.cat([y_train, y_part], 0)

    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    """K折交叉验证"""
    train_l_sum, valid_l_sum = 0, 0

    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)  # 获取第i折的数据
        net = get_net()
        net.apply(init_weights)  # 初始化权重
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]  # 累加训练RMSE
        valid_l_sum += valid_ls[-1]  # 累加验证RMSE

        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')  # 绘制训练和验证曲线

        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, 验证log rmse{float(valid_ls[-1]):f}')

    return train_l_sum / k, valid_l_sum / k


def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    """使用所有数据训练并预测"""
    net = get_net()
    net.apply(init_weights)  # 初始化权重
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')  # 绘制训练曲线
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    d2l.plt.show()
    preds = net(test_features).detach().numpy()  # 预测测试集结果
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])  # 添加预测结果到测试数据
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)  # 构建提交文件
    submission.to_csv('submission.csv', index=False)  # 保存为CSV文件


# 设置超参数
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.1, 0.2, 128

# 进行K折交叉验证
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, 平均验证log rmse: {float(valid_l):f}')

# 使用所有数据进行最终训练和预测
train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)

