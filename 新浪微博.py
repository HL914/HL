import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
 
# 读取数据
tdata = []  # 训练数据
pdata = []  # 预测数据
 
with open('C:/Users/黄乐/Desktop/weibo_train_data.txt', 'r', encoding='utf-8') as file:
    for line in file:
        try:
            row = line.strip().split('\t')
            tdata.append(row)
        except Exception as e:
            print(f"Error parsing line: {line}")
 
with open('C:/Users/黄乐/Desktop/weibo_predict_data.txt', 'r', encoding='utf-8') as file:
    for line in file:
        try:
            row = line.strip().split('\t')
            pdata.append(row)
        except Exception as e:
            print(f"Error parsing line: {line}")
 
tdata = pd.DataFrame(tdata)
pdata = pd.DataFrame(pdata)
 
# 手动添加列名
tdata.columns = ["uid", "mid", "time", "forward_count", "comment_count", "like_count", "content"]
pdata.columns = ["uid", "mid", "time", "content"]
 
# 对内容列进行简单处理
tdata['content'] = tdata['content'].apply(lambda x: len(x) if x is not None else 0)
pdata['content'] = pdata['content'].apply(lambda x: len(x) if x is not None else 0)
 
# 选择特征和目标变量
X_train = tdata[["content"]]#特征是微博内容的长度（content）
y_train = tdata[["forward_count", "comment_count", "like_count"]]#目标变量是转发数、评论数和点赞数。
 
X_test = pdata[["content"]]
 
# 划分训练集和验证集,使用train_test_split函数将数据集按照80%训练集和20%验证集的比例进行划分。
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
 
# 训练随机森林模型
model = RandomForestRegressor()
model.fit(X_train, y_train)
 
# 在验证集上进行预测
y_pred_val = model.predict(X_val)
# 计算均方误差
mse = mean_squared_error(y_val, y_pred_val)
print("验证集均方误差:", mse)

# 对测试集进行预测
y_pred_test = model.predict(X_test)

# 输出预测结果,包括转发数、评论数和点赞数的预测值。
for i, pred in enumerate(y_pred_test):
    print(f"forward_count: {int(pred[0])}, comment_count: {int(pred[1])},like_count: {int(pred[2])}")
# 将预测结果与 pdata 的相关列组合成新的数据框
result = pd.DataFrame({
    'uid': pdata['uid'],
    'mid': pdata['mid'],
    'forward_count_pred': [int(pred[0]) for pred in y_pred_test],
    'comment_count_pred': [int(pred[1]) for pred in y_pred_test],
    'like_count_pred': [int(pred[2]) for pred in y_pred_test]
})
# 将新数据框保存为 txt 文件
#将新的DataFrame保存为weibo_result_data.txt文件，以制表符（\t）分隔，并且不保存索引。
result.to_csv(r'C:\Users\黄乐\Desktop\weibo_result_data.txt', sep='\t', index=False)