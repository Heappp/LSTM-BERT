import pandas as pd
import matplotlib.pyplot as plt


# 显示中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 #有中文出现的情况，需要u'内容'

# 数据
df1 = pd.read_json('archive/Sarcasm_Headlines_Dataset.json', lines=True)
df2 = pd.read_json('archive/Sarcasm_Headlines_Dataset_v2.json', lines=True)
df = pd.concat([df1, df2], axis=0)

# 显示信息
print(df.info())

# 画数量图
grouper = dict(map(lambda x: (x[0], len(x[1])), df.groupby('is_sarcastic').groups.items()))
plt.bar(x=0, height=grouper[0], color='blue', label='True')
plt.bar(x=1, height=grouper[1], color='red', label='False')
plt.legend()
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.xlabel('is_sarcastic')
plt.ylabel('count')
plt.show()
