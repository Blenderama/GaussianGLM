import pandas
from datetime import datetime
import numpy as np
import datetime
import akshare as ak
import time
import json

data_dict = {}
start_time = time.time()
f = open('raw_data.json', 'w', encoding='Utf-8')
# days_num = 1000 #考虑多久的数据
# pred_days = 4 #预测多久的结果
# writer = pandas.ExcelWriter(str(datetime.datetime.now().date()) + '.xlsx')
# writer_review = pandas.ExcelWriter(str(datetime.datetime.now().date()) +'_'+str(days_num)+ '_review.xlsx')
# writer_row = 1
shenzhen_code = pandas.read_excel('all_code.xls')
for code_i in range(len(shenzhen_code)):
# for code_i in range(5):
    # if (datetime.datetime.now()-shenzhen_code.iloc[code_i,6]).days < min(days_num, 1000):
    #     print('上市时间太短')
    #     continue
    code = "%06d" % shenzhen_code.iloc[code_i,4]
    market = shenzhen_code.iloc[code_i,0]
    name = shenzhen_code.iloc[code_i,5]
    print('Loading history data for ' + code + '...')
    try:
        df_base = ak.stock_zh_a_hist(symbol=code, period="daily", adjust='qfq')
        code_inf = ak.stock_individual_info_em(code)
        bus_type = code_inf.value[2]
    except:
        print('网络请求出错！')
        continue
    # df_base = df_base.set_index('日期')
    # price = df_base.iloc[-1]['收盘']
    # money = price*100 if code_i == 0 else 50000
    # if price * 100 > money:
    #     print('资金不足！')
    #     continue
    # try:
    weekdays = [datetime.datetime.strptime(x.strftime("%Y-%m-%d"), "%Y-%m-%d").weekday() for x in df_base['日期']]
    #     print(1)
    # except:
    #     weekdays = [datetime.datetime.strptime(x.strftime("%Y-%m-%d"), "%Y-%m-%d").weekday() for x in df_base.index]
    df_base['weekdays'] = weekdays
    open_close = ((df_base['收盘']-df_base['开盘'])/df_base['开盘'])
    high_low = ((df_base['最高']-df_base['最低'])/df_base['开盘'])
    df_base['open_close'] = (open_close*100).round(2)
    df_base['high_low'] = (high_low*100).round(2)
    data_dict[code] = {}
    data_dict[code]['type'] = bus_type
    for i, k in enumerate(['日期', 'weekdays', 'open_close']):
        if i == 0:
            data_dict[code][k] = [x.strftime('%Y-%m-%d') for x in list(df_base[k])]
        else:
            data_dict[code][k] = list(df_base[k])
    # break
    # f.write(str(data_dict))
json_data = json.dumps(data_dict, ensure_ascii=False)
f.write(json_data)
f.close()
elapse = time.time() - start_time
print(elapse)
# writer_review.close()