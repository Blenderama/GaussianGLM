import json
import numpy as np
from tqdm import tqdm

with open('raw_data.json') as f:
    lines = f.read()
    js = json.loads(lines)

# print([str(21+i)+':'+x for i,x in enumerate(set([d['type'] for d in js.values()]))]) #共87个行业


# '-10': 0, '-9': 2, ..., '9': 20, '10': 21
# ['21:燃气', '22:环保行业', '23:医药商业', '24:农牧饲渔', '25:通用设备', '26:美容护理', '27:电池', '28:石油行业', '29:中药', '30:塑料制品', '31:互联网服务', '32:风电设备', '33:工程建设', '34:电源设备', '35:仪器仪表', '36:保险', '37:造纸印刷', '38:文化传媒', '39:消费电子', '40:铁路公路', '41:纺织服装', '42:证券', '43:化学制药', '44:多元金融', '45:化纤行业', '46:汽车零部件', '47:航天航空', '48:橡胶制品', '49:汽车整车', '50:电力行业', '51:能源金属', '52:农药兽药', '53:化学原料', '54:船舶制造', '55:软件开发', '56:商业百货', '57:银行', '58:光伏设备', '59:玻璃玻纤', '60:水泥建材', '61:医疗服务', '62:家用轻工', '63:专业服务', '64:生物制品', '65:煤炭行业', '66:装修建材', '67:电子元件', '68:钢铁行业', '69:电子化学品', '70:化肥行业', '71:小金属', '72:装修装饰', '73:教育', '74:电机', '75:物流行业', '76:汽车服务', '77:交运设备', '78:化学制品', '79:游戏', '80:-', '81:计算机设备', '82:旅游酒店', '83:房地产服务', '84:房地产开发', '85:贵金属', '86:工程咨询服务', '87:食品饮料', '88:电网设备', '89:非金属材料', '90:有色金属', '91:光学光电子', '92:通信设备', '93:包装材料', '94:珠宝首饰', '95:通信服务', '96:酿酒行业', '97:医疗器械', '98:工程机械', '99:家电行业', '100:采掘行业', '101:综合行业', '102:专用设备', '103:半导体', '104:贸易行业', '105:公用事业', '106:航空机场', '107:航运港口']


type_dict = {v:str(k+11) for k,v in enumerate(set([d['type'] for d in js.values()]))}

min_len = 20
max_len = 60
f = open('text.txt', 'w')
for code, dict_values in tqdm(js.items()):
    paragraph = dict_values['open_close'][-1000:]
    while True:
        rand_len = np.random.randint(min_len, max_len)
        sentence = paragraph[:rand_len]
        paragraph = paragraph[rand_len:]
        if len(sentence) < min_len:
            break
        if (min(sentence) < -10) or (max(sentence) > 10):
            continue
        line = ' '.join([type_dict[dict_values['type']]] + [str(x) for x in sentence]) + '\n'
        f.write(line)
        
f.close()
