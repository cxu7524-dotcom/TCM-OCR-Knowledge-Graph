from pathlib import Path
import hashlib,json
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/small_columns_step1_holdout'; OUT.mkdir(exist_ok=True)
base=Path('古籍评测集/医家类/思濟堂方書')
pos=[]
def add(page,main,outer,mb,rb,rt,lb,lt,dose=None,applies=None,note=''):
 pos.append(dict(id=f'P{page}-{len(pos)+1:02}',page=page,label='positive',outer_bbox=outer,main_bbox=mb,main_text=main,columns=[dict(side='right',bbox=rb,text=rt),dict(side='left',bbox=lb,text=lt)],reading_order=['right','left'],dose={'verbatim':dose,'applies_to':applies or [],'status':'可读' if dose else '该双列无剂量'},uncertainty=note or '未发现疑字'))
add(37,'揀芍',[600,405,636,542],[603,405,634,461],[619,466,636,542],'二錢半',[600,466,619,512],'酒炒','二錢半',['揀芍'])
add(39,'川附子',[940,405,981,570],[945,405,978,484],[962,487,980,570],'各二錢',[941,487,961,518],'製','各二錢',['人參','川附子'])
add(39,'揀白芍',[895,405,932,570],[897,405,930,486],[915,487,932,570],'各三錢',[895,487,914,542],'酒炒','各三錢',['白茯苓','揀白芍'])
add(39,'白朮',[940,574,981,738],[945,574,977,624],[962,626,980,738],'四錢黃土',[941,626,961,675],'拌炒','四錢',['白朮'])
add(43,'巴豆',[46,242,82,402],[48,242,77,299],[65,300,82,402],'五錢去皮',[46,300,64,402],'心研如脂','五錢',['巴豆'])
add(46,'半夏',[848,244,883,330],[849,244,882,300],[867,300,883,330],'薑',[848,300,866,330],'製')
add(46,'蒼朮',[848,565,883,710],[849,565,882,630],[867,630,883,710],'各一錢',[848,630,866,655],'炒','各一錢',['半夏','厚朴','蒼朮'],'共享劑量的適用範圍依本頁相鄰配伍判讀')
add(46,'麻黃',[799,565,835,740],[800,565,833,630],[817,630,835,740],'八分若頭不',[799,630,816,740],'疼無汗不用','八分',['麻黃'],'末字及切行待複核')
add(50,'黃芩',[336,571,371,740],[337,571,369,630],[353,630,371,740],'各一錢半',[336,630,352,680],'酒炒','各一錢半',['防風','黃芩'])
add(34,'大〔?〕棗',[39,244,75,410],[40,244,74,330],[57,330,75,410],'十枚煮',[39,330,56,410],'去皮核','十枚',['大〔?〕棗'],'主字第二字待核')
add(34,'羚羊角',[39,540,75,735],[40,540,74,630],[57,630,75,735],'一錢二分',[39,630,56,710],'用尖鎊','一錢二分',['羚羊角'])
add(49,'杏仁',[139,242,176,410],[141,242,173,300],[158,300,176,410],'七枚去皮',[139,300,157,355],'尖研','七枚',['杏仁'])
add(49,'親切',[699,470,738,690],[700,470,735,520],[718,520,738,690],'表裏寒熱虛實',[699,520,717,690],'邪正此名八要',None,[],'非藥量夾注，僅用於幾何分欄')
add(34,'杏仁',[282,240,321,385],[284,240,318,300],[300,300,321,385],'去皮尖',[282,300,299,355],'另研',None,[],'非藥量夾注，僅用於幾何分欄')
add(43,'溫服',[799,285,840,435],[800,285,838,350],[817,350,838,435],'方見中',[799,350,816,405],'風門',None,[],'非藥量夾注，僅用於幾何分欄')
add(43,'送下',[749,515,791,660],[750,515,789,575],[770,575,790,660],'方見痰',[750,575,769,605],'門',None,[],'非藥量夾注，僅用於幾何分欄')
add(45,'二錢',[799,398,835,565],[800,398,834,460],[817,460,835,565],'滾水泡五',[799,460,816,515],'次用',None,[],'二錢在大字主文，不納入雙列內劑量評估')
add(45,'法',[749,378,785,461],[750,378,784,408],[768,408,785,461],'方見',[749,408,767,461],'暑門',None,[],'非藥量夾注，僅用於幾何分欄')
add(45,'之',[799,653,835,736],[800,653,834,682],[817,682,835,736],'方見',[799,682,816,736],'中風',None,[],'非藥量夾注，僅用於幾何分欄')
add(49,'之',[43,340,79,460],[43,340,78,380],[62,380,79,460],'方見中',[43,380,61,435],'暑門',None,[],'非藥量夾注，僅用於幾何分欄')
for item in pos:
 if item['page']==45 and item['main_text']=='二錢':
  item['context_dose']={'verbatim':'二錢','applies_to':['吳茱萸'],'location':'主文','included_in_double_column_dose_evaluation':False}
neg=[]
def negative(page,bbox,text,kind='普通大字列'):
 neg.append(dict(id=f'N{page}-{len(neg)+1:02}',page=page,label='negative',negative_type=kind,outer_bbox=bbox,main_bbox=bbox,main_text=text,columns=[],reading_order=['top_to_bottom'],dose=None,uncertainty='未發現疑字'))
for x,t in [(865,'水四鍾'),(815,'此方不'),(765,'骨節疼'),(715,'者相宜'),(665,'只此一'),(615,'則不死'),(565,'之至寶'),(450,'論全重'),(350,'此方專'),(300,'酸疼足'),(250,'禁白濁'),(200,'渴頭目')]: negative(39,[x-17,238,x+18,330],t)
for x,t in [(965,'發熱無'),(915,'遍身瘙'),(865,'手足拘'),(815,'濕氣加'),(765,'足浮腫'),(715,'遍身疼')]:negative(37,[x-17,238,x+18,330],t)
negative(46,[801,245,836,357],'白芷八分','單列小字')
negative(46,[801,410,836,520],'甘草五分','單列小字')
for item,main,b,word,dose in [(neg[-2],'白芷',[817,300,836,357],'八分','八分'),(neg[-1],'甘草',[817,460,836,520],'五分','五分')]:
 item['main_text']=main;item['main_bbox']=[801,item['outer_bbox'][1],836,b[1]];item['columns']=[{'side':'single','bbox':b,'text':word}];item['reading_order']=['main','single'];item['dose']={'verbatim':dose,'applies_to':[main],'status':'可讀'}
neg.append({'id':'N39-bleed','page':39,'label':'negative','negative_type':'透印','outer_bbox':[899,325,934,390],'main_bbox':None,'main_text':None,'columns':[],'reading_order':[],'dose':None,'uncertainty':'背面透印，不作正面原文轉寫'})
neg.append({'id':'N39-near','page':39,'label':'negative','negative_type':'相鄰大字列靠近','outer_bbox':[595,238,685,330],'main_bbox':None,'main_text':None,'columns':[{'side':'right','bbox':[648,238,683,330],'text':'只此一'},{'side':'left','bbox':[598,238,633,330],'text':'則不死'}],'reading_order':['right','left'],'dose':None,'uncertainty':'兩條獨立大字正文列，不是夾注'})
pages={}
for n in sorted({s['page'] for s in pos+neg}):
 p=ROOT/base/f'page_{n}.png'; pages[str(n)]={'path':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size':list(Image.open(p).size),'split':'holdout'}
obj={'status':'fixed_holdout_for_small_column_task','scope':'局部幾何及可辨認劑量，不是整頁方劑驗收','development_page':35,'pages':pages,'samples':pos+neg,'notes':['圖像像素框；以右列在前、每列自上而下為讀序。','37/39/50曾用於本項目其他方劑任務驗收；本次雙列小字不應將其描述為從未在項目中使用的盲頁。','疑字和適用範圍記錄在相應樣本中；非劑量雙列只參與幾何評估。']}
(OUT/'annotations.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
print('positive',len(pos),'negative',len(neg),'pages',','.join(pages))
