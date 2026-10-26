from pathlib import Path
import json,csv
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE,MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE,XL_LABEL_POSITION,XL_LEGEND_POSITION,XL_TICK_MARK
R=Path(__file__).resolve().parent;O=R/'generated';O.mkdir(exist_ok=True)
D=json.loads((R.parents[1]/'data/models/planned-v1-evaluation.json').read_text())
sets=[D['test']]+[x['result'] for x in D['additional_frozen_checkpoint_checks']]
metrics=['avg_comfort','energy_kwh','total_reward','unsafe_steps']
means={name:{k:sum(x['summary'][name][k] for x in sets)/3 for k in metrics} for name in ['model','deterministic']}
P=Presentation();P.slide_width=Inches(13.333);P.slide_height=Inches(7.5)
NAVY='112A3B';INK='193648';TEAL='087E83';MINT='D9EEEA';BG='F5F7F5';GRAY='587080';LIGHT='E4EAEB';WHITE='FFFFFF';AMBER='A26117';GOLD='F6E6CA'
FONT='Liberation Sans'
def color(c):return RGBColor.from_string(c)
def rect(s,x,y,w,h,fill,line=None,radius=False):
 shape=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h));shape.fill.solid();shape.fill.fore_color.rgb=color(fill);shape.line.fill.background() if line is None else None
 if line:shape.line.color.rgb=color(line)
 if radius:
  try:shape.adjustments[0]=.12
  except:pass
 return shape
def text(s,x,y,w,h,t,size=22,c=INK,bold=False,align=PP_ALIGN.LEFT):
 box=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));f=box.text_frame;f.word_wrap=True;f.margin_left=f.margin_right=Inches(.015);f.margin_top=f.margin_bottom=0
 for i,line in enumerate(t.split('\n')):
  p=f.paragraphs[0] if i==0 else f.add_paragraph();p.text=line;p.font.name=FONT;p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=color(c);p.alignment=align;p.space_after=Pt(4)
 return box
def new(n,title,tag='МАГІСТЕРСЬКА РОБОТА'):
 s=P.slides.add_slide(P.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=color(BG)
 rect(s,.45,.42,.10,.32,TEAL);text(s,.72,.40,10,.3,tag,12,TEAL,True);text(s,.65,.95,12.0,.9,title,32,INK,True)
 rect(s,.65,7.02,12.03,.012,LIGHT);text(s,.65,7.13,11,.2,'Головко Ярослав Ігорович  ·  ЛНУ імені Івана Франка  ·  2026',11,GRAY);text(s,12.05,7.08,.6,.3,f'{n:02d} / 10',11,GRAY,align=PP_ALIGN.RIGHT)
 return s
def note(s,t):s.notes_slide.notes_text_frame.text=t+'\n\nДжерело: docs/planned-policy.md; data/models/planned-v1-evaluation.json; код 82a624b168e334f26c191276c95fd98cb36f679d.'
def card(s,x,y,w,h,title,body,accent=TEAL):
 rect(s,x,y,w,h,WHITE,radius=True);rect(s,x,y,.06,h,accent);text(s,x+.23,y+.2,w-.45,.5,title,22,accent,True);text(s,x+.23,y+.88,w-.45,h-.98,body,17 if h<1.6 else 20)
def arrow(s,x1,y1,x2,y2):
 a=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1),Inches(x2),Inches(y2));a.line.color.rgb=color(TEAL);a.line.width=Pt(2)
 # Native DrawingML arrowhead.
 from pptx.oxml.xmlchemy import OxmlElement
 end=OxmlElement('a:tailEnd');end.set('type','triangle');a.line._get_or_add_ln().append(end)
def badge(s,x,y,w,num,label,fill=MINT,c=TEAL):
 rect(s,x,y,w,1.18,fill,radius=True);text(s,x+.17,y+.12,w-.34,.5,num,30,c,True);text(s,x+.17,y+.72,w-.34,.32,label,15,c)
def chart(s,x,y,w,h,categories,values,colors,title,maxvalue=None):
 data=CategoryChartData();data.categories=categories;data.add_series(title,values)
 ch=s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED,Inches(x),Inches(y),Inches(w),Inches(h),data).chart
 ch.has_legend=False;ch.has_title=False;ch.chart_style=10
 ch.category_axis.tick_labels.font.name=FONT;ch.category_axis.tick_labels.font.size=Pt(19)
 ch.value_axis.tick_labels.font.name=FONT;ch.value_axis.tick_labels.font.size=Pt(12);ch.value_axis.minimum_scale=0
 if maxvalue:ch.value_axis.maximum_scale=maxvalue
 ch.value_axis.has_major_gridlines=False;ch.category_axis.major_tick_mark=XL_TICK_MARK.NONE;ch.value_axis.major_tick_mark=XL_TICK_MARK.NONE
 plot=ch.plots[0];plot.has_data_labels=True;plot.data_labels.position=XL_LABEL_POSITION.OUTSIDE_END;plot.data_labels.font.size=Pt(18);plot.data_labels.font.name=FONT;plot.data_labels.number_format='0.0';plot.gap_width=90
 for point,c in zip(ch.series[0].points,colors):point.format.fill.solid();point.format.fill.fore_color.rgb=color(c);point.format.line.fill.background()
 return ch
# 1
s=P.slides.add_slide(P.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=color(NAVY)
text(s,.65,.50,11.9,.7,'Львівський національний університет імені Івана Франка\nФакультет електроніки та комп’ютерних технологій',16,'BCCFD3')
text(s,.65,1.85,9.0,3.25,'Імплементація агентної системи\nштучного інтелекту для керування\nмікрокліматом розумного будинку\nз використанням датчиків клімату',29,WHITE,True)
rect(s,.67,5.20,1.2,.055,'53C7B3')
text(s,.65,5.6,9.0,1.1,'Головко Ярослав Ігорович\nНауковий керівник: доц. Мисюк Р. В.',21,WHITE)
text(s,.65,7.05,10,.2,'МАГІСТЕРСЬКА КВАЛІФІКАЦІЙНА РОБОТА  ·  ЛЬВІВ, 2026',11,'BCCFD3')
# Editable house symbol.
roof=s.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE,Inches(10.0),Inches(2.2),Inches(2.65),Inches(1.8));roof.fill.solid();roof.fill.fore_color.rgb=color(TEAL);roof.line.fill.background()
rect(s,10.35,3.5,1.95,1.8,'244B5A',radius=True);rect(s,10.65,3.85,.45,.45,'86D5C2');rect(s,11.5,3.85,.45,.45,'86D5C2');rect(s,11.05,4.55,.5,.75,'86D5C2')
text(s,9.7,5.65,3.0,.8,'ДАТЧИКИ → АГЕНТ\n→ РЕКОМЕНДАЦІЇ',16,'86D5C2',True,PP_ALIGN.CENTER)
note(s,'Презентація відображає оновлений метод — дистиляцію політики від модельного планувальника. Висновки про якість належать симуляційному експерименту; натурні дані датчиків ще не аналізувалися.')
# 2
s=new(2,'Задача: узгодити комфорт і ресурси')
card(s,.65,2.0,3.85,2.35,'Мікроклімат','Температура, вологість\nCO₂ та PM2.5\nПрисутність мешканців')
card(s,4.75,2.0,3.85,2.35,'Взаємний вплив','Вентиляція змінює\nякість повітря й тепло.\nОпалення потребує енергії.')
card(s,8.85,2.0,3.85,2.35,'Критерій якості','Вищий комфорт\nМенше порушень\nКонтроль енергії')
rect(s,.65,4.75,12.05,1.65,NAVY,radius=True);text(s,.95,4.97,11.4,.35,'МЕТА РОБОТИ',14,'86D5C2',True);text(s,.95,5.46,11.4,.7,'Створити агентну систему, яка обирає режими обладнання\nта дозволяє відтворювано перевірити якість цих рішень.',24,WHITE)
note(s,'Об’єкт — процеси автоматизованого керування мікрокліматом. Предмет — гібридна політика, опрацювання показників і відтворюване оцінювання. Зниження енергії не розглядається окремо від комфорту та порушень.')
# 3
s=new(3,'Завдання дослідження')
items=[('01','Проаналізувати предметну область','Визначити показники мікроклімату й критерії якості.'),('02','Спроєктувати агентну систему','Підготувати архітектуру, формат даних і правила.'),('03','Реалізувати та навчити модель','Передати мережі вибір корекцій від планувальника.'),('04','Створити інструменти перевірки','Реалізувати вебпанель, імпорт та порівняння моделей.'),('05','Оцінити отримані рішення','Порівняти комфорт, порушення та енерговитрати.')]
for i,(num,title,body) in enumerate(items):
 y=1.95+i*.86
 rect(s,.7,y,.65,.62,MINT,radius=True);text(s,.74,y+.1,.55,.35,num,21,TEAL,True,PP_ALIGN.CENTER)
 text(s,1.65,y-.01,10.6,.35,title,22,TEAL,True);text(s,1.65,y+.39,10.6,.35,body,18)
note(s,'Завдання охоплюють аналіз, проєктування, реалізацію та відтворюване симуляційне оцінювання. Фізичні датчики наявні; аналіз натурних результатів залишається наступним етапом.')
# 4
s=new(4,'Архітектура та підготовка даних')
labels=[('Вебпанель / CLI','запити, імпорт, експорт'),('Сервіс аналізу','валідація та історія'),('Агент + контролер','вибір однієї корекції'),('Остаточні правила','обмеження режимів'),('Симулятор','динаміка, погода, мешканці'),('Лабораторія','порівняння та статистика')]
pos=[(.65,2.0),(4.75,2.0),(8.85,2.0),(8.85,4.1),(4.75,4.1),(.65,4.1)]
for (title,body),(x,y) in zip(labels,pos):
 rect(s,x,y,3.85,1.5,WHITE,radius=True);text(s,x+.2,y+.2,3.45,.45,title,22,TEAL,True);text(s,x+.2,y+.85,3.45,.4,body,18)
for a,b in [((4.5,2.75),(4.75,2.75)),((8.6,2.75),(8.85,2.75)),((10.78,3.5),(10.78,4.1)),((8.85,4.85),(8.6,4.85)),((4.75,4.85),(4.5,4.85))]:arrow(s,*a,*b)
text(s,.85,5.95,11.8,.4,'JSON / CSV / TXT → перевірка показників → 54 ознаки стану',21,TEAL,True)
text(s,.85,6.48,11.8,.3,'Python  ·  PyTorch  ·  NumPy  ·  SQLite  ·  HTML / JavaScript',17,GRAY)
note(s,'Схема показує логічний контур роботи, а не окремі сервери. Лабораторія також звертається до спільного сервісу й бази. SQLite зберігає імпорти, задачі та оцінки. У режимі рекомендацій фізичні пристрої не перемикаються.')
# 5
s=new(5,'Метод: навчання від модельного планувальника')
card(s,.65,1.95,3.85,2.65,'01  Прогноз','19 корекцій у копіях\nсимулятора. Горизонт:\n6 кроків по 10 хвилин.\nВибір кращої корекції.')
card(s,4.75,1.95,3.85,2.65,'02  Навчання','1728 прикладів.\nМережа: 54 ознаки\n→ 19 варіантів.\nДо 30 епох навчання.')
card(s,8.85,1.95,3.85,2.65,'03  Застосування','Поточний стан\n→ навчена корекція\n→ остаточні правила\n→ рекомендація.')
rect(s,.65,4.95,12.05,1.65,NAVY,radius=True)
text(s,.92,5.15,11.4,.35,'ДИСТИЛЯЦІЯ ПОЛІТИКИ',16,'86D5C2',True)
text(s,.92,5.65,11.4,.65,'Прогноз потрібен під час формування навчальних міток.\nМодель обирається на валідаційних даних; тестові сценарії відкладено.',21,WHITE)
note(s,'Дія 0 залишає детерміновану основу, дії 1–18 змінюють режим одного пристрою. Модель: приховані шари 256→256→128, дві dueling-гілки; виходи навчаються як logits політики. TRAIN: 24 епізоди, мітка кожні 2 кроки, 1728 прикладів; validation: 6 сценаріїв seed 41000 кожні 5 епох. Канонічний seed 42, епоха 30. Учитель оцінює discounted reward v2 мінус 10 за unsafe-крок і має привілейований прогноз погоди. При inference тільки поточні 54 ознаки. Це model-based policy distillation.')
# 6
s=new(6,'Наукова новизна отриманих результатів')
card(s,.65,2.0,3.85,3.15,'Гібридна політика','Удосконалено спосіб\nвибору режимів:\nдетермінована основа,\nнавчена корекція\nта остаточні правила.')
card(s,4.75,2.0,3.85,3.15,'Навчальні мітки','Адаптовано прогноз\nнаслідків 19 корекцій\nдля навчання мережі,\nяка працює лише\nз поточним станом.')
card(s,8.85,2.0,3.85,3.15,'Оцінювання якості','Досліджено вплив\nнавченої корекції\nна комфорт, порушення\nта енергію за однакових\nсценаріїв симуляції.')
rect(s,.65,5.5,12.05,.88,MINT,radius=True)
text(s,.9,5.72,11.5,.48,'Внесок роботи: адаптація гібридної політики для керування мікрокліматом.',20,TEAL,True)
note(s,'Новизна сформульована як прикладний методичний внесок цієї роботи. Дистиляція політики та модельне планування — відомі підходи; світова першість чи нова фундаментальна архітектура не заявляються. Остаточні правила не доводять фізичну безпеку системи. Практичне значення — реалізований сервіс та відтворюване порівняння.')
# 7
s=new(7,'Веблабораторія для перевірки моделей')
for y,title,body in [(2.0,'Ручний тест','Показники → рекомендація\nта пояснення змін правил.'),(3.5,'Порівняння','До трьох checkpoint,\nоднакові сценарії й seed.'),(5.0,'Збереження','Фонові задачі, кеш,\nісторія та експорт JSON.')]:
 text(s,.75,y,4.2,.4,title,23,TEAL,True);text(s,.75,y+.52,4.2,.8,body,20)
image=R/'application.png'
if image.exists():s.shapes.add_picture(str(image),Inches(5.35),Inches(1.9),width=Inches(7.33),height=Inches(4.8))
else:raise FileNotFoundError(image)
note(s,'На слайді фактичний знімок локальної вебпанелі з checkpoint planned-v1.pt і прикладом високого CO₂ під час сну. Збереження в SQLite та пояснення працюють; модель не перенавчається за оцінками користувача. Цей тест не є натурним вимірюванням.')
# 8
s=new(8,'Результат: кращі умови за додаткової енергії')
text(s,.75,1.85,11.9,.4,'Незмінний checkpoint  ·  60 добових test-сценаріїв  ·  144 кроки / епізод',18,GRAY)
text(s,.85,2.5,5.5,.4,'Комфорт, %  ↑',23,INK,True);text(s,7.0,2.5,5.5,.4,'Небезпечні кроки / епізод  ↓',23,INK,True)
chart(s,.65,2.95,5.7,2.0,['Правила','Модель'],[means['deterministic']['avg_comfort'],means['model']['avg_comfort']],[GRAY,TEAL],'Комфорт',100)
chart(s,6.8,2.95,5.7,2.0,['Правила','Модель'],[means['deterministic']['unsafe_steps'],means['model']['unsafe_steps']],[GRAY,TEAL],'Unsafe',14)
badge(s,.65,5.35,3.85,'+3,11 в.п.','середній комфорт');badge(s,4.75,5.35,3.85,'−16,80%','небезпечні кроки');badge(s,8.85,5.35,3.85,'+3,57%','енергоспоживання',GOLD,AMBER)
note(s,'Канонічний seed 42: усереднення трьох наборів по 20 сценаріїв, test seeds 61000, 71000, 81000. Комфорт: правила 79.5193%, модель 82.6277%. Енергія: 25.100125 проти 25.997181 кВт·год / епізод. Unsafe: 10.316667 проти 8.583333. Винагорода: 91.480154 проти 123.401975 (+34.89%). Пороги unsafe — специфікація симулятора: температура поза 5–32 °C, CO₂ понад 1500, PM2.5 понад 55. Це компроміс заради умов, не економія енергії.')
# 9
s=new(9,'Висновки та подальша робота')
card(s,.65,1.95,7.65,1.3,'Систему реалізовано','Імпорт, агентні рекомендації, правила й веблабораторія.')
card(s,.65,3.45,7.65,1.3,'Навчена корекція корисна в симуляції','Вищий комфорт і менше порушень за більшої енергії.')
card(s,.65,4.95,7.65,1.3,'Наступний етап — фізичні датчики','Додати вимірювання й оцінити якість рекомендацій.',AMBER)
badge(s,8.65,1.95,4.05,'28','автоматизованих тестів')
badge(s,8.65,3.45,4.05,'7','предметних перевірок правил')
badge(s,8.65,4.95,4.05,'3','незалежні запуски навчання')
note(s,'Повторні навчальні seeds 42,43,44 мали власний validation-відбір. На спільних 20 test-сценаріях seed 61000 усі підвищили комфорт і зменшили unsafe, але витрачали більше енергії. Це окрема перевірка від 60 епізодів канонічної моделі на слайді 8. Навчальні повтори не є окремими фізичними будинками. Натурні результати очікуються. Тести виконано для версії коду 82a624b. На 20 синтетичних станах медіана inference 0.177 мс проти 43.638 мс планувальника без HTTP/SQLite; переносимість на фізичний будинок потребує перевірки.')
# 10
s=P.slides.add_slide(P.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=color(NAVY)
rect(s,5.95,1.75,1.4,.065,'53C7B3')
text(s,1.0,2.5,11.33,1.0,'Дякую за увагу!',46,WHITE,True,PP_ALIGN.CENTER)
text(s,1.0,3.85,11.33,.65,'Готовий відповісти на ваші запитання',24,'86D5C2',False,PP_ALIGN.CENTER)
text(s,1.0,5.5,11.33,.5,'Головко Ярослав Ігорович',22,WHITE,False,PP_ALIGN.CENTER)
text(s,1.0,6.12,11.33,.5,'Науковий керівник: доц. Мисюк Р. В.',18,'BCCFD3',False,PP_ALIGN.CENTER)
text(s,12.05,7.08,.6,.3,'10 / 10',11,'BCCFD3',align=PP_ALIGN.RIGHT)
note(s,'Завершення доповіді та перехід до запитань. Не додавати нових кількісних тверджень.')
# Embed the full slide-aligned speech, keeping technical notes separate.
speech=json.loads((R/'speech.json').read_text())
assert len(speech)==len(P.slides)==10
for slide,item in zip(P.slides,speech):
 technical=slide.notes_slide.notes_text_frame.text
 slide.notes_slide.notes_text_frame.text=f"Слайд {item['slide']}. {item['title']}\nОрієнтовний час: {item['time']}\n\nЩо показати: {item['cue']}\n\nТекст виступу\n{item['text']}\n\nДовідка для запитань\n{technical}"
P.core_properties.title='Агентна система керування мікрокліматом розумного будинку';P.core_properties.author='Головко Ярослав Ігорович';P.core_properties.subject='Магістерська робота: оновлена модель та перевірені результати'
P.save(O/'Презентація_Головко_10_слайдів.pptx')
with (O/'Дані_графіків.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['metric','deterministic','model'])
 for k in metrics:w.writerow([k,means['deterministic'][k],means['model'][k]])
(O/'Нотатки_до_слайдів.md').write_text('\n\n'.join(f"## Слайд {i+1}\n\n{slide.notes_slide.notes_text_frame.text}" for i,slide in enumerate(P.slides)))
print('Created',len(P.slides),'slides')
