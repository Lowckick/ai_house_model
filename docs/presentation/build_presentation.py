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
s=new(3,'Від показників датчиків до стану агента')
card(s,.65,2.0,3.85,3.35,'01  Джерело','Фізичні датчики наявні.\nФорматований експорт:\nJSON / CSV / TXT.\n\nНатурні результати\nочікуються.',AMBER)
card(s,4.75,2.0,3.85,3.35,'02  Перевірка','Обов’язково: час,\nтемпература, вологість.\n\nПеревірка типів, меж,\nпропусків і порядку.\nДо 5000 записів / файл.')
card(s,8.85,2.0,3.85,3.35,'03  Ознаки','54 числові компоненти:\nпогода, стан будинку,\nпристрої, циклічний час.\n\nCO₂, PM2.5 та інші поля\nдодаються за наявності.')
text(s,.85,5.8,11.65,.7,'Поточні кількісні результати отримано в симуляції.\nЯкість фізичних датчиків та їх калібрування ще потребують аналізу.',20,GRAY)
note(s,'Необов’язкові значення замінюються типовими зі звітом попереджень. При імпорті зовнішня погода фіксована, рядки незалежні. Дані майбутніх фізичних вимірювань потрібно описати за моделями датчиків, часом, одиницями та калібруванням. Не заявляємо завершеної апаратної інтеграції.')
# 4
s=new(4,'Модульна архітектура системи')
labels=[('Вебпанель / CLI','запити, імпорт, експорт'),('Сервіс аналізу','валідація та історія'),('Агент + контролер','вибір однієї корекції'),('Остаточні правила','обмеження режимів'),('Симулятор','динаміка, погода, мешканці'),('Лабораторія','порівняння та статистика')]
pos=[(.65,2.0),(4.75,2.0),(8.85,2.0),(8.85,4.1),(4.75,4.1),(.65,4.1)]
for (title,body),(x,y) in zip(labels,pos):
 rect(s,x,y,3.85,1.5,WHITE,radius=True);text(s,x+.2,y+.2,3.45,.45,title,22,TEAL,True);text(s,x+.2,y+.85,3.45,.4,body,18)
for a,b in [((4.5,2.75),(4.75,2.75)),((8.6,2.75),(8.85,2.75)),((10.78,3.5),(10.78,4.1)),((8.85,4.85),(8.6,4.85)),((4.75,4.85),(4.5,4.85))]:arrow(s,*a,*b)
text(s,.85,6.13,11.8,.4,'Python  ·  PyTorch  ·  NumPy  ·  SQLite  ·  HTML / JavaScript',20,GRAY)
note(s,'Схема показує логічний контур роботи, а не окремі сервери. Лабораторія також звертається до спільного сервісу й бази. SQLite зберігає імпорти, задачі та оцінки. У режимі рекомендацій фізичні пристрої не перемикаються.')
# 5
s=new(5,'Мережа обирає одну з 19 корекцій')
for x,w,num,lab in [(1.0,2.4,'54','ознаки стану'),(4.95,3.3,'256 → 256 → 128','приховані шари'),(9.8,2.4,'19','індексів корекцій')]:
 rect(s,x,2.05,w,1.2,MINT,radius=True);text(s,x+.1,2.25,w-.2,.5,num,23 if len(num)>8 else 28,TEAL,True,PP_ALIGN.CENTER);text(s,x-.2,3.42,w+.4,.5,lab,19,INK,False,PP_ALIGN.CENTER)
arrow(s,3.55,2.65,4.7,2.65);arrow(s,8.45,2.65,9.6,2.65)
rect(s,.65,4.35,7.7,1.65,WHITE,radius=True);text(s,.9,4.55,7.2,.4,'Детермінована основа + навчена корекція',23,TEAL,True);text(s,.9,5.15,7.1,.6,'Дія 0 — залишити основу.\nДії 1–18 — змінити режим одного пристрою.',21)
rect(s,8.65,4.35,4.05,1.65,NAVY,radius=True);text(s,8.9,4.57,3.55,.35,'ОСТАТОЧНИЙ RuleLayer',16,'86D5C2',True);text(s,8.9,5.12,3.55,.6,'Обмеження мають\nпріоритет над комфортом.',20,WHITE)
note(s,'Точна архітектура: Linear 54→256, 256→256, 256→128; дві гілки 128→64→1 і 128→64→19. Збережено dueling-структуру, але в новому режимі її виходи навчаються як logits політики. Режими корекцій: HVAC, вентиляція, очищувач, зволожувач і робот. Остаточні правила не є формально доведеною гарантією фізичної безпеки.')
# 6
s=new(6,'Навчання через прогноз наслідків дій')
card(s,.65,2.0,3.85,2.95,'Планувальник','Перебір 19 корекцій\nу копіях симулятора.\n\nГоризонт: 6 кроків\nпо 10 хвилин.')
card(s,4.75,2.0,3.85,2.95,'Навчання мережі','1728 прикладів / seed\n30 епох класифікації.\n\nМітки містять\nрізноманітні корекції.')
card(s,8.85,2.0,3.85,2.95,'Відбір і застосування','Checkpoint — лише\nза validation.\n\nПри застосуванні:\nтільки поточні 54 ознаки.')
rect(s,.65,5.3,12.05,1.1,NAVY,radius=True);text(s,.9,5.5,11.5,.7,'Дистиляція політики з модельного планувальника.\nПривілейований прогноз використовується лише під час підготовки навчальних міток.',20,WHITE)
note(s,'Вчитель оцінює discounted reward v2 та додатковий штраф 10 за unsafe-крок. TRAIN: 24 епізоди, мітка кожні 2 кроки, суміш базових і вчительських траєкторій. Validation: 6 сценаріїв, seed 41000, кожні 5 епох. Канонічний навчальний seed 42, обрана епоха 30. Тести 61000, 71000, 81000 не обирають checkpoint. Метод — model-based policy distillation; це не підтвердження успіху online RL.')
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
s=new(9,'Повторюваність і програмні перевірки')
text(s,.75,1.85,7.2,.7,'Повтори навчання: комфорт на спільному\nвідкладеному наборі з 20 сценаріїв, %',20,INK,True)
chart(s,.65,2.65,7.1,3.55,['Правила','Seed 42','Seed 43','Seed 44'],[80.3901528,83.4587674,87.3666076,82.4307118],[GRAY,TEAL,TEAL,TEAL],'Комфорт',100)
badge(s,8.15,2.1,4.5,'28','автоматизованих тестів');badge(s,8.15,3.55,4.5,'7','предметних перевірок правил');badge(s,8.15,5.0,4.5,'74–76%','команд відрізняються від правил')
note(s,'Три навчальні seeds 42, 43, 44 мають власний validation-відбір; на спільному test seed 61000 усі підвищили комфорт і зменшили unsafe, але витрачали більше енергії. Епохи 30,10,25 відповідно. Кращий test seed 43 не використано для заміни канонічного 42. Відмінні фінальні команди: 2130/2880 та 2190/2880 на test seeds 71000 і 81000 для канонічної моделі. Це не показник точності класифікації. Три студенти на одних 20 сценаріях не є 60 незалежними будинками.')
# 10
s=new(10,'Висновки та наступний етап')
card(s,.65,2.0,7.65,1.3,'Агентна система працює','Імпорт, рекомендації, правила та веблабораторія.',TEAL)
card(s,.65,3.55,7.65,1.3,'Мережа має корисну роль','Швидко відтворює вибір модельного планувальника.',TEAL)
card(s,.65,5.1,7.65,1.3,'Перевага має визначені межі','Більше комфорту й менше порушень, але більше енергії.',AMBER)
rect(s,8.65,2.0,4.05,4.4,NAVY,radius=True);text(s,8.9,2.3,3.55,.8,'НАСТУПНИЙ ЕТАП',18,'86D5C2',True);text(s,8.9,3.1,3.55,2.8,'Додати фактичні\nвимірювання датчиків.\n\nПеревірити калібрування,\nпропуски та якість\nрекомендацій.',22,WHITE)
note(s,'Практичний внесок — прозорий, відтворюваний стенд і швидка політика корекцій. На 20 синтетичних станах медіана inference 0.177 мс, планувальника 43.638 мс, без SQLite/HTTP. Оптимальність апроксимації і переносимість на фізичний будинок не доведено. Натурні дані очікуються; цей етап має власний паспорт вимірювань. Репозиторій: https://github.com/Lowckick/ai_house_model/tree/feat/planned-policy-training .')
P.core_properties.title='Агентна система керування мікрокліматом розумного будинку';P.core_properties.author='Головко Ярослав Ігорович';P.core_properties.subject='Магістерська робота: оновлена модель та перевірені результати'
P.save(O/'Презентація_Головко_10_слайдів.pptx')
with (O/'Дані_графіків.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['metric','deterministic','model'])
 for k in metrics:w.writerow([k,means['deterministic'][k],means['model'][k]])
(O/'ПРОЧИТАТИ.md').write_text('''# Презентація на 10 слайдів

Редагований PowerPoint: тексти, блок-схеми та графіки — нативні об’єкти. Лише фактичний знімок вебпанелі є зображенням. У нотатках до кожного слайда є пояснення й джерело даних. Формат 16:9, гарнітура Liberation Sans для однакового відображення у PPTX і PDF.

Використано актуальний метод дистиляції політики від модельного планувальника й checkpoint planned-v1.pt, а не вироджену політику з постійною дією 0. Не заявляється успіх online RL або реальна економія енергії. Усі кількісні результати — симуляційні; натурні результати фізичних датчиків очікуються.

Слайд 8: канонічний навчальний seed 42, 60 епізодів (3 test seeds по 20). Слайд 9: три різні навчальні seed на спільних 20 сценаріях — інший дизайн порівняння. 74–76% — частка фінальних команд, відмінних від правил, а не «точність моделі».

Дані: docs/planned-policy.md, data/models/planned-v1-evaluation.json, docs/experiments/planned-v1. Автор: Головко Ярослав Ігорович. Науковий керівник: доц. Мисюк Р. В.
''')
print('Created',len(P.slides),'slides')
