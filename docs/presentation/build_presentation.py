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
def new(n,title,tag='СЕМІНАР · МАГІСТЕРСЬКА РОБОТА'):
 s=P.slides.add_slide(P.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=color(BG)
 rect(s,.45,.42,.10,.32,TEAL);text(s,.72,.40,10,.3,tag,12,TEAL,True);text(s,.65,.95,12.0,.9,title,32,INK,True)
 rect(s,.65,7.02,12.03,.012,LIGHT);text(s,.65,7.13,11,.2,'Головко Ярослав Ігорович  ·  26 жовтня 2026',11,GRAY);text(s,12.05,7.08,.6,.3,f'{n:02d} / 10',11,GRAY,align=PP_ALIGN.RIGHT)
 return s
def note(s,t):s.notes_slide.notes_text_frame.text=t+'\n\nДжерела: оновлена магістерська, коміт 0a79f0b10ff0d9a9d6f3bfd2f924cf643469a701; data/models/planned-v1-evaluation.json; код методу 82a624b168e334f26c191276c95fd98cb36f679d.'
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
# 1. Титул
s=P.slides.add_slide(P.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=color(NAVY)
text(s,.65,.55,11.9,.4,'СЕМІНАР · МАГІСТЕРСЬКА РОБОТА · ЧЕРНЕТКА',16,'86D5C2',True)
text(s,.65,1.85,9.0,3.25,'Імплементація агентної системи\nштучного інтелекту для керування\nмікрокліматом розумного будинку\nз використанням датчиків клімату',29,WHITE,True)
rect(s,.67,5.20,1.2,.055,'53C7B3')
text(s,.65,5.6,9.0,1.1,'Головко Ярослав Ігорович\nНауковий керівник: доц. Мисюк Р. В.',21,WHITE)
text(s,.65,7.05,10,.2,'ЛЬВІВ · 26 ЖОВТНЯ 2026',11,'BCCFD3')
roof=s.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE,Inches(10.0),Inches(2.2),Inches(2.65),Inches(1.8));roof.fill.solid();roof.fill.fore_color.rgb=color(TEAL);roof.line.fill.background()
rect(s,10.35,3.5,1.95,1.8,'244B5A',radius=True);rect(s,10.65,3.85,.45,.45,'86D5C2');rect(s,11.5,3.85,.45,.45,'86D5C2');rect(s,11.05,4.55,.5,.75,'86D5C2')
text(s,9.7,5.65,3.0,.8,'СТАН → АГЕНТ\n→ РЕКОМЕНДАЦІЇ',16,'86D5C2',True,PP_ALIGN.CENTER)
note(s,'Порядок слайдів 2–10 відповідає дев’яти пунктам семінарного плану: актуальність, об’єкт, предмет, задача, рішення, дослідження, розробка ПЗ, тестування, висновки. Натурні вимірювання ще не аналізувалися.')
# 2. Актуальність
s=new(2,'Актуальність роботи')
card(s,.65,2.0,3.85,2.65,'Якість мікроклімату','Температура, вологість,\nCO₂ та PM2.5\nвпливають на умови\nперебування мешканців.')
card(s,4.75,2.0,3.85,2.65,'Конфлікт цілей','Вентиляція покращує\nповітрообмін, але може\nзбільшувати тепловтрати\nй споживання енергії.')
card(s,8.85,2.0,3.85,2.65,'Потреба в перевірці','Користь агента\nперевіряється через\nпорівняння з правилами\nза однакових умов.')
rect(s,.65,5.05,12.05,1.3,NAVY,radius=True)
text(s,.95,5.3,11.4,.8,'Потрібна система, яка узгоджує комфорт, обмеження та енергію\nі дозволяє перевірити результат за однакових умов.',24,WHITE)
note(s,'Актуальність стосується багатокритеріального керування та перевірюваності агентних рішень. Вплив показників не інтерпретується як виміряний медичний ефект. Порівняння з простим контролером необхідне для обґрунтування додаткової складності.')
# 3. Об’єкт
s=new(3,'Об’єкт дослідження')
text(s,.8,1.95,11.8,1.3,'Процеси автоматизованого керування мікрокліматом\nі пов’язаними інженерними підсистемами\nжитлового середовища.',28,INK,True)
for x,title,body in [(.65,'Зовнішні збурення','Погода, присутність,\nактивність мешканців'),(4.75,'Стан будинку','Температура, вологість,\nякість повітря'),(8.85,'Керувальні впливи','Опалення, вентиляція,\nочищення, зволоження')]:
 card(s,x,3.7,3.85,1.85,title,body)
arrow(s,4.5,4.6,4.75,4.6);arrow(s,8.85,4.6,8.6,4.6)
text(s,.85,6.05,11.7,.65,'Об’єкт оцінюється в замкненій симуляції: команда змінює стан,\nа наступне рішення формується вже за оновленими показниками.',21,GRAY)
note(s,'Об’єкт — процеси керування, а не сама нейромережа. Погода й мешканці є збуреннями; режим обладнання є керованою дією. Фізичні пристрої не підключено до замкненого контуру; кількісні висновки стосуються симулятора.')
# 4. Предмет
s=new(4,'Предмет дослідження')
text(s,.8,1.95,11.8,1.15,'Моделі, алгоритми та програмні засоби поєднання\nдетермінованого керування, нейромережних корекцій\nі правил обмеження дій.',27,INK,True)
card(s,.65,3.5,3.85,1.7,'Подання стану','54 числові ознаки\nта 19 корекцій')
card(s,4.75,3.5,3.85,1.7,'Навчання політики','Мітки планувальника\n→ дистиляція мережі')
card(s,8.85,3.5,3.85,1.7,'Оцінювання','Парні сценарії\nта аудит результатів')
rect(s,.65,5.55,12.05,1.03,MINT,radius=True)
text(s,.9,5.72,11.5,.8,'Елемент новизни: адаптація гібридної політики та її порівняння\nз детермінованою системою за комфортом, порушеннями й енергією.',20,TEAL,True)
note(s,'Предмет конкретизує спосіб дослідження об’єкта. Новизна — прикладний методичний внесок у цій постановці; модельне планування та дистиляція є відомими підходами. Порівняння охоплює реалізований детермінований контролер, а не всі можливі детерміновані системи.')
# 5. Постановка задачі
s=new(5,'Постановка задачі')
rect(s,.65,1.95,12.05,1.25,NAVY,radius=True)
text(s,.92,2.16,11.5,.9,'Мета: створити агентну систему керування мікрокліматом\nі перевірити додаткову користь навченої корекції.',25,WHITE,True)
for x,y,n,title,body in [(.65,3.45,'01','Підготувати стан','Показники, погода, час і режими.'),(6.85,3.45,'02','Обрати команду','Основа + корекція + остаточні правила.'),(.65,5.0,'03','Реалізувати застосунок','Імпорт, рекомендації, історія й вебпанель.'),(6.85,5.0,'04','Порівняти контролери','Спільні умови; комфорт, порушення, енергія.')]:
 rect(s,x,y,5.85,1.3,WHITE,radius=True);text(s,x+.2,y+.17,5.45,.4,n+'  '+title,22,TEAL,True);text(s,x+.2,y+.78,5.45,.38,body,18)
note(s,'Вхід: валідовані записи JSON/CSV/TXT або стан симулятора. Вихід: рекомендовані режими, пояснення правил та збережений результат. Задача не зводиться до мінімізації енергії; оцінюється її ціна за зміни умов. Апаратний збір і натурний аналіз не завершено.')
# 6. Запропоноване рішення
s=new(6,'Запропоноване рішення')
card(s,.65,1.95,3.85,3.0,'01  Учитель','19 корекцій у копіях\nсимулятора.\nПрогноз: 6 кроків\nпо 10 хвилин.\nОбрати кращу мітку.')
card(s,4.75,1.95,3.85,3.0,'02  Навчена мережа','24 train-епізоди\n→ 1728 прикладів.\nДо 30 епох навчання.\nВідбір моделі — лише\nна validation.')
card(s,8.85,1.95,3.85,3.0,'03  Рекомендація','Поточні 54 ознаки\n→ одна з 19 корекцій\n→ остаточні правила\n→ режими пристроїв.\nБез перебору дій.')
rect(s,.65,5.25,12.05,1.45,NAVY,radius=True)
text(s,.9,5.43,11.5,.35,'ДИСТИЛЯЦІЯ ПОЛІТИКИ ВІД МОДЕЛЬНОГО ПЛАНУВАЛЬНИКА',16,'86D5C2',True)
text(s,.9,5.83,11.5,.65,'Прогноз — лише для підготовки навчальних міток.\nПід час запиту — поточний стан без перебору дій.',19,WHITE)
note(s,'Початковий warmup використовував лише мітку 0. У діагностичних 112 епізодах усі 16128 дій мали індекс 0. Новий учитель оцінює discounted reward v2 мінус 10 за unsafe-крок; кандидат повторює індекс на горизонті, а не оптимізує всі 19^6 послідовності. Учитель має привілейований доступ до синтетичного прогнозу. Канонічний seed 42, validation seed 41000, шість сценаріїв; обрана епоха 30. Метод — model-based policy distillation. Це не доказ успіху онлайн-RL. Мережа зберігає dueling-структуру з прихованими шарами 256→256→128, виходи є логітами політики.')
# 7. Результати досліджень
s=new(7,'Результати проведених досліджень')
text(s,.75,1.85,11.9,.4,'60 парних добових test-сценаріїв · незмінна модель · однакові умови',18,GRAY)
text(s,.85,2.45,5.5,.4,'Комфорт, %  ↑',23,INK,True);text(s,6.95,2.45,5.6,.4,'Кроки поза межами / епізод  ↓',22,INK,True)
chart(s,.65,2.9,5.7,1.95,['Детермінована','Модель'],[means['deterministic']['avg_comfort'],means['model']['avg_comfort']],[GRAY,TEAL],'Комфорт',100)
chart(s,6.8,2.9,5.7,1.95,['Детермінована','Модель'],[means['deterministic']['unsafe_steps'],means['model']['unsafe_steps']],[GRAY,TEAL],'Порушення',14)
badge(s,.65,5.1,3.85,'+3,11 в.п.','середній комфорт');badge(s,4.75,5.1,3.85,'−16,80%','кроки поза межами');badge(s,8.85,5.1,3.85,'+3,57%','енергоспоживання',GOLD,AMBER)
text(s,.85,6.57,11.8,.32,'Комфорт вищий у 34 сценаріях, нижчий у 26. Усі кількісні результати — симуляційні.',17,GRAY)
note(s,'Канонічний навчальний seed 42, test seeds 61000/71000/81000 по 20 епізодів, кожен 144 кроки. Комфорт 82.6277048611 проти 79.5193043981%; енергія 25.9971805556 проти 25.100125 кВт·год; винагорода 123.4019754524 проти 91.480154 (+34.89%); unsafe 8.5833333333 проти 10.3166666667. Вибіркове SD парної різниці комфорту 10.6381 в.п. Порушення нижчі у 21, рівні у 38, вищі в одному сценарії. Статистичної значущості не встановлено. 74–76% фінальних команд відрізнялися від основи на двох наборах; це не точність. Результат — середнє покращення за додаткової енергії.')
# 8. Розробка ПЗ
s=new(8,'Розробка програмного забезпечення')
for y,title,body in [(2.0,'Спільний сервіс аналізу','Валідація JSON / CSV / TXT,\nрекомендації й пояснення.'),(3.5,'Веблабораторія','Ручний тест і порівняння\nдо трьох моделей.'),(5.0,'Збереження та швидкодія','SQLite, фонові задачі,\nкеш, історія й експорт.')]:
 text(s,.75,y,4.55,.4,title,22,TEAL,True);text(s,.75,y+.52,4.55,.8,body,20)
s.shapes.add_picture(str(R/'application.png'),Inches(5.6),Inches(1.95),width=Inches(7.08),height=Inches(4.64))
text(s,.85,6.64,11.8,.3,'Python · PyTorch · NumPy · SQLite · HTML / JavaScript',17,GRAY)
note(s,'Фактичний знімок локальної панелі з planned-v1.pt та прикладом підвищеного CO₂ під час сну. Модулі: environment — дані та фізика; ml/planning.py — мітки й навчання; analytics — аналіз, історія та порівняння; web — HTTP API та клієнт. Лише скриншот є растровим. Вебрежим не навчає модель й не виконує апаратних команд.')
# 9. Результати тестування
s=new(9,'Результати тестування')
badge(s,.65,2.0,3.85,'28','автоматизованих тестів — OK');badge(s,4.75,2.0,3.85,'7','перевірок правил — пройдено');badge(s,8.85,2.0,3.85,'3','незалежні запуски навчання')
text(s,.85,3.65,6.4,.4,'Затримка сервісного виклику, мс  ↓',22,INK,True)
ch=chart(s,.65,4.15,6.65,2.1,['Без кешу','З кешем'],[2.25751899779425,1.3058555014140438],[GRAY,TEAL],'Медіана, мс',3.5);ch.plots[0].data_labels.number_format='0.000'
text(s,7.75,3.65,4.75,.4,'Перевірено',23,TEAL,True)
text(s,7.75,4.22,4.75,2.0,'Валідацію та транзакції\nПріоритети правил\nПовторюваність і RNG\nAPI, кеш і фонову чергу',21)
text(s,.85,6.57,11.8,.32,'26.10.2026 · по 40 викликів на режим · із записом SQLite, без HTTP',17,GRAY)
note(s,'Тестовий журнал актуальної чернетки: 28 тестів OK, сім перевірок нового checkpoint пройдено. Planning-тести перевіряють повторюваність, незмінність фізики/RNG, корисну ненульову корекцію та відхилення неправильних параметрів. Медіани сервісного бенчмарку: 2.2575189978 і 1.3058555014 мс; без HTTP. Ці числа відрізняються від inference-only 0.177 мс для мережі. Три навчальні seed 42/43/44 мають власний validation-відбір; на спільних 20 test-сценаріях усі покращили середній комфорт і зменшили порушення за додаткової енергії. Це не ще 60 незалежних будинків і не відсоток покриття коду.')
# 10. Висновки
s=new(10,'Висновки')
card(s,.65,1.95,7.65,1.3,'Систему реалізовано','Агент, правила, імпорт і вебінструменти перевірки.')
card(s,.65,3.45,7.65,1.3,'Користь корекції перевірено','Вищий середній комфорт і менше порушень.')
card(s,.65,4.95,7.65,1.3,'Перевага має енергетичну ціну','Споживання вище; є сценарії з гіршим комфортом.',AMBER)
rect(s,8.65,1.95,4.05,4.3,NAVY,radius=True);text(s,8.92,2.25,3.5,.65,'ПОДАЛЬША РОБОТА',18,'86D5C2',True)
text(s,8.92,3.12,3.5,2.8,'Натурні дані датчиків\nта їх калібрування.\n\nАналіз гірших сценаріїв\nі компромісу\nкомфорт / енергія.',21,WHITE)
note(s,'Прикладний внесок включає адаптацію навченої корекції та відтворюване порівняння з детермінованою системою. Переносимість на фізичний будинок, реальну економію енергії та формальні гарантії безпеки не встановлено. Натурні дані ще очікуються. Завершальний слайд залишається на екрані під час запитань; окремого слайда подяки немає.')
# Full slide-aligned speech and technical notes.
speech=json.loads((R/'speech.json').read_text());assert len(speech)==len(P.slides)==10
for slide,item in zip(P.slides,speech):
 technical=slide.notes_slide.notes_text_frame.text
 slide.notes_slide.notes_text_frame.text=f"Слайд {item['slide']}. {item['title']}\nЧас: {item['time']}\n\nЩо показати: {item['cue']}\n\nТекст виступу\n{item['text']}\n\nДовідка для запитань\n{technical}"
P.core_properties.title='Агентна система керування мікрокліматом — семінар';P.core_properties.author='Головко Ярослав Ігорович';P.core_properties.subject='Презентація за планом семінару, станом на 26.10.2026'
P.save(O/'Презентація_Головко_10_слайдів.pptx')
with (O/'Дані_графіків.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['metric','deterministic','model'])
 for k in metrics:w.writerow([k,means['deterministic'][k],means['model'][k]])
 w.writerow(['service_latency_median_ms_without_cache',2.25751899779425,'']);w.writerow(['service_latency_median_ms_with_cache',1.3058555014140438,''])
(O/'Нотатки_до_слайдів.md').write_text('\n\n'.join(f"## Слайд {i+1}\n\n{slide.notes_slide.notes_text_frame.text}" for i,slide in enumerate(P.slides)))
print('Created',len(P.slides),'slides')
