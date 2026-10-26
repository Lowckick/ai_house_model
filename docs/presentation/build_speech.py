from pathlib import Path
import json,re
from docx import Document
from docx.shared import Pt,Cm,RGBColor
R=Path(__file__).resolve().parent;O=R/'generated';O.mkdir(exist_ok=True)
rows=json.loads((R/'speech.json').read_text())
intro='Орієнтовна тривалість — 6–7 хвилин із короткими паузами. Час біля слайдів є планом для репетиції. Текст синхронізовано з оновленими 10 слайдами; підказки «Що показати» не потрібно зачитувати. Під час виступу достатньо статичних слайдів, запускати застосунок не обов’язково.'
md='# Текст виступу до презентації\n\n'+intro+'\n\n'
for r in rows:md+=f"## Слайд {r['slide']}. {r['title']} · {r['time']}\n\n**Що показати:** {r['cue']}\n\n{r['text']}\n\n"
md+='## Якщо потрібно вкластися у 4–5 хвилин\n\nСкоротіть пояснення схеми на слайді 3, деталі міток і епох на слайді 6 та опис інтерфейсу на слайді 8. Для об’єкта і предмета збережіть окремі чіткі формулювання. Збережіть пояснення новизни, результати зі зростанням енергії та висновок про необхідність натурної перевірки.\n'
(O/'Виступ_до_презентації.md').write_text(md)
doc=Document();sec=doc.sections[0];sec.top_margin=sec.bottom_margin=Cm(1.7);sec.left_margin=sec.right_margin=Cm(2)
style=doc.styles['Normal'];style.font.name='Liberation Sans';style.font.size=Pt(11);style.paragraph_format.line_spacing=1.15;style.paragraph_format.space_after=Pt(7)
for sty in ['Title','Heading 1','Heading 2']:doc.styles[sty].font.name='Liberation Sans';doc.styles[sty].font.color.rgb=RGBColor.from_string('087E83')
doc.add_heading('Виступ до магістерської роботи',0);doc.add_paragraph('Головко Ярослав Ігорович · Науковий керівник: доц. Мисюк Р. В.')
doc.add_paragraph(intro)
for r in rows:
 doc.add_heading(f"Слайд {r['slide']}. {r['title']} · {r['time']}",level=2)
 p=doc.add_paragraph();p.paragraph_format.keep_with_next=True;p.add_run('Що показати: ').bold=True;p.add_run(r['cue']).italic=True
 for idx,para in enumerate(r['text'].split('\n')):
  p=doc.add_paragraph(para);p.paragraph_format.keep_together=True;p.paragraph_format.keep_with_next=idx<len(r['text'].split('\n'))-1
doc.core_properties.author='Головко Ярослав Ігорович';doc.save(O/'Виступ_до_презентації.docx')
print('Speech words:',sum(len(re.findall(r"\S+",r['text'])) for r in rows))
