from pathlib import Path
import re,json,copy,sys,subprocess
from docx import Document
from docx.shared import Cm,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.enum.section import WD_SECTION_START
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image
ROOT=Path(__file__).resolve().parent; OUT=ROOT.parent/'Згенеровані_документи';OUT.mkdir(parents=True,exist_ok=True)
TITLE='Імплементація агентної системи штучного інтелекту для керування мікрокліматом розумного будинку з використанням датчиків клімату'
AUTHOR='Головко Ярослав Ігорович';SUPERVISOR='доц. Мисюк Р. В.'
E=json.loads((ROOT/'aggregate.json').read_text());M=json.loads((ROOT/'measurements.json').read_text());V=E['summary']['model']
def num(x,n=3):return f'{x:.{n}f}'.replace('.',',')
META={'comfort':num(V['avg_comfort'],2),'energy':num(V['energy_kwh'],2),'reward':num(V['total_reward'],2),'unsafe':num(V['unsafe_steps'],2),'cached':num(M['cached_median_ms']),'uncached':num(M['uncached_median_ms']),'cached_p95':num(M['cached_p95_ms']),'uncached_p95':num(M['uncached_p95_ms']),'cold':num(M['cold_ms']),'speedup':num(M['uncached_median_ms']/M['cached_median_ms'],2),'reduction':num(100*(M['uncached_median_ms']-M['cached_median_ms'])/M['uncached_median_ms'],1),'digest':M['checkpoint_sha256'],'commit':'82a624b168e334f26c191276c95fd98cb36f679d','python':M['python'],'torch':M['torch'],'numpy':M['numpy']}
for _key in ['comfort','energy','uncached','cached']:META[_key+'_en']=META[_key].replace(',','.')
def fill(s):
 for k,v in META.items():s=s.replace('{{'+k+'}}',str(v))
 return s
latex_groups=[
 [r'z(s,a)=V(s)+A(s,a)-\frac{1}{|\mathcal{A}|}\sum_{b\in\mathcal{A}}A(s,b)'],
 [r'a_t=\pi(s_t),\quad s_{t+1}\sim P(\cdot|s_t,a_t),\quad J(\pi)=\mathbb{E}\!\left[\sum_{t=0}^{T-1}\gamma^t r_t\right]'],
 [r's=\operatorname{concat}(w/\kappa_w,x/\kappa_x,u/\kappa_u,\tau)',r'\tau=(\sin(2\pi h/24),\cos(2\pi h/24),\sin(2\pi d/365),\cos(2\pi d/365))'],
 [r'u_t=S(s_t,C(b(s_t),a_t)),\quad a_t\in\{0,\ldots,18\}'],
 [r'T_{t+1}=T_t+(q_{out}+q_{HVAC}+q_{occ}+q_{solar})\Delta t+\xi_t',r'q_{out}=k_{out}(T_{out}-T_t)'],
 [r'E_{episode}=\sum_t P_t\Delta t,\qquad\Delta t=10/60\;\mathrm{h}'],
 [r'r_t=2(C_t/50-1)+0.2(L_t/50-1)-0.15P_t-0.03N_{sw}-P_{dev}-3I_{unsafe}',r'P_{dev}=0.15\delta_T+0.015\delta_H+0.3\delta_{CO_2}+0.2\delta_{PM}'],
 [r'a^*=\arg\max_a\sum_{k=0}^{H-1}\gamma^k(r_{t+k}-10I_{unsafe})',r'L=-\log \operatorname{softmax}(z(s))_{a^*}'],
 [r'K(\pi)=(-U_{val},C_{val},-E_{val},R_{val})',r'best=\arg\max_{lex}K'],
 [r'\bar d=\frac1n\sum_i d_i,\qquad s_d=\sqrt{\frac{\sum_i(d_i-\bar d)^2}{n-1}}'],
 [r'reduction=100\,\frac{median_{uncached}-median_{cached}}{median_{uncached}}']
]
(ROOT/'equations.md').write_text('\n\n'.join('$$\n'+x+'\n$$' for group in latex_groups for x in group))
subprocess.run(['pandoc',str(ROOT/'equations.md'),'-o',str(ROOT/'equations.docx')],check=True)
ED=Document(ROOT/'equations.docx');OM=ED.element.xpath('//m:oMath')
MATH_GROUPS=[];_offset=0
for _group in latex_groups:
 MATH_GROUPS.append(OM[_offset:_offset+len(_group)]);_offset+=len(_group)
PAGE_MAP=json.loads((ROOT/'page-map.json').read_text()) if (ROOT/'page-map.json').exists() else {}
HEADINGS=[]
def runfont(run,name='Times New Roman',size=None):
 run.font.name=name
 if size:run.font.size=Pt(size)
 p=run._element.get_or_add_rPr();f=p.find(qn('w:rFonts'))
 if f is None:f=OxmlElement('w:rFonts');p.insert(0,f)
 for k in ['ascii','hAnsi','eastAsia','cs']:f.set(qn('w:'+k),name)
 lang=OxmlElement('w:lang');lang.set(qn('w:val'),'uk-UA');p.append(lang)
def styles(doc):
 for sec in doc.sections:
  sec.page_width=Cm(21);sec.page_height=Cm(29.7);sec.top_margin=Cm(2.2);sec.bottom_margin=Cm(2.2);sec.left_margin=Cm(2.5);sec.right_margin=Cm(1.0);sec.header_distance=Cm(.9);sec.footer_distance=Cm(1.0)
 normal=doc.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(14)
 normal.paragraph_format.line_spacing=1.5;normal.paragraph_format.space_after=Pt(0);normal.paragraph_format.first_line_indent=Cm(1.25);normal.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
 normal.paragraph_format.widow_control=True
 for name,size in [('Heading 1',14),('Heading 2',14),('Heading 3',14)]:
  st=doc.styles[name];st.font.name='Times New Roman';st.font.size=Pt(size);st.font.bold=True;st.font.color.rgb=RGBColor(0,0,0)
  f=st.paragraph_format;f.keep_with_next=True;f.keep_together=True;f.space_before=Pt(18);f.space_after=Pt(18);f.line_spacing=1.5
  f.first_line_indent=Cm(0 if name=='Heading 1' else 1.25)
  f.alignment=WD_ALIGN_PARAGRAPH.CENTER if name=='Heading 1' else WD_ALIGN_PARAGRAPH.JUSTIFY
 for st in doc.styles:
  if st.type==1 and st.name.startswith('TOC'):st.font.name='Times New Roman'
 settings=doc.settings.element
 compat=OxmlElement('w:updateFields');compat.set(qn('w:val'),'true');settings.append(compat)
 doc.core_properties.author=AUTHOR;doc.core_properties.title=TITLE;doc.core_properties.subject='Магістерська робота — чорновий варіант станом на 26.10.2026';doc.core_properties.keywords='мікроклімат, дистиляція політики, детермінована система'
def blank_header(section):
 section.header.is_linked_to_previous=False
 for p in section.header.paragraphs:p.clear()
def numbered_header(section):
 section.header.is_linked_to_previous=False;p=section.header.paragraphs[0];p.clear();p.alignment=WD_ALIGN_PARAGRAPH.RIGHT;p.paragraph_format.first_line_indent=Cm(0)
 r=p.add_run();runfont(r,size=12);field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');r._r.addnext(field)
def section(doc,number=False):
 s=doc.add_section(WD_SECTION_START.NEW_PAGE)
 if number:numbered_header(s)
 else:blank_header(s)
 return s
def ptext(doc,text='',size=None,align=None,bold=False,indent=True,spacing=None):
 p=doc.add_paragraph();p.paragraph_format.first_line_indent=Cm(1.25 if indent else 0)
 if align is not None:p.alignment=align
 if spacing:p.paragraph_format.line_spacing=spacing
 r=p.add_run(text);runfont(r,size=size);r.bold=bold;return p
def cover(doc,kind):
 blank_header(doc.sections[0])
 p=ptext(doc,'ЧОРНОВИЙ ВАРІАНТ',14,WD_ALIGN_PARAGRAPH.CENTER,True,False,1.15);p.paragraph_format.space_before=Pt(70)
 ptext(doc,'Магістерська робота' if kind=='thesis' else 'Звіт про підготовлені матеріали практики',18,WD_ALIGN_PARAGRAPH.CENTER,True,False,1.15)
 p=ptext(doc,TITLE,18,WD_ALIGN_PARAGRAPH.CENTER,True,False,1.25);p.paragraph_format.space_before=Pt(35);p.paragraph_format.space_after=Pt(45)
 ptext(doc,'Автор: '+AUTHOR,14,WD_ALIGN_PARAGRAPH.CENTER,indent=False,spacing=1.15)
 ptext(doc,'Науковий керівник: '+SUPERVISOR,14,WD_ALIGN_PARAGRAPH.CENTER,indent=False,spacing=1.15)
 p=ptext(doc,'Період робочого плану: вересень — жовтень 2026 року',13,WD_ALIGN_PARAGRAPH.CENTER,indent=False,spacing=1.15);p.paragraph_format.space_before=Pt(35)
 ptext(doc,'Станом на 26 жовтня 2026 року',13,WD_ALIGN_PARAGRAPH.CENTER,indent=False,spacing=1.15)
 ptext(doc,'Львів, 2026',14,WD_ALIGN_PARAGRAPH.CENTER,indent=False,spacing=1.15)
def assignment(doc):
 section(doc)
 ptext(doc,'РОБОЧИЙ ПЛАН ЧЕРНЕТКИ',14,WD_ALIGN_PARAGRAPH.CENTER,True,False,1.15)
 ptext(doc,'Календар охоплює 01.09–26.10.2026. Дати до 26 жовтня є орієнтовним розподілом готових етапів для чернетки; це не первинний журнал виконання. Контрольні запуски та оновлення результатів зафіксовано 26.10.2026.',13,spacing=1.15)
 table(doc,[['Етап','Період','Стан'],['Аналіз предметної області та наукових праць','01.09–10.09.2026','підготовлено'],['Модель станів, корекцій та архітектура','11.09–20.09.2026','підготовлено'],['Реалізація сервісу, імпорту та веблабораторії','21.09–15.10.2026','готово'],['Підготовка перевірок та початкового експерименту','16.10–25.10.2026','готово'],['Діагностика дії 0, новий метод, навчання і test','26.10.2026','виконано'],['28 тестів, 7 перевірок, вимір сервісної затримки','26.10.2026','виконано'],['Оновлення тексту, графіків, джерел і висновків','26.10.2026','чернетка готова'],['Фізичні показники датчиків і натурний аналіз','після отримання даних','не завершено']])
 ptext(doc,'Наукові результати в тексті стосуються симуляції. Для натурного етапу дата завершення не призначається. Публікація, усний виступ і захист цим планом не засвідчуються.',13,spacing=1.15)
def table(doc,rows):
 t=doc.add_table(rows=1,cols=len(rows[0]));t.style='Table Grid';t.autofit=False
 usable=17.5
 if len(rows[0])==2:widths=[4.6,12.9]
 elif len(rows[0])==3:widths=[4.5,6.5,6.5]
 elif len(rows[0])==4:widths=[4.6,4.3,4.3,4.3]
 elif len(rows[0])==5:widths=[2.1,3.9,4.0,3.8,3.7]
 elif len(rows[0])==7:widths=[.9,4.1,1.9,1.9,2.7,2.7,3.3]
 else:widths=[usable/len(rows[0])]*len(rows[0])
 for col,w in zip(t.columns,widths):col.width=Cm(w)
 for ri,row in enumerate(rows):
  cells=t.rows[0].cells if ri==0 else t.add_row().cells
  for cell,w,val in zip(cells,widths,row):
   cell.width=Cm(w);p=cell.paragraphs[0];p.paragraph_format.first_line_indent=Cm(0);p.paragraph_format.line_spacing=1.0;p.paragraph_format.space_after=Pt(4);p.paragraph_format.space_before=Pt(4);p.alignment=WD_ALIGN_PARAGRAPH.LEFT
   if len(rows)<=6 and ri<len(rows)-1:p.paragraph_format.keep_with_next=True
   r=p.add_run(str(val));runfont(r,size=11 if len(row)>=5 else 12);r.bold=ri==0
   cell.vertical_alignment=1
   if ri==0:
    shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'EEEEEE');cell._tc.get_or_add_tcPr().append(shade)
  trpr=t.rows[ri]._tr.get_or_add_trPr();keep=OxmlElement('w:cantSplit');trpr.append(keep)
 head=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(head)
 doc.add_paragraph().paragraph_format.space_after=Pt(0)
 return t
def bookmark(p,name,i):
 a=OxmlElement('w:bookmarkStart');a.set(qn('w:id'),str(i));a.set(qn('w:name'),name);b=OxmlElement('w:bookmarkEnd');b.set(qn('w:id'),str(i));p._p.insert(0,a);p._p.append(b)
def heading(doc,text,level,break_before=True):
 p=doc.add_paragraph(text,style=f'Heading {level}')
 if level==1 and break_before:p.paragraph_format.page_break_before=True
 for r in p.runs:runfont(r,size=14)
 return p
EQ_INDEX=0
def parse(doc,text,collect=False,first_break=True):
 global EQ_INDEX
 lines=fill(text).splitlines();i=0;first=True
 while i<len(lines):
  line=lines[i].strip()
  if not line:i+=1;continue
  if line.startswith('```'):
   i+=1
   while i<len(lines) and not lines[i].startswith('```'):
    p=doc.add_paragraph();p.paragraph_format.first_line_indent=Cm(0);p.paragraph_format.line_spacing=1.0;p.paragraph_format.space_after=Pt(0);p.paragraph_format.widow_control=False
    r=p.add_run(lines[i]);runfont(r,'Liberation Mono',9.0);i+=1
   i+=1;continue
  if line.startswith('#'):
   m=re.match(r'^(#{1,3}) (.+)$',line)
   if m:
    level=len(m[1]);title=m[2];p=heading(doc,title,level,first_break if first else True)
    if collect:
     name='h'+str(len(HEADINGS));bookmark(p,name,len(HEADINGS));HEADINGS.append((title,level,name))
    first=False;i+=1;continue
  if line.startswith('|'):
   rows=[]
   while i<len(lines) and lines[i].strip().startswith('|'):
    vals=[x.strip() for x in lines[i].strip().strip('|').split('|')]
    if not all(re.fullmatch(r'[:\-\s]+',v) for v in vals):rows.append(vals)
    i+=1
   table(doc,rows);continue
  if line.startswith('{{FIG:'):
   name,caption=line[len('{{FIG:'):-2].split('|',1)
   image=ROOT/'assets'/f'{name}.png';w,h=Image.open(image).size
   maxw=17.0;maxh=16.5 if name=='network' else (19.5 if name in ('classes','training','sensors') else 15.5)
   width=min(maxw,maxh*w/h);p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.first_line_indent=Cm(0);p.paragraph_format.keep_with_next=True
   p.add_run().add_picture(str(image),width=Cm(width))
   p=ptext(doc,caption,12,WD_ALIGN_PARAGRAPH.CENTER,indent=False,spacing=1.15);p.paragraph_format.space_after=Pt(10);i+=1;continue
  if line.startswith('[EQ]'):
   number=re.search(r'\((\d+\.\d+)\)\s*$',line)[1]
   group=MATH_GROUPS[EQ_INDEX];EQ_INDEX+=1
   for index,source_math in enumerate(group):
    p=doc.add_paragraph();p.paragraph_format.first_line_indent=Cm(0);p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.line_spacing=1.2;p.paragraph_format.space_before=Pt(14 if index==0 else 3);p.paragraph_format.space_after=Pt(14 if index==len(group)-1 else 3);p.paragraph_format.keep_with_next=index<len(group)-1
    math=copy.deepcopy(source_math)
    for mr in math.iter(qn('m:r')):
     pr=mr.find(qn('w:rPr'))
     if pr is None:pr=OxmlElement('w:rPr');mr.insert(0,pr)
     sz=OxmlElement('w:sz');sz.set(qn('w:val'),'28');pr.append(sz)
    p._p.append(math)
    if index==len(group)-1:runfont(p.add_run('   ('+number+')'),size=12)
   i+=1;continue
  if line.startswith('Таблиця '):
   p=ptext(doc,line,12,WD_ALIGN_PARAGRAPH.LEFT,True,False,1.15);p.paragraph_format.keep_with_next=True;i+=1;continue
  if line.startswith('- ') or re.match(r'^\d+\. ',line):
   p=ptext(doc,('– '+line[2:]) if line.startswith('- ') else line);p.paragraph_format.left_indent=Cm(.5);p.paragraph_format.first_line_indent=Cm(.75);i+=1;continue
  if line=='ABSTRACT':
   p=ptext(doc,line,14,WD_ALIGN_PARAGRAPH.CENTER,True,False,1);p.paragraph_format.space_before=Pt(10);i+=1;continue
  para=[line];i+=1
  while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','```','{{FIG:','[EQ]','- ')):
   para.append(lines[i].strip());i+=1
  p=ptext(doc,' '.join(para))
  if 'СТАТУС:' in line or '[ДОДАТИ]' in line:p.runs[0].bold=True
 return doc
def toc(doc,headings):
 p=heading(doc,'ЗМІСТ',1,False)
 for title,level,name in headings:
  if level>2 or (title.startswith(('А.','Б.','В.','Г.','Д.')) and level==2):continue
  p=doc.add_paragraph();f=p.paragraph_format;f.first_line_indent=Cm(0);f.left_indent=Cm(.4 if level==2 else 0);f.line_spacing=1.0;f.space_after=Pt(2);f.tab_stops.add_tab_stop(Cm(17.2-(.4 if level==2 else 0)),WD_TAB_ALIGNMENT.RIGHT,WD_TAB_LEADER.DOTS)
  link=OxmlElement('w:hyperlink');link.set(qn('w:anchor'),name)
  r=OxmlElement('w:r');pr=OxmlElement('w:rPr');font=OxmlElement('w:rFonts');font.set(qn('w:ascii'),'Times New Roman');font.set(qn('w:hAnsi'),'Times New Roman');pr.append(font);size=OxmlElement('w:sz');size.set(qn('w:val'),'24');pr.append(size)
  if level==1:pr.append(OxmlElement('w:b'))
  r.append(pr);t=OxmlElement('w:t');t.text=title;r.append(t);link.append(r);p._p.append(link)
  runfont(p.add_run('\t'+str(PAGE_MAP.get(title,'…'))),size=12)
def main():
 global HEADINGS,EQ_INDEX
 doc=Document();styles(doc);cover(doc,'thesis');assignment(doc)
 text=(ROOT/'main.md').read_text();ann,tail=text.split('# ПЕРЕЛІК СКОРОЧЕНЬ',1);abbrev,rest=('# ПЕРЕЛІК СКОРОЧЕНЬ'+tail).split('# ВСТУП',1)
 section(doc);parse(doc,ann,False,False)
 # Collect bookmarks and build body first in a temporary document, then copy it after the contents.
 body=Document();styles(body);HEADINGS=[];EQ_INDEX=0
 parse(body,abbrev+'# ВСТУП'+rest,True,False)
 references=json.loads((ROOT/'references.json').read_text())
 ph=heading(body,'СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ',1);name='h'+str(len(HEADINGS));bookmark(ph,name,len(HEADINGS));HEADINGS.append(('СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ',1,name))
 for n,(title,url) in enumerate(references,1):
  p=ptext(body,f'{n}. {title} {url}',12,spacing=1.1);p.paragraph_format.space_after=Pt(6);p.paragraph_format.first_line_indent=Cm(0)
 parse(body,(ROOT/'appendices.md').read_text(),True)
 (ROOT/'headings.json').write_text(json.dumps(HEADINGS,ensure_ascii=False,indent=2))
 section(doc);toc(doc,HEADINGS)
 section(doc,True)
 for child in list(body.element.body):
  if child.tag!=qn('w:sectPr'):doc.element.body.insert(len(doc.element.body)-1,copy.deepcopy(child))
 # Images and hyperlinks from temporary body need relationship remapping.
 relationships={}
 for old,rel in body.part.rels.items():
  if 'image' in rel.reltype:relationships[old]=doc.part.relate_to(rel.target_part,rel.reltype)
 for blip in doc.element.xpath('//a:blip'):
  old=blip.get(qn('r:embed'))
  if old in relationships:blip.set(qn('r:embed'),relationships[old])
 doc.save(OUT/'01_Магістерська_робота.docx')
 practice=Document();styles(practice);cover(practice,'practice');section(practice,True);parse(practice,(ROOT/'practice.md').read_text(),False,False);practice.save(OUT/'02_Практика_чернетка.docx')
 seminar=Document();styles(seminar);numbered_header(seminar.sections[0]);parse(seminar,(ROOT/'seminar.md').read_text(),False,False);seminar.save(OUT/'03_Семінарний_рукопис.docx')
 print('DOCX files created',OUT)
if __name__=='__main__':main()
