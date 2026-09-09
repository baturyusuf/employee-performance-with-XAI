from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

ROOT=Path(__file__).resolve().parents[3]/'reports/submission_eswa/qc'
for folder in ['render_main','render_supp','render_supp_v2']:
    paths=sorted((ROOT/folder).glob('page-*.png'))
    thumbs=[]
    for i,p in enumerate(paths,1):
        im=Image.open(p).convert('RGB'); im.thumbnail((330,460))
        canvas=Image.new('RGB',(350,500),'white'); canvas.paste(im,((350-im.width)//2,25))
        ImageDraw.Draw(canvas).text((8,5),f'Page {i}',fill='black')
        thumbs.append(canvas)
    for start in range(0,len(thumbs),12):
        sheet=Image.new('RGB',(1400,1500),'#dddddd')
        for j,im in enumerate(thumbs[start:start+12]): sheet.paste(im,((j%4)*350,(j//4)*500))
        sheet.save(ROOT/f'{folder}_contact_{start//12+1}.png')
