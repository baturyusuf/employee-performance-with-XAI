import concurrent.futures, json
from audit_metadata import audit, get, OUT

CANDIDATES=[
 ('lofstrom2024calibrated','10.1016/j.eswa.2024.123154'),
 ('sepulveda2025enhancing','10.1016/j.eswa.2025.126922'),
 ('manafivarkiani2025predicting','10.1016/j.eswa.2025.126575'),
 ('rass2026statistically','10.1016/j.eswa.2025.130156'),
 ('abusitta2024survey','10.1016/j.eswa.2024.124710'),
 ('kapoor2024reforms','10.1126/sciadv.adk3452'),
 ('bargil2024ai','10.1016/j.techsoc.2024.102527'),
 ('lones2024avoiding','10.1016/j.patter.2024.101046'),
]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    rows=list(pool.map(audit,[{'key':k,'doi':d} for k,d in CANDIDATES]))
(OUT/'addition_metadata.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for r in rows:
    m=r.get('registered_metadata',{})
    print(json.dumps({'key':r['key'],'status':r.get('http_status',r.get('error')),'title':m.get('title'),'authors':m.get('author'),'volume':m.get('volume'),'issue':m.get('issue'),'pages':m.get('page'),'article':m.get('article-number'),'published':m.get('published'),'print':m.get('published-print')},ensure_ascii=False))
