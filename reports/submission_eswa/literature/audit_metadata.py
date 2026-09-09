"""Read-only public bibliographic metadata audit; no model or paid service calls."""
import concurrent.futures, datetime, hashlib, html, json, pathlib, re, urllib.request, urllib.parse

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = pathlib.Path(__file__).resolve().parent

def parse_bib(text):
    entries=[]
    for match in re.finditer(r'@([a-zA-Z]+)\{([^,]+),',text):
        start=match.end(); level=1; pos=start
        while pos<len(text) and level:
            if text[pos]=='{': level+=1
            elif text[pos]=='}': level-=1
            pos+=1
        body=text[start:pos-1]; fields={}
        for m in re.finditer(r'(\w+)\s*=\s*\{',body):
            begin=m.end(); end=begin; depth=1
            while end<len(body) and depth:
                if body[end]=='{': depth+=1
                elif body[end]=='}': depth-=1
                end+=1
            fields[m.group(1)]=body[begin:end-1]
        entries.append({'key':match[2], 'type':match[1], **fields})
    return entries

def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'ESWA-submission-reference-audit/1.0 (public metadata; non-commercial research)', 'Accept':'application/json,text/html;q=0.9,*/*;q=0.5'})
    try:
        with urllib.request.urlopen(req,timeout=35) as r:
            data=r.read()
            return {'http_status':r.status,'resolved_url':r.url,'sha256':hashlib.sha256(data).hexdigest(),'body':data.decode('utf-8','replace')}
    except Exception as e:
        return {'error':str(e)}

def audit(entry):
    doi=entry.get('doi','')
    url='https://api.crossref.org/works/'+urllib.parse.quote(doi,safe='') if doi else entry.get('url','')
    response=get(url)
    result={'key':entry['key'],'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_bib':entry,'source_url':url,**{k:v for k,v in response.items() if k!='body'}}
    if 'body' in response:
        if doi:
            msg=json.loads(response['body'])['message']
            result['registered_metadata']={k:msg[k] for k in ['DOI','title','subtitle','author','container-title','published','published-print','published-online','issued','volume','issue','page','article-number','type','URL','link','relation','update-to','updated-by','assertion','publisher','resource'] if k in msg}
        else:
            body=response['body']
            result['primary_metadata_tags']=[html.unescape(tag) for tag in re.findall(r'<meta\b[^>]*>',body,re.I) if re.search(r'citation_|dc\.|DC\.|og:title',tag)]
            result['page_title']=re.findall(r'<title[^>]*>(.*?)</title>',body,re.S|re.I)
    return result

if __name__=='__main__':
    entries=parse_bib((ROOT/'manuscript/mdpi_information/references.bib').read_text(encoding='utf-8-sig'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(audit,entries))
    (OUT/'existing_reference_metadata.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for row in results:
        meta=row.get('registered_metadata',{})
        print(json.dumps({'key':row['key'],'status':row.get('http_status',row.get('error')),'title':meta.get('title'),'year':meta.get('published'),'print':meta.get('published-print'),'updates':meta.get('update-to',meta.get('updated-by'))},ensure_ascii=False))
