"""Separated discovery/extraction with immutable dated snapshots and explicit failures."""
from datetime import datetime,timezone
import hashlib
import fcntl
import ipaddress
import json
from pathlib import Path
import socket
import time
from urllib.parse import urlsplit,urlunsplit
from scripts.redesign.traffic import Traffic, TrafficBlocked


class RetrievalFailure(RuntimeError):pass


def public_url(url, *, resolve=True):
    if type(url) is not str:raise ValueError('URL must be text')
    p=urlsplit(url)
    if p.scheme not in ('http','https') or not p.hostname or p.username or p.password or p.port not in (None,80,443):raise ValueError('Unsafe URL form')
    host=p.hostname.lower()
    if host=='localhost' or host.endswith(('.localhost','.local','.internal')):raise ValueError('Private host')
    try:addresses=[ipaddress.ip_address(host)]
    except ValueError:
        if not resolve:return urlunsplit((p.scheme,p.netloc,p.path or '/',p.query,''))
        addresses=[ipaddress.ip_address(x[4][0]) for x in socket.getaddrinfo(host,p.port or 443,type=socket.SOCK_STREAM)]
    if not addresses or any(not a.is_global for a in addresses):raise ValueError('Private/reserved address')
    return urlunsplit((p.scheme,p.netloc,p.path or '/',p.query,''))


def deduplicate(results):
    seen=set();out=[]
    for row in results:
        if type(row) is not dict:continue
        try:url=public_url(row.get('url') or row.get('href'),resolve=False)
        except (ValueError,TypeError):continue
        key=urlsplit(url)._replace(scheme='',netloc=urlsplit(url).netloc.lower()).geturl().rstrip('/')
        if key in seen:continue
        seen.add(key)
        out.append({'url':url,'title':row.get('title') if type(row.get('title')) is str else '',
                    'snippet':next((row[k] for k in ('content','body','snippet') if type(row.get(k)) is str),''),
                    'engines':[e for e in row.get('engines',[]) if type(e) is str] if type(row.get('engines',[])) is list else []})
    return out


class SnapshotStore:
    def __init__(self,path):self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
    def put(self,kind,payload,settings):
        doc={'kind':kind,'retrieved_at':datetime.now(timezone.utc).isoformat(),'settings':settings,'payload':payload}
        data=json.dumps(doc,sort_keys=True,ensure_ascii=False).encode()
        ident=hashlib.sha256(data).hexdigest()
        with (self.path/f'{ident}.json').open('xb') as f:f.write(data)
        return ident
    def get(self,ident):
        if type(ident) is not str or len(ident)!=64 or any(c not in '0123456789abcdef' for c in ident):raise ValueError('Invalid snapshot ID')
        data=(self.path/f'{ident}.json').read_bytes()
        if hashlib.sha256(data).hexdigest()!=ident:raise ValueError('Snapshot integrity failure')
        return json.loads(data)


def searx_result(raw,http_status,engines,cap):
    if http_status!=200:return {'status':'TOOL_ERROR','detail':'JSON_DISABLED_OR_ACCESS_DENIED' if http_status==403 else 'HTTP_ERROR','results':[],'requested_engines':engines,'responding_engines':[],'upstream_requests':None}
    if type(raw) is not dict or type(raw.get('results')) is not list:return {'status':'TOOL_ERROR','detail':'NON_JSON_OR_BAD_SCHEMA','results':[],'upstream_requests':None}
    rows=deduplicate(raw['results'])[:cap]
    failed=raw.get('unresponsive_engines',[])
    responding=sorted({e for r in rows for e in r['engines']})
    return {'status':'PARTIAL_FAILURE' if failed else 'SUCCESS' if rows else 'NO_RESULTS','results':rows,
            'requested_engines':engines,'responding_engines':responding,'unresponsive_engines':failed,
            'upstream_requests':None}


class Discovery:
    def __init__(self,admission,ledger,store,*,backend,endpoint=None,engines=('duckduckgo',),transport=None,cap=5,traffic=None):
        if backend not in ('duckduckgo','searxng'):raise ValueError('Explicit supported backend required')
        if type(cap) is not int or not 1<=cap<=5:raise ValueError('Candidate cap exceeded')
        if backend=='searxng' and endpoint!='http://127.0.0.1:8080':raise ValueError('Only reviewed local SearXNG endpoint permitted')
        self.admission=admission;self.ledger=ledger;self.store=store;self.backend=backend;self.endpoint=endpoint;self.engines=list(engines);self.transport=transport;self.cap=cap
        self.traffic=traffic if traffic is not None else Traffic()
    def _block_state(self,reason=None):
        """Persistent stop on observed access/rate failures; no automatic probing.

        Clearing this file is not a supported retry mechanism. A supervisor must
        first verify the provider's backoff/access prerequisites and record a new
        authorised diagnostic protocol. All original snapshots stay immutable.
        """
        key=json.dumps([self.backend,self.endpoint,self.engines],sort_keys=True)
        path=self.store.path.parent/'discovery_blocks.jsonl'
        with path.open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            try:
                f.seek(0);events=[json.loads(x) for x in f if x.strip()]
                blocked=any(e['key']==key for e in events)
                if reason and not blocked:
                    f.seek(0,2);f.write(json.dumps({'key':key,'reason':reason,'time':datetime.now(timezone.utc).isoformat()})+'\n');f.flush()
                    return True
                return blocked
            finally:fcntl.flock(f,fcntl.LOCK_UN)
    def search(self,sample_id,query,*,replay=None):
        self.admission.check(sample_id)
        if type(query) is not str or not query.strip() or len(query)>2000:raise ValueError('Invalid bounded query')
        settings={'backend':self.backend,'endpoint':self.endpoint,'engines':self.engines,'cap':self.cap,'language':'en','safesearch':1}
        if replay:
            doc=self.store.get(replay)
            if doc['settings']!=settings or doc['payload']['query']!=query:raise RetrievalFailure('Replay configuration/query mismatch')
            return dict(doc['payload'],snapshot_id=replay,mode='REPLAY')
        if self._block_state():
            result={'query':query,'backend':self.backend,'status':'SERVICE_BLOCKED','results':[],
                    'detail':'Prior access/rate failure requires backoff/access review','logical_queries':0,
                    'retries':0,'upstream_requests':0,'latency_s':0}
            ident=self.store.put('discovery',result,settings)
            return dict(result,snapshot_id=ident,mode='NO_DISPATCH')
        scopes=['search:all']+['engine:'+e for e in (self.engines if self.backend=='searxng' else ['duckduckgo'])]
        try:ticket=self.traffic.start('search',scopes,deadline=self.ledger.deadline)
        except TrafficBlocked as exc:
            result={'query':query,'backend':self.backend,'status':'SERVICE_BLOCKED','results':[],
                    'detail':exc.reason,'retry_at':exc.retry_at,'logical_queries':0,'retries':0,'upstream_requests':0,'latency_s':0}
            return dict(result,snapshot_id=self.store.put('discovery',result,settings),mode='NO_DISPATCH')
        try:
            self.admission.check(sample_id)
            rid=self.ledger.reserve(self.backend,0)
        except BaseException:
            self.traffic.finish(ticket)
            raise
        started=time.monotonic();raw=None;status=None;retry_after=None
        try:
            if self.transport:raw,status=self.transport(query,settings)
            elif self.backend=='searxng':
                import httpx
                with httpx.Client(timeout=25,follow_redirects=False,trust_env=False) as c:
                    r=c.get(self.endpoint+'/search',params={'q':query,'format':'json','engines':','.join(self.engines),'language':'en','safesearch':1})
                    status=r.status_code
                    retry_after=r.headers.get('Retry-After')
                    try:raw=r.json()
                    except ValueError:raw=None
            else:
                from ddgs import DDGS
                # Pin actual backend: never DDGS auto metasearch in the DDG condition.
                raw=list(DDGS(timeout=25).text(query,backend='duckduckgo',region='us-en',safesearch='moderate',max_results=self.cap));status=200
            result=searx_result(raw,status,self.engines,self.cap) if self.backend=='searxng' else {'status':'SUCCESS' if raw else 'NO_RESULTS','results':deduplicate(raw)[:self.cap],'upstream_requests':None,'requested_engines':['duckduckgo'],'responding_engines':[]}
        except Exception as exc:result={'status':'TOOL_ERROR','results':[],'error_type':type(exc).__name__,'upstream_requests':None}
        result['api_status']=status
        failures=result.get('unresponsive_engines',[])
        blocked=[];access=False
        if status in (202,403,429):blocked=scopes;access=status!=429
        elif result.get('error_type'):blocked=scopes
        else:
            for failure in failures:
                if isinstance(failure,(list,tuple)) and len(failure)>=2:
                    reason=str(failure[1]).lower()
                    if any(s in reason for s in ('too many requests','captcha','suspended','access denied')):
                        scope='engine:'+str(failure[0])
                        blocked.append(scope if scope in scopes else 'search:all')
                        access=access or any(s in reason for s in ('captcha','access denied'))
        self.traffic.finish(ticket,blocked_scopes=blocked,access=access,retry_after=retry_after,
                            reason='Observed discovery access/rate/tool failure; supervised review required')
        # These installed local/free discovery paths incur no metered API charge.
        self.ledger.settle(rid,0)
        failure_text=json.dumps(result.get('unresponsive_engines',[])).lower()
        if any(s in failure_text for s in ('too many requests','captcha','suspended','access denied')):
            self._block_state('Observed upstream rate/access failure; see immutable discovery snapshot')
        result.update(query=query,backend=self.backend,latency_s=time.monotonic()-started,logical_queries=1,retries=0,request_id=rid)
        snapshot=self.store.put('discovery',dict(result,raw=raw),settings)
        return dict(result,snapshot_id=snapshot,mode='LIVE')


def extraction_result(raw,status,original_url,max_chars=20000):
    base={'url':original_url,'final_url':None,'publication_date':None,'date_origin':'unknown','api_status':status,'target_status':None,'text':''}
    if status!=200 or type(raw) is not dict or raw.get('success') is not True:return dict(base,status='TOOL_ERROR')
    data=raw.get('data',{});meta=data.get('metadata',{}) if type(data) is dict else {}
    if type(meta) is not dict:return dict(base,status='TOOL_ERROR')
    final=meta.get('url') or meta.get('sourceURL') or original_url
    try:final=public_url(final,resolve=False)
    except ValueError:return dict(base,status='UNSAFE_REDIRECT')
    base.update(final_url=final,target_status=meta.get('statusCode'))
    if type(base['target_status']) is not int or base['target_status']<200 or base['target_status']>=300:return dict(base,status='TARGET_ERROR')
    body=data.get('markdown')
    if type(body) is not str or not body.strip():return dict(base,status='EMPTY_OR_TRUNCATED')
    low=body.casefold()
    if any(marker in low[:1500] for marker in ('access denied','verify you are human','sign in to continue','captcha','log in to continue','before you continue to','consent.youtube.com','consent.google.com')):return dict(base,status='BLOCKED_OR_LOGIN')
    truncated=len(body)>max_chars or meta.get('truncated') is True
    base['text']=body[:max_chars]
    date=meta.get('publishedTime') or meta.get('article:published_time')
    if type(date) is str:base.update(publication_date=date,date_origin='page_metadata_unverified')
    return dict(base,status='EMPTY_OR_TRUNCATED' if truncated else 'USABLE',truncated=truncated)


class Extraction:
    def __init__(self,admission,ledger,store,*,endpoint,transport=None,resolver=public_url,traffic=None):
        if endpoint!='http://127.0.0.1:3002/v2/scrape':raise ValueError('Only reviewed self-hosted Firecrawl endpoint supported in this run')
        self.admission=admission;self.ledger=ledger;self.store=store;self.endpoint=endpoint;self.transport=transport;self.resolver=resolver
        self.traffic=traffic if traffic is not None else Traffic()
    def scrape(self,sample_id,url,*,replay=None):
        self.admission.check(sample_id)
        settings={'endpoint':self.endpoint,'formats':['markdown'],'onlyMainContent':True,'maxAge':0,'timeout':20000}
        if replay:
            doc=self.store.get(replay)
            if doc['settings']!=settings or doc['payload']['url']!=url:raise RetrievalFailure('Replay URL/settings mismatch')
            return dict(doc['payload'],snapshot_id=replay,mode='REPLAY')
        url=self.resolver(url)
        scopes=['extract:all','host:'+urlsplit(url).hostname.lower()]
        try:ticket=self.traffic.start('extract',scopes,deadline=self.ledger.deadline)
        except TrafficBlocked as exc:
            result={'url':url,'status':'SERVICE_BLOCKED','text':'','detail':exc.reason,'retry_at':exc.retry_at,
                    'retries':0,'logical_queries':0,'upstream_requests':0,'latency_s':0}
            return dict(result,snapshot_id=self.store.put('extraction',result,settings),mode='NO_DISPATCH')
        try:
            self.admission.check(sample_id)
            rid=self.ledger.reserve('firecrawl_self_hosted',0)
        except BaseException:
            self.traffic.finish(ticket)
            raise
        started=time.monotonic();raw=None;status=None;retry_after=None
        try:
            request=dict(settings);request.pop('endpoint');request['url']=url
            if self.transport:raw,status=self.transport(self.endpoint,request)
            else:
                import httpx
                with httpx.Client(timeout=30,follow_redirects=False,trust_env=False) as c:
                    r=c.post(self.endpoint,json=request);status=r.status_code;retry_after=r.headers.get('Retry-After');raw=r.json()
            result=extraction_result(raw,status,url)
        except Exception as exc:result={'url':url,'status':'TOOL_ERROR','text':'','error_type':type(exc).__name__}
        blocked=[];access=False
        if status in (202,403,429) or result.get('error_type'):
            blocked=scopes;access=status in (202,403)
        elif result.get('target_status') in (403,429) or result.get('status')=='BLOCKED_OR_LOGIN':
            blocked=[scopes[1]];access=result.get('target_status')!=429
        self.traffic.finish(ticket,blocked_scopes=blocked,access=access,retry_after=retry_after,
                            reason='Observed extraction access/rate/tool failure; supervised review required')
        self.ledger.settle(rid,0)
        result.update(latency_s=time.monotonic()-started,retries=0,request_id=rid)
        snapshot=self.store.put('extraction',dict(result,raw=raw),settings)
        return dict(result,snapshot_id=snapshot,mode='LIVE')


def passages(result,claim_id):
    if result['status']!='USABLE':return []
    pieces=[p.strip() for p in result['text'].split('\n\n') if p.strip()]
    return [{'id':result['snapshot_id']+':'+str(i),'claim_id':claim_id,'text':p,'kind':'passage','url':result['final_url'],'snapshot_id':result['snapshot_id'],'publication_date':result['publication_date'],'date_origin':result['date_origin']} for i,p in enumerate(pieces)]
