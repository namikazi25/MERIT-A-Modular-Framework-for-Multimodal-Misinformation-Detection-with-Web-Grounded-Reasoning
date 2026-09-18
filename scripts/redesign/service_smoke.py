"""Fixed public-document service probes; no dataset access."""
import argparse,json
from pathlib import Path
from scripts.redesign.ledger import Ledger
from scripts.redesign.retrieval import Discovery,Extraction,SnapshotStore

class _PublicProbe:
    def check(self,ident):
        if ident!='public-doc-probe':raise ValueError('Only fixed public probe allowed')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();run=Path(a.run_dir)
    out=run/'services/live_probes.json'
    if out.exists():raise SystemExit('Probe already recorded; do not duplicate')
    state=json.loads((run/'state.json').read_text())
    ledger=Ledger(run/'spend.jsonl',15,applicable_remaining=15,deadline=state['deadline_utc'])
    store=SnapshotStore(run/'snapshots_web');guard=_PublicProbe();results={}
    for backend in ('duckduckgo','searxng'):
        d=Discovery(guard,ledger,store,backend=backend,endpoint='http://127.0.0.1:8080' if backend=='searxng' else None,engines=('duckduckgo','bing'))
        r=d.search('public-doc-probe','IANA example domains')
        results[backend]={k:v for k,v in r.items() if k not in ('results','raw')};results[backend]['n_results']=len(r['results'])
    e=Extraction(guard,ledger,store,endpoint='http://127.0.0.1:3002/v2/scrape')
    r=e.scrape('public-doc-probe','https://example.com/')
    results['firecrawl']={k:v for k,v in r.items() if k not in ('text','raw')};results['firecrawl']['text_length']=len(r.get('text',''))
    out.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
if __name__=='__main__':main()
