"""Fixed synthetic live probes; no benchmark loader and no caller-supplied sample content."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from PIL import Image
from scripts.redesign.clients import GLMClient,JevClient
from scripts.redesign.ledger import Ledger
from scripts.redesign_registry import claim_sha256

HEAD='The synthetic square is red.'

class _SyntheticAdmission:
    def __init__(self,image):self.image=image
    def check(self,ident):
        if ident!='fixed-synthetic-probe':raise ValueError('Fixed synthetic identity required')
        return {'claim_sha256':claim_sha256(HEAD),'image_byte_sha256':hashlib.sha256(self.image).hexdigest()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--service',choices=['glm','jev'],required=True)
    a=p.parse_args();run=Path(a.run_dir);state=json.loads((run/'state.json').read_text())
    output=run/f'{a.service}_synthetic_live.json'
    if output.exists():raise SystemExit('Existing probe result; refusing duplicate live call')
    config=json.loads(Path('config/redesign_runtime.json').read_text())
    if not config.get('live_enabled'):raise SystemExit('Live requests disabled in runtime configuration')
    ledger=Ledger(run/'spend.jsonl',15,applicable_remaining=15,deadline=state['deadline_utc'])
    if ledger.summary()['pending_ids']:raise SystemExit('Outstanding requests need reconciliation before retry')
    # Secrets loaded only by this intended client process, never returned or logged.
    from dotenv import load_dotenv
    load_dotenv('.env',override=False)
    b=io.BytesIO();Image.new('RGB',(64,64),'red').save(b,format='PNG');image=b.getvalue()
    admission=_SyntheticAdmission(image)
    payload={'headline':HEAD,'claims':[{'id':'c1','text':HEAD,'source_span':HEAD}],
             'evidence':[{'id':'e1','claim_id':'c1','text':'The synthetic drawing consists of one solid red square.', 'kind':'synthetic','snapshot_id':'synthetic-v1'}]}
    if a.service=='glm':
        result=GLMClient(json.loads(Path('config/llm.json').read_text()),admission,ledger,max_output=512).complete('fixed-synthetic-probe',payload,'Inspect the image. Return only JSON with color (a color name) and shape (a shape name).',image)
        result['probe_passed']=result['parsed'].get('color','').lower()=='red' and result['parsed'].get('shape','').lower()=='square'
    else:
        result=JevClient(admission,ledger).ask('fixed-synthetic-probe',payload,{'supports':{'type':'noul','instructions':'Does `evidence[0].text` directly support `claims[0].text`?'}})
        result['probe_passed']=result.get('status')=='OK' and 'supports' in result['answers']
    result['mode']='LIVE_SYNTHETIC';result['benchmark_examples_used']=0
    with output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
