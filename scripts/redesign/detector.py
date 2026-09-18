"""Pinned local AI-generation detector; no manipulation or provenance claim."""
import hashlib
import io
import json
from pathlib import Path
import sys
import time

MODEL_REVISION='5d647683672ffd0080441f4e8b4345934c44cc61'
EXPECTED={
 'config.json':'44fab7b53b09d27a073fc7c911cf1fa0f51c5b204525ada5c62956f335cd8d33',
 'preprocessor_config.json':'ff4f6287fdbe3e1e291b826aa8b7d8e7d7789a1b8a3cc62c817391b5d9e6aa4a',
 'model.safetensors':'4f5d0d4e8e6475faccca9385a2869648bb1f18d3f64c0239ad2bca909d3adc50'}


class GenerationDetector:
    def __init__(self,admission,model_dir='models/redesign-generation-v1',site_dir='.runtime/redesign-detector/site'):
        self.admission=admission;self.path=Path(model_dir);self.results={}
        for name,digest in EXPECTED.items():
            if hashlib.sha256((self.path/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Detector checkpoint/configuration hash mismatch')
        sys.path.insert(0,str(Path(site_dir).resolve()))
        from transformers import ViTImageProcessor,ViTForImageClassification
        self.processor=ViTImageProcessor.from_pretrained(self.path,local_files_only=True)
        self.model=ViTForImageClassification.from_pretrained(self.path,local_files_only=True,use_safetensors=True).eval().to('cpu')
        if self.model.config.id2label!={0:'REAL',1:'FAKE'}:raise ValueError('Detector label semantics changed')

    def __call__(self,sample):
        row=self.admission.check(sample['sample_id'])
        digest=hashlib.sha256(sample['image_bytes']).hexdigest()
        if digest!=row['image_byte_sha256']:raise ValueError('Detector image identity mismatch')
        if digest in self.results:return dict(self.results[digest]['signal'])
        from PIL import Image
        import torch
        started=time.monotonic()
        with Image.open(io.BytesIO(sample['image_bytes'])) as source:inputs=self.processor(images=source.convert('RGB'),return_tensors='pt')
        with torch.inference_mode():probs=self.model(**inputs).logits.softmax(dim=-1)[0].tolist()
        positive=probs[1]>=.5
        signal={'ai_generated':positive,'confidence':probs[1] if positive else probs[0],
                'explanation':f'Dedicated ViT generation classifier: computed P(AI-generated)={probs[1]:.6f}, fixed threshold 0.5. Not a manipulation detector, image-origin match, or factual-veracity conclusion. Calibration and modern-generator transfer are unvalidated.',
                'anomalies':[]}
        self.results[digest]={'signal':signal,'probabilities':{'REAL':probs[0],'AI_GENERATED':probs[1]},
            'latency_s':time.monotonic()-started,'revision':MODEL_REVISION,'image_sha256':digest,'device':'cpu'}
        return dict(signal)


def main():
    import argparse
    from PIL import Image
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args()
    out=Path(a.run_dir)/'detector_synthetic_live.json'
    if out.exists():raise SystemExit('Existing detector smoke result preserved')
    b=io.BytesIO();Image.new('RGB',(64,64),'red').save(b,format='PNG');data=b.getvalue()
    class Synthetic:
        def check(self,sid):
            if sid!='synthetic-red-square':raise ValueError('Synthetic fixture only')
            return {'image_byte_sha256':hashlib.sha256(data).hexdigest()}
    detector=GenerationDetector(Synthetic());detector({'sample_id':'synthetic-red-square','image_bytes':data})
    result={'mode':'REAL_LOCAL_CHECKPOINT_SYNTHETIC_SMOKE','results':detector.results,
            'limitation':'Confirms local loading and inference only, not detector accuracy or calibration.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
