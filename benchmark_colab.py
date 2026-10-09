"""Run real, sequential TXT→IMG measurements against an existing local Studio."""
import argparse
import getpass
import json
import time
from pathlib import Path
import requests

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:7860')
    parser.add_argument('--count',type=int,default=20)
    parser.add_argument('--model',default='wai-anima')
    parser.add_argument('--output',default='benchmark-results.json')
    args=parser.parse_args()
    if not 1<=args.count<=100: parser.error('count deve estar entre 1 e 100')
    client=requests.Session(); login=client.post(args.url+'/api/login',json={'password':getpass.getpass('Senha do Studio: ')},timeout=15)
    login.raise_for_status(); client.headers['X-CSRF-Token']=login.json()['csrf']
    results={'initial':client.get(args.url+'/api/diagnostics',timeout=15).json(),'jobs':[]}
    for index in range(args.count):
        size=(512,768,1024)[index%3]; started=time.monotonic()
        response=client.post(args.url+'/api/jobs',json={'prompt':'portrait, blue hair, detailed background','seed':1000+index,'model':args.model,'width':size,'height':size,'steps':24,'guidance':5},timeout=30)
        response.raise_for_status(); job=response.json(); identifier=job['id']
        deadline=time.monotonic()+1800
        while job['status'] in {'queued','running'}:
            if time.monotonic()>deadline:
                client.post(args.url+f'/api/jobs/{identifier}/cancel',timeout=15).raise_for_status()
                raise TimeoutError('Benchmark cancelou job após 1800 segundos')
            time.sleep(2); response=client.get(args.url+f'/api/jobs/{identifier}',timeout=15); response.raise_for_status(); job=response.json()
        results['jobs'].append({'index':index,'size':size,'wall_seconds':round(time.monotonic()-started,3),'job':job})
        results['final']=client.get(args.url+'/api/diagnostics',timeout=15).json()
        Path(args.output).write_text(json.dumps(results,ensure_ascii=False,indent=2))
        print(index+1,job['status'],results['jobs'][-1]['wall_seconds'])
        if job['status']!='completed': raise RuntimeError(job.get('error') or job['status'])
    print('Resultados:',args.output)
if __name__=='__main__': main()
