"""Compare original local input bytes with a pinned public provider Git tree."""
from semg.paths import RUN_ROOT, SOURCE_ROOT
import hashlib
import json
import os
from pathlib import Path
ROOT=RUN_ROOT
raw=Path(os.environ['SEMG_RAW'])
tree=json.loads((ROOT/'private/diagnostic_v2/provider_tree.json').read_text());assert not tree['truncated']
blobs={Path(z['path']).name:z['sha'] for z in tree['tree'] if z['type']=='blob' and z['path'].endswith('.csv')}
rows=[]
for path in sorted(raw.glob('*.csv')):
    data=path.read_bytes();actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    rows.append(dict(file=path.name,local_git_blob=actual,provider_git_blob=blobs.get(path.name),matched=actual==blobs.get(path.name)))
(ROOT/'private/diagnostic_v2/provider_comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
summary=dict(provider='TestUser-alt-dev/sEMG-Hand-Gesture-Elbow-Angle-Dataset',commit='cb64ded658b83ae97e6388dd373efad0e97bc47c',
    local_files=len(rows),provider_csv_files=len(blobs),matching_git_blobs=sum(z['matched'] for z in rows),all_match=all(z['matched'] for z in rows),
    method='Git SHA-1 over blob header and exact file bytes; no signal uploaded')
(ROOT/'results/diagnostic_v2/provider_verification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
