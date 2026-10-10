"""A one-page local GUI for the frame selector: pick videos, run, review, delete rejected frames."""
import csv
import glob
import os
import threading
import webbrowser

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from .pipeline import run

VIDEO_EXT = ('.dav', '.mp4', '.mkv', '.avi', '.mov')
app = FastAPI(title='Frame selector')
state = {'running': False, 'log': [], 'error': None, 'out': None}


class RunRequest(BaseModel):
    videos: list[str]
    out: str
    model: str = ''
    budget: int = 100


class DeleteRequest(BaseModel):
    out: str
    images: list[str]


def _dump_for(video, out):
    """An existing detection dump for this video, next to it or in the output folder."""
    stem = os.path.splitext(os.path.basename(video))[0]
    for d in (os.path.dirname(video), out):
        hits = glob.glob(os.path.join(glob.escape(d), f"detections_{glob.escape(stem)}*.npz"))
        if hits:
            return hits[0]
    return None


@app.get('/api/scan')
def scan(dir: str, out: str = ''):
    if not os.path.isdir(dir):
        raise HTTPException(400, f"not a folder: {dir}")
    vids = sorted(f for f in os.listdir(dir) if f.lower().endswith(VIDEO_EXT))
    return [dict(path=os.path.join(dir, f), name=f, size_gb=round(os.path.getsize(os.path.join(dir, f)) / 1e9, 2),
                 dump=_dump_for(os.path.join(dir, f), out or dir)) for f in vids]


@app.post('/api/run')
def start(req: RunRequest):
    if state['running']:
        raise HTTPException(409, 'a run is already in progress')
    dumps = [_dump_for(v, req.out) for v in req.videos]
    if not req.model and not all(dumps):
        raise HTTPException(400, 'a model is needed: some videos have no detection dump yet')
    state.update(running=True, log=[], error=None, out=req.out)

    def work():
        try:
            run(req.videos, req.out, req.model or None, dumps, req.budget, progress=lambda m: state['log'].append(m))
        except Exception as e:                       # noqa: BLE001 - shown to the user
            state['error'] = f"{type(e).__name__}: {e}"
        finally:
            state['running'] = False
    threading.Thread(target=work, daemon=True).start()
    return {'ok': True}


@app.get('/api/status')
def status():
    return {k: state[k] for k in ('running', 'log', 'error', 'out')}


@app.get('/api/results')
def results(out: str):
    path = os.path.join(out, 'manifest.csv')
    if not os.path.exists(path):
        return []
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


@app.get('/file')
def file(out: str, path: str):
    full = os.path.abspath(os.path.join(out, path))
    if not full.startswith(os.path.abspath(out)) or not os.path.isfile(full):
        raise HTTPException(404)
    return FileResponse(full)


@app.post('/api/delete')
def delete(req: DeleteRequest):
    path = os.path.join(req.out, 'manifest.csv')
    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    keep = [r for r in rows if r['image'] not in set(req.images)]
    for r in rows:
        if r['image'] in req.images:
            for p in (r['image'], r['label'], 'review/' + os.path.basename(r['image'])):
                try:
                    os.remove(os.path.join(req.out, p))
                except OSError:
                    pass
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys() if rows else ['image'])
        w.writeheader()
        w.writerows(keep)
    return {'deleted': len(rows) - len(keep), 'remaining': len(keep)}


PAGE = r'''<!doctype html><meta charset="utf-8"><title>Frame selector</title>
<style>
body{font:14px system-ui;margin:0;background:#111;color:#ddd} main{max-width:1400px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:0 0 12px} h2{font-size:16px;margin:20px 0 8px} label{display:block;margin:6px 0}
input[type=text],input[type=number]{width:100%;max-width:640px;padding:6px;background:#222;color:#eee;border:1px solid #444;border-radius:4px}
button{font:inherit;padding:7px 14px;margin:4px 4px 4px 0;border:0;border-radius:4px;background:#2d6cdf;color:#fff;cursor:pointer} button:disabled{background:#444}
button.danger{background:#b3261e} .row{display:flex;gap:16px;flex-wrap:wrap} .col{flex:1;min-width:320px}
#videos label{margin:2px 0} pre{background:#000;padding:8px;height:160px;overflow:auto;font-size:12px;white-space:pre-wrap}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:10px}
figure{margin:0;background:#1c1c1c;padding:6px;border-radius:6px} figure img{width:100%;display:block} figcaption{padding:6px 2px;line-height:1.4;font-size:12px}
.legend span{display:inline-block;margin-right:14px;padding-left:16px;border-left:12px solid} .muted{color:#888}
</style>
<main>
<h1>Frame selector</h1>
<div class="row"><div class="col">
<label>Video folder <input type="text" id="dir" placeholder="C:\path\to\videos"></label>
<button onclick="scan()">Scan folder</button>
<div id="videos" class="muted">No folder scanned yet.</div>
</div><div class="col">
<label>Model (.engine or .pt) <input type="text" id="model" placeholder="leave empty to reuse existing detection dumps only"></label>
<label>Output folder <input type="text" id="out" value="mined"></label>
<label>Frames per 2 hours of video <input type="number" id="budget" value="100" min="10" max="2000"></label>
<button id="runbtn" onclick="startRun()">Run</button> <button onclick="loadResults()">Load results from output folder</button>
<pre id="log"></pre>
</div></div>
<h2>Results <span id="count" class="muted"></span></h2>
<p class="legend">Green = pseudo-label to correct. Flagged box: <span style="border-color:#f00">missed</span><span style="border-color:#ffa500">false positive</span><span style="border-color:#f0f">wrong class</span><span style="border-color:#dcdc00">low confidence</span></p>
<p><button class="danger" onclick="deleteRejected()">Delete rejected frames</button> <span class="muted">Ticked frames are removed from images/, labels/ and manifest.csv. What remains is ready to label.</span></p>
<div id="results"></div>
</main>
<script>
const $=id=>document.getElementById(id); let timer=null;
const TXT={fn:'missed vehicle',fp:'false positive / hard negative','class':'wrong class',borderline:'low confidence'};
async function scan(){const r=await fetch(`/api/scan?dir=${encodeURIComponent($('dir').value)}&out=${encodeURIComponent($('out').value)}`);
 if(!r.ok){alert((await r.json()).detail);return} const v=await r.json();
 $('videos').innerHTML=v.length?v.map(x=>`<label><input type="checkbox" class="vid" value="${x.path}" checked> ${x.name} <span class="muted">${x.size_gb} GB${x.dump?' · detection dump found, model not needed':''}</span></label>`).join(''):'No videos in that folder.';}
async function startRun(){const videos=[...document.querySelectorAll('.vid:checked')].map(e=>e.value); if(!videos.length){alert('Select at least one video');return}
 const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({videos,out:$('out').value,model:$('model').value,budget:+$('budget').value})});
 if(!r.ok){alert((await r.json()).detail);return} $('runbtn').disabled=true; timer=setInterval(poll,1500); poll();}
async function poll(){const s=await (await fetch('/api/status')).json(); $('log').textContent=s.log.join('\n')+(s.error?'\nERROR '+s.error:''); $('log').scrollTop=1e9;
 if(!s.running){clearInterval(timer);$('runbtn').disabled=false;if(!s.error)loadResults();}}
async function loadResults(){const out=$('out').value; const rows=await (await fetch(`/api/results?out=${encodeURIComponent(out)}`)).json();
 $('count').textContent=rows.length?`${rows.length} frames`:'nothing in that folder yet'; const by={};
 rows.forEach(r=>{(by[r.reasons.split('|')[0]]=by[r.reasons.split('|')[0]]||[]).push(r)});
 $('results').innerHTML=Object.keys(TXT).filter(k=>by[k]).map(k=>`<h2>${TXT[k]} (${by[k].length})</h2><div class="grid">`+by[k].map(r=>
  `<figure><img loading="lazy" src="/file?out=${encodeURIComponent(out)}&path=${encodeURIComponent('review/'+r.image.split('/').pop())}" onclick="window.open('/file?out=${encodeURIComponent(out)}&path=${encodeURIComponent(r.image)}')">
   <figcaption><label><input type="checkbox" class="rej" value="${r.image}"> reject</label><b>${r.video}</b> ${r.clock||r.time_s+'s'} · frame ${r.frame}<br>${r.notes}</figcaption></figure>`).join('')+'</div>').join('');}
async function deleteRejected(){const images=[...document.querySelectorAll('.rej:checked')].map(e=>e.value); if(!images.length){alert('Nothing ticked');return}
 if(!confirm(`Delete ${images.length} frames from the output folder?`))return;
 const r=await (await fetch('/api/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({out:$('out').value,images})})).json();
 alert(`Deleted ${r.deleted}; ${r.remaining} frames remain`); loadResults();}
</script>'''


@app.get('/', response_class=HTMLResponse)
def index():
    return PAGE


def serve(port=8090, open_browser=True):
    import uvicorn
    url = f"http://127.0.0.1:{port}"
    print(f"Frame selector GUI: {url}")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host='127.0.0.1', port=port, log_level='warning')
