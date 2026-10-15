"""A one-page local GUI for the frame selector: pick videos, run, review, delete rejected frames."""
import contextlib
import csv
import glob
import io
import os
import re
import subprocess
import sys
import threading
import webbrowser

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from .detect import configured_model, dump_is_usable
from .export import read_manifest
from .pipeline import run

VIDEO_EXT = ('.dav', '.mp4', '.mkv', '.avi', '.mov')
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO_DIR, 'frames_for_annotation')
app = FastAPI(title='Frame selector')
state = {'running': False, 'log': [], 'error': None, 'out': None}
_dialog_lock = threading.Lock()


class RunRequest(BaseModel):
    videos: list[str]
    out: str
    budget: int = 100


class DeleteRequest(BaseModel):
    out: str
    images: list[str]


def _video_dir():
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            from config import PATHS
            return PATHS.get('video_dir') or ''
        except Exception:                            # noqa: BLE001 - config is per machine
            return ''


def _dump_for(video, out):
    """An existing detection dump for this video, next to it or in the output folder."""
    stem = os.path.splitext(os.path.basename(video))[0]
    for d in (os.path.dirname(video), out):
        exact = os.path.join(d, f"detections_{stem}.npz")
        hits = [exact] if os.path.exists(exact) else sorted(glob.glob(os.path.join(glob.escape(d), f"detections_{glob.escape(stem)}*.npz")))
        for h in hits:
            if dump_is_usable(h):
                return h
    return None


@app.get('/api/info')
def info():
    m = configured_model()
    vd = _video_dir()
    return {'model': m, 'model_exists': os.path.exists(m), 'default_out': DEFAULT_OUT,
            'video_dir': vd if os.path.isdir(vd) else '', 'running': state['running']}


_PICK_SCRIPT = """
import sys, tkinter
from tkinter import filedialog
root = tkinter.Tk(); root.withdraw(); root.attributes('-topmost', True)
start = sys.argv[1] if len(sys.argv) > 1 else None
print(filedialog.askdirectory(initialdir=start or None, title='Choose a folder') or '')
"""


@app.get('/api/pick_folder')
def pick_folder(start: str = ''):
    """Native folder dialog on the machine running the tool (the browser is on the same machine).
    Runs in its own process so a misbehaving dialog can never freeze the server."""
    if not _dialog_lock.acquire(blocking=False):
        raise HTTPException(409, 'a dialog is already open')
    try:
        proc = subprocess.run([sys.executable, '-c', _PICK_SCRIPT, start if os.path.isdir(start) else ''],
                              capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            return {'path': '', 'error': proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else 'dialog failed'}
        return {'path': proc.stdout.strip()}
    except subprocess.TimeoutExpired:
        return {'path': ''}
    except Exception as e:                           # noqa: BLE001 - fall back to the in-page browser
        return {'path': '', 'error': str(e)}
    finally:
        _dialog_lock.release()


@app.get('/api/browse')
def browse(path: str = ''):
    """Fallback folder browser: subfolders of `path` (drives / root when empty)."""
    if not path:
        if os.name == 'nt':
            import string
            roots = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
        else:
            roots = ['/']
        return {'path': '', 'parent': None, 'folders': roots}
    if not os.path.isdir(path):
        raise HTTPException(400, f"not a folder: {path}")
    try:
        subs = sorted(os.path.join(path, d) for d in os.listdir(path)
                      if os.path.isdir(os.path.join(path, d)) and not d.startswith('.'))
    except PermissionError:
        subs = []
    parent = os.path.dirname(path.rstrip('\\/'))
    return {'path': path, 'parent': parent if parent != path else '', 'folders': subs}


@app.get('/api/scan')
def scan(dir: str, out: str = ''):
    if not os.path.isdir(dir):
        raise HTTPException(400, f"not a folder: {dir}")
    vids = sorted(f for f in os.listdir(dir) if f.lower().endswith(VIDEO_EXT))
    done = {}
    for r in (read_manifest(out) if out else []):
        done[r['video']] = done.get(r['video'], 0) + 1
    return [dict(path=os.path.join(dir, f), name=f, size_gb=round(os.path.getsize(os.path.join(dir, f)) / 1e9, 2),
                 dump=_dump_for(os.path.join(dir, f), out or dir), done=done.get(f, 0)) for f in vids]


@app.post('/api/run')
def start(req: RunRequest):
    if state['running']:
        raise HTTPException(409, 'a run is already in progress')
    dumps = [_dump_for(v, req.out) for v in req.videos]
    if not all(dumps) and not os.path.exists(configured_model()):
        raise HTTPException(400, f"model not found: {configured_model()} (set MODEL_PATH / PC in config.py)")
    state.update(running=True, log=[], error=None, out=req.out)

    def work():
        try:
            run(req.videos, req.out, dumps, req.budget, progress=lambda m: state['log'].append(m))
        except Exception as e:                       # noqa: BLE001 - shown to the user
            state['error'] = f"{type(e).__name__}: {e}"
        finally:
            state['running'] = False
    threading.Thread(target=work, daemon=True).start()
    return {'ok': True}


@app.get('/api/status')
def status():
    log = state['log']
    percent, stage = None, (log[-1] if log else '')
    for line in reversed(log):
        m = re.search(r'(\d+)/(\d+) frames', line)
        if m:
            percent = round(100 * int(m.group(1)) / max(int(m.group(2)), 1))
            break
    return {'running': state['running'], 'log': log, 'error': state['error'], 'out': state['out'],
            'stage': stage, 'percent': percent}


@app.get('/api/results')
def results(out: str):
    return read_manifest(out)


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
                with contextlib.suppress(OSError):
                    os.remove(os.path.join(req.out, p))
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys() if rows else ['image'])
        w.writeheader()
        w.writerows(keep)
    return {'deleted': len(rows) - len(keep), 'remaining': len(keep)}


PAGE = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Frame selector</title>
<style>
:root{--bg:#0f1115;--card:#171a21;--line:#262b36;--text:#e6e8ee;--muted:#8b93a7;--accent:#3b82f6;--ok:#22c55e;--bad:#ef4444;
 --fn:#ef4444;--fp:#f59e0b;--class:#d946ef;--borderline:#eab308}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif}
header{display:flex;align-items:center;gap:14px;padding:14px 24px;border-bottom:1px solid var(--line);background:var(--card)}
header h1{font-size:18px;margin:0;font-weight:600} header .model{margin-left:auto;color:var(--muted);font-size:13px;max-width:60%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
main{max-width:1500px;margin:0 auto;padding:20px 24px}
.steps{display:grid;grid-template-columns:1fr 1fr;gap:16px} @media(max-width:900px){.steps{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px}
.card h2{font-size:15px;margin:0 0 12px;display:flex;align-items:center;gap:10px}
.num{display:inline-flex;width:22px;height:22px;border-radius:50%;background:var(--accent);color:#fff;align-items:center;justify-content:center;font-size:12px}
.field{display:flex;gap:8px;align-items:center;margin:8px 0} .field input[type=text]{flex:1;min-width:0}
input[type=text]{cursor:pointer} input[type=text],input[type=number]{padding:8px 10px;background:#0c0e12;color:var(--text);border:1px solid var(--line);border-radius:6px;font:inherit}
input[type=number]{width:90px}
button{font:inherit;padding:8px 14px;border:0;border-radius:6px;background:#2a3140;color:var(--text);cursor:pointer} button:hover{filter:brightness(1.15)}
button.primary{background:var(--accent);color:#fff} button.danger{background:var(--bad);color:#fff} button:disabled{opacity:.5;cursor:default}
.videos{max-height:260px;overflow:auto;border:1px solid var(--line);border-radius:6px;margin-top:8px}
.videos label{display:flex;gap:10px;align-items:center;padding:7px 10px;border-bottom:1px solid var(--line);cursor:pointer} .videos label:last-child{border:0}
.videos .meta{margin-left:auto;color:var(--muted);font-size:12px;white-space:nowrap} .muted{color:var(--muted)} .empty{padding:14px;color:var(--muted)}
.bar{height:8px;background:#0c0e12;border-radius:4px;overflow:hidden;margin:10px 0 6px} .bar i{display:block;height:100%;width:0;background:var(--accent);transition:width .4s}
.bar.busy i{width:30%;animation:slide 1.2s infinite linear} @keyframes slide{from{margin-left:-30%}to{margin-left:100%}}
pre{background:#0c0e12;border:1px solid var(--line);border-radius:6px;padding:10px;height:150px;overflow:auto;font-size:12px;white-space:pre-wrap;margin:8px 0 0}
.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:22px 0 12px}
.chip{display:inline-flex;align-items:center;gap:6px;padding:5px 10px;border-radius:999px;border:1px solid var(--line);background:var(--card);cursor:pointer;font-size:13px}
.chip.on{border-color:var(--accent);background:#1e2a44} .chip b{font-weight:600} .dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;position:relative}
figure.rejected{opacity:.45} figure img{width:100%;display:block;cursor:zoom-in;aspect-ratio:16/9;object-fit:cover}
figcaption{padding:8px 10px;font-size:12px} figcaption .tags{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:4px}
.tag{padding:1px 7px;border-radius:999px;font-size:11px;color:#000;font-weight:600} .when{color:var(--muted)}
figure .rej{position:absolute;top:8px;left:8px;background:rgba(0,0,0,.6);padding:4px 8px;border-radius:6px;font-size:12px;cursor:pointer}
#lightbox{position:fixed;inset:0;background:rgba(0,0,0,.92);display:none;align-items:center;justify-content:center;flex-direction:column;z-index:10;cursor:zoom-out}
#lightbox img{max-width:96vw;max-height:88vh} #lightbox p{color:#ddd;margin:8px}
#browser{position:fixed;inset:0;background:rgba(0,0,0,.6);display:none;align-items:center;justify-content:center;z-index:9}
#browser .box{background:var(--card);border:1px solid var(--line);border-radius:10px;width:min(640px,92vw);max-height:80vh;display:flex;flex-direction:column}
#browser .list{overflow:auto;padding:4px 0} #browser .list div{padding:7px 14px;cursor:pointer} #browser .list div:hover{background:#222838}
#browser .hd,#browser .ft{padding:10px 14px;border-bottom:1px solid var(--line)} #browser .ft{border:0;border-top:1px solid var(--line);display:flex;gap:8px;justify-content:flex-end}
.stats{display:flex;gap:16px;flex-wrap:wrap;margin:6px 0 0;color:var(--muted);font-size:13px}
</style>
<header><h1>Frame selector</h1><span class="muted">picks the frames worth labelling</span><span class="model" id="model" title="MODEL_PATH from config.py"></span></header>
<main>
<div class="steps">
 <section class="card"><h2><span class="num">1</span>Videos</h2>
  <div class="field"><input type="text" id="dir" placeholder="click to choose the folder with videos" readonly onclick="pick('dir')"><button onclick="pick('dir')">Browse…</button><button onclick="scan()">Scan</button></div>
  <div class="videos" id="videos"><div class="empty">Choose a folder and click Scan.</div></div>
  <div class="field" style="margin-top:8px"><button onclick="setAll(true)">Select all</button><button onclick="setAll(false)">Select none</button><button onclick="setDone()" title="tick the videos that already have results, to regenerate them from their cached detections">Select done</button><span class="muted" id="vidcount"></span></div>
 </section>
 <section class="card"><h2><span class="num">2</span>Run</h2>
  <div class="field"><span style="width:150px">Output folder</span><input type="text" id="out" readonly onclick="pick('out')"><button onclick="pick('out')">Browse…</button></div>
  <div class="field"><span style="width:150px">Frames per 2 h video</span><input type="number" id="budget" value="100" min="10" max="2000"></div>
  <div class="field"><button class="primary" id="runbtn" onclick="startRun()">Run</button><button onclick="loadResults()">Load existing results</button><span class="muted" id="stage"></span></div>
  <div class="bar" id="bar"><i></i></div>
  <pre id="log"></pre>
 </section>
</div>
<div class="toolbar">
 <h2 style="margin:0;font-size:15px"><span class="num">3</span> Review <span class="muted" id="count"></span></h2>
 <span id="chips"></span>
 <span style="margin-left:auto"></span>
 <button onclick="rejectVisible(true)">Reject all shown</button><button onclick="rejectVisible(false)">Clear</button>
 <button class="danger" onclick="deleteRejected()">Delete rejected</button>
</div>
<div class="stats" id="stats"></div>
<p class="muted" style="font-size:12px">Each frame is outlined on the vehicle it was chosen for. Click an image to enlarge; tick <b>reject</b> on frames you don't want, then <b>Delete rejected</b>. The label files hold every tracked vehicle, ready to correct.</p>
<div class="grid" id="results"></div>
</main>
<div id="lightbox" onclick="this.style.display='none'"><img id="lbimg"><p id="lbcap"></p></div>
<div id="browser"><div class="box"><div class="hd"><b>Choose a folder</b><div class="muted" id="bpath" style="font-size:12px;word-break:break-all"></div></div><div class="list" id="blist"></div><div class="ft"><button onclick="closeBrowser()">Cancel</button><button class="primary" onclick="chooseBrowser()">Use this folder</button></div></div></div>
<script>
const $=id=>document.getElementById(id); let timer=null, rows=[], filter=null, bTarget=null, bCur='';
const TXT={fn:'miss',fp:'no vehicle',class:'confusion',borderline:'low confidence'};
const COL={fn:'var(--fn)',fp:'var(--fp)',class:'var(--class)',borderline:'var(--borderline)'};
fetch('/api/info').then(r=>r.json()).then(i=>{$('model').textContent=(i.model_exists?'model: ':'MODEL NOT FOUND: ')+i.model;$('model').style.color=i.model_exists?'var(--ok)':'var(--bad)';
 $('out').value=localStorage.fsOut||i.default_out;$('dir').value=localStorage.fsDir||i.video_dir;if($('dir').value)scan();if(i.running){$('runbtn').disabled=true;timer=setInterval(poll,1500)}});
async function pick(target){const r=await (await fetch(`/api/pick_folder?start=${encodeURIComponent($(target).value)}`)).json();
 if(r.error!==undefined){openBrowser(target);return} if(r.path){$(target).value=r.path;if(target==='dir')scan()}}
async function openBrowser(target){bTarget=target;$('browser').style.display='flex';await listDir($(target).value)}
async function listDir(p){const r=await fetch(`/api/browse?path=${encodeURIComponent(p)}`);if(!r.ok){await listDir('');return}const d=await r.json();bCur=d.path;$('bpath').textContent=d.path||'Drives';
 $('blist').innerHTML=(d.parent!==null&&d.path?`<div onclick="listDir(${JSON.stringify(d.parent)})">⬑ up</div>`:'')+d.folders.map(f=>`<div onclick="listDir(${JSON.stringify(f)})">📁 ${f.split(/[\\/]/).filter(Boolean).pop()||f}</div>`).join('')}
function closeBrowser(){$('browser').style.display='none'} function chooseBrowser(){if(bCur){$(bTarget).value=bCur;if(bTarget==='dir')scan()}closeBrowser()}
async function scan(){localStorage.fsDir=$('dir').value;const r=await fetch(`/api/scan?dir=${encodeURIComponent($('dir').value)}&out=${encodeURIComponent($('out').value)}`);
 if(!r.ok){$('videos').innerHTML=`<div class="empty">${(await r.json()).detail}</div>`;return} const v=await r.json();
 $('videos').innerHTML=v.length?v.map(x=>`<label><input type="checkbox" class="vid" value="${x.path}" data-done="${x.done?1:0}" ${x.done?'':'checked'} onchange="countVids()"> <span>${x.name}</span><span class="meta">${x.size_gb} GB${x.dump?' · detections cached':''}${x.done?' · '+x.done+' frames selected already':''}</span></label>`).join(''):'<div class="empty">No videos in this folder.</div>';countVids()}
function setAll(on){document.querySelectorAll('.vid').forEach(e=>e.checked=on);countVids()}
function setDone(){document.querySelectorAll('.vid').forEach(e=>e.checked=e.dataset.done==='1');countVids()}
function countVids(){const n=document.querySelectorAll('.vid:checked').length;$('vidcount').textContent=n?`${n} selected`:''}
async function startRun(){const videos=[...document.querySelectorAll('.vid:checked')].map(e=>e.value);if(!videos.length){alert('Select at least one video');return}
 localStorage.fsOut=$('out').value;const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({videos,out:$('out').value,budget:+$('budget').value})});
 if(!r.ok){alert((await r.json()).detail);return}$('runbtn').disabled=true;$('results').innerHTML='';timer=setInterval(poll,1500);poll()}
async function poll(){const s=await (await fetch('/api/status')).json();$('log').textContent=s.log.join('\n')+(s.error?'\nERROR '+s.error:'');$('log').scrollTop=1e9;$('stage').textContent=s.running?s.stage:'';
 const bar=$('bar');if(s.running&&s.percent===null){bar.classList.add('busy')}else{bar.classList.remove('busy');bar.firstElementChild.style.width=(s.running?s.percent:100)+'%'}
 if(!s.running){clearInterval(timer);$('runbtn').disabled=false;if(!s.error){$('out').value=s.out||$('out').value;loadResults()}}}
async function loadResults(){const out=$('out').value;rows=await (await fetch(`/api/results?out=${encodeURIComponent(out)}`)).json();filter=null;render()}
function render(){const out=$('out').value;const counts={};rows.forEach(r=>{const k=r.reasons.split('|')[0];counts[k]=(counts[k]||0)+1});
 $('count').textContent=rows.length?`${rows.length} frames in ${out}`:'no results in the output folder yet';
 $('chips').innerHTML=Object.keys(TXT).map(k=>`<span class="chip ${filter===k?'on':''}" onclick="filter=filter==='${k}'?null:'${k}';render()"><i class="dot" style="background:${COL[k]}"></i>${TXT[k]} <b>${counts[k]||0}</b></span>`).join('');
 const vids=[...new Set(rows.map(r=>r.video))];$('stats').innerHTML=rows.length?`<span>${vids.length} video${vids.length>1?'s':''}</span><span>${rows.filter(r=>+r.n_labels>0).length} frames with pseudo-labels</span>`:'';
 const shown=rows.filter(r=>!filter||r.reasons.split('|')[0]===filter);
 $('results').innerHTML=shown.map(r=>{const tags=r.reasons.split('|').map(k=>`<span class="tag" style="background:${COL[k]}">${TXT[k]}</span>`).join('');
  const th=`/file?out=${encodeURIComponent(out)}&path=${encodeURIComponent('review/'+r.image.split('/').pop())}`,full=`/file?out=${encodeURIComponent(out)}&path=${encodeURIComponent(r.image)}`;
  return `<figure data-img="${esc(r.image)}"><label class="rej"><input type="checkbox" onchange="this.closest('figure').classList.toggle('rejected',this.checked)"> reject</label>
   <img loading="lazy" src="${th}" data-full="${esc(full)}" data-cap="${esc(tidy(r.notes))}" onclick="lightbox(this)"><figcaption><div class="tags">${tags}</div>${esc(tidy(r.notes))}<br><span class="when">${esc(r.video)} · ${r.clock||r.time_s+'s'} · frame ${r.frame}</span></figcaption></figure>`}).join('')}
const OLD={'missed vehicle':'miss','false positive / hard negative':'no vehicle','wrong class':'confusion','low confidence':'low confidence'};
function tidy(n){n=n.split(' ; ')[0];for(const k in OLD)if(n.startsWith(k+':'))n=OLD[k]+':'+n.slice(k.length+1);return n}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;')}
function lightbox(img){$('lbimg').src=img.dataset.full;$('lbcap').textContent=img.dataset.cap;$('lightbox').style.display='flex'}
function rejectVisible(on){document.querySelectorAll('#results figure input').forEach(e=>{e.checked=on;e.closest('figure').classList.toggle('rejected',on)})}
async function deleteRejected(){const images=[...document.querySelectorAll('figure.rejected')].map(f=>f.dataset.img);if(!images.length){alert('Nothing is marked as rejected');return}
 if(!confirm(`Delete ${images.length} frame(s) from the output folder?`))return;
 const r=await (await fetch('/api/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({out:$('out').value,images})})).json();loadResults()}
</script></html>'''


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
