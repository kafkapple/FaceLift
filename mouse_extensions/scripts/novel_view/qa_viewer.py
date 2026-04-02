# no-split: single-file interactive viewer with tightly coupled UI state and rendering logic
#!/usr/bin/env python3
"""Dataset QA Viewer v2 — dual-mode (GT-view + Novel-view), zero-dependency.

Usage:
    cd /home/joon/dev/FaceLift
    python -m mouse_extensions.scripts.novel_view.qa_viewer

    # Mac: SSH tunnel + browser
    ssh -L 8899:localhost:8899 gpu03
    open http://localhost:8899
"""
import argparse
import json
import mimetypes
import os
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# ============================================================
# Config
# ============================================================
DEFAULT_DATASET = "/home/joon/data/derived/FaceLift/novel_view"
SPECIES_DATASET = "mouse_m5t2"
PORT = 8899

# ============================================================
# State
# ============================================================
cfg = {"data_dir": "", "exclude_path": ""}
exclude_data: dict = {
    "version": "1.0",
    "excluded_frames": {},
    "reason_tags": [
        "mesh_fitting_failure", "gs_lrm_artifact", "misalignment",
        "occlusion", "missing_data", "other",
    ],
}
_frames_cache = None
_tiers_cache = None


def save_exclude():
    with open(cfg["exclude_path"], "w") as f:
        json.dump(exclude_data, f, indent=2)


def discover_frames() -> list[int]:
    global _frames_cache
    if _frames_cache is not None:
        return _frames_cache
    meta_dir = os.path.join(cfg["data_dir"], "metadata")
    if not os.path.isdir(meta_dir):
        return []
    _frames_cache = sorted(
        int(f.replace(".json", ""))
        for f in os.listdir(meta_dir)
        if f.endswith(".json")
    )
    return _frames_cache


def discover_tiers() -> dict[str, list[str]]:
    global _tiers_cache
    if _tiers_cache is not None:
        return _tiers_cache
    tiers = {}
    try:
        entries = os.listdir(cfg["data_dir"])
    except FileNotFoundError:
        return tiers
    for name in sorted(entries):
        path = os.path.join(cfg["data_dir"], name)
        if os.path.isdir(path) and name != "metadata":
            subs = sorted(
                d for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d))
            )
            if subs:
                tiers[name] = subs
    _tiers_cache = tiers
    return _tiers_cache


# ============================================================
# Request Handler
# ============================================================
class QAHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        # Suppress verbose GET logs, show errors only
        if args and str(args[0]).startswith(("4", "5")):
            print(f"  [{args[0]}] {args[1] if len(args)>1 else ''}")

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path):
        if not os.path.isfile(path):
            self.send_error(404)
            return
        mime, _ = mimetypes.guess_type(path)
        mime = mime or "application/octet-stream"
        size = os.path.getsize(path)
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", size)
        self.send_header("Cache-Control", "public, max-age=3600")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        with open(path, "rb") as f:
            self.wfile.write(f.read())

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/":
            self._html(HTML_PAGE)

        elif path == "/api/config":
            self._json({
                "tiers": discover_tiers(),
                "reason_tags": exclude_data.get("reason_tags", []),
            })

        elif path == "/api/frame_indices":
            self._json({"indices": discover_frames()})

        elif path == "/api/frames":
            page = int(qs.get("page", [1])[0])
            page_size = int(qs.get("page_size", [6])[0])
            frames = discover_frames()
            total = len(frames)
            start = (page - 1) * page_size
            end = min(start + page_size, total)
            result = []
            for idx in frames[start:end]:
                key = str(idx)
                exc = exclude_data["excluded_frames"].get(key, {})
                result.append({
                    "frame_idx": idx,
                    "excluded": key in exclude_data["excluded_frames"],
                    "reason": exc.get("reason", ""),
                })
            self._json({
                "frames": result,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": max(1, (total + page_size - 1) // page_size),
            })

        elif path == "/api/stats":
            frames = discover_frames()
            self._json({
                "total_frames": len(frames),
                "excluded_count": len(exclude_data["excluded_frames"]),
            })

        elif path.startswith("/img/"):
            parts = path.split("/")
            if len(parts) != 5:
                self.send_error(400)
                return
            tier, view, frame_str = parts[2], parts[3], parts[4]
            try:
                frame_idx = int(frame_str)
            except ValueError:
                self.send_error(400)
                return
            fpath = os.path.join(
                cfg["data_dir"], tier, view, f"{frame_idx:05d}.png"
            )
            self._serve_file(fpath)

        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/exclude":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            key = str(body["frame_idx"])
            if body.get("exclude", False):
                exclude_data["excluded_frames"][key] = {
                    "reason": body.get("reason", ""),
                    "views": body.get("views", "all"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            else:
                exclude_data["excluded_frames"].pop(key, None)
            save_exclude()
            self._json({
                "ok": True,
                "total_excluded": len(exclude_data["excluded_frames"]),
            })
        else:
            self.send_error(404)


# ============================================================
# HTML (single-page app, inline)
# ============================================================
HTML_PAGE = r"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Dataset QA Viewer</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,system-ui,sans-serif;background:#f5f5f5;color:#333;font-size:13px;
  --iw:200px}
.hdr{position:sticky;top:0;z-index:100;background:#fff;border-bottom:2px solid #1976D2;
  padding:6px 10px}
.hr{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:3px}
.hr:last-child{margin-bottom:0}
.hdr h1{font-size:15px;color:#1976D2;white-space:nowrap}
.stats{font-size:11px;color:#888}
.sep{color:#ddd;user-select:none}
.lb{font-size:11px;color:#888;white-space:nowrap}
.mg,.ag,.vf{display:flex;gap:3px;align-items:center}
.mb{padding:3px 10px;border:1px solid #1976D2;background:none;color:#1976D2;
  border-radius:3px;cursor:pointer;font-size:11px}
.mb.on{background:#1976D2;color:#fff}
.ab{padding:2px 7px;border:1px solid #F57C00;background:none;color:#F57C00;
  border-radius:3px;cursor:pointer;font-size:11px;min-width:24px}
.ab.on{background:#F57C00;color:#fff}
.ab:disabled{opacity:.25;cursor:not-allowed}
.vb{padding:2px 8px;border:1px solid #7B1FA2;background:none;color:#7B1FA2;
  border-radius:3px;cursor:pointer;font-size:11px}
.vb.on{background:#7B1FA2;color:#fff}
.nav{display:flex;gap:4px;align-items:center}
.nav button{padding:3px 8px;border:1px solid #999;background:none;color:#555;
  border-radius:3px;cursor:pointer;font-size:11px}
.nav button:hover{background:#eee}
.ni{width:50px;padding:3px;border:1px solid #ccc;border-radius:3px;text-align:center;font-size:11px}
.zc{display:flex;align-items:center;gap:4px}
.zc input[type=range]{width:100px}
.zc span{font-size:11px;color:#666;min-width:42px}
.sv{padding:3px 10px;border:1px solid #43A047;background:none;color:#43A047;
  border-radius:3px;cursor:pointer;font-size:11px}
.sv:hover{background:#43A04711}
.fc{margin:4px 6px;border:1px solid #ddd;border-radius:4px;overflow:hidden;
  overflow-x:auto;background:#fff}
.fc.ex{border-color:#e53935;opacity:.35}
.fh{display:flex;align-items:center;justify-content:space-between;
  padding:4px 8px;background:#fafafa;border-bottom:1px solid #eee}
.fh b{color:#1976D2;font-size:12px}
.fa{display:flex;gap:4px;align-items:center}
.sr{padding:2px 4px;border:1px solid #ccc;border-radius:3px;font-size:10px}
.bx{padding:2px 8px;border:1px solid #e53935;background:none;color:#e53935;
  border-radius:3px;cursor:pointer;font-size:10px}
.bx.on{background:#e53935;color:#fff}
.be{padding:2px 8px;border:1px solid #43A047;background:none;color:#43A047;
  border-radius:3px;cursor:pointer;font-size:10px}
.be:hover{background:#43A04711}
.fg{display:grid;gap:1px;background:#e0e0e0}
.fg>div{background:#fff}
.co{background:#f5f5f5}
.ch{font-size:10px;font-weight:bold;color:#1976D2;text-align:center;padding:3px 2px;background:#f5f5f5}
.vl{font-size:10px;color:#666;display:flex;align-items:center;justify-content:center;
  background:#f9f9f9;padding:2px;min-width:64px}
.ic{text-align:center}
.ic img{width:var(--iw);height:auto;display:block;cursor:zoom-in;image-rendering:auto}
.ic img.err{opacity:.08;min-height:60px}
.zo{display:none;position:fixed;top:0;left:0;width:100%;height:100%;
  background:rgba(0,0,0,.9);z-index:200;cursor:zoom-out;
  justify-content:center;align-items:center}
.zo.on{display:flex}
.zo img{max-width:90vw;max-height:90vh}
kbd{background:#f0f0f0;padding:1px 4px;border-radius:2px;border:1px solid #ddd;font-size:10px}
</style>
</head><body>
<div class="hdr">
  <div class="hr">
    <h1>QA Viewer</h1>
    <span class="stats" id="stats">loading...</span>
    <span class="sep">|</span>
    <div class="mg">
      <button class="mb on" data-mode="novel" onclick="setMode('novel')">Novel View</button>
      <button class="mb" data-mode="gt" onclick="setMode('gt')">GT View</button>
    </div>
    <div class="ag" id="as" style="display:none">
      <span class="lb">Input:</span>
      <button class="ab" id="a1" data-n="1" onclick="setAbl(1)">1</button>
      <button class="ab" id="a2" data-n="2" onclick="setAbl(2)">2</button>
      <button class="ab" id="a3" data-n="3" onclick="setAbl(3)">3</button>
      <button class="ab" id="a4" data-n="4" onclick="setAbl(4)">4</button>
      <button class="ab" id="a5" data-n="5" onclick="setAbl(5)">5</button>
      <button class="ab on" id="a6" data-n="6" onclick="setAbl(6)">6</button>
    </div>
  </div>
  <div class="hr">
    <span class="lb">Views:</span>
    <div class="vf" id="vfb"></div>
    <span class="sep">|</span>
    <span class="lb">Zoom:</span>
    <div class="zc">
      <input type="range" id="zr" min="80" max="384" value="200"
             oninput="setZoom(+this.value)">
      <span id="zv">200px</span>
    </div>
    <span class="sep">|</span>
    <span class="lb">Per page:</span>
    <input class="ni" id="psi" type="number" min="1" max="48" value="6"
           style="width:40px" onchange="setPS(+this.value)">
    <span class="sep">|</span>
    <div class="nav">
      <button onclick="go(1)">&laquo;</button>
      <button onclick="go(P-1)">&lsaquo;</button>
      <input class="ni" id="pi" type="number" min="1" onchange="go(+this.value)">
      <span id="pp"></span>
      <button onclick="go(P+1)">&rsaquo;</button>
      <button onclick="go(TP)">&raquo;</button>
    </div>
    <span class="sep">|</span>
    <span class="lb">Frame:</span>
    <input class="ni" id="fi" type="number" min="0" placeholder="idx"
           style="width:60px" onchange="goF(+this.value)">
    <span class="sep">|</span>
    <button class="sv" onclick="savePage()">Save Page</button>
    <span style="font-size:10px;color:#bbb"><kbd>&larr;</kbd><kbd>&rarr;</kbd> page
      &nbsp; click=zoom &nbsp; <kbd>Esc</kbd>=close</span>
  </div>
</div>
<div id="ct"></div>
<div class="zo" id="zo" onclick="cz()"><img id="zi" src=""></div>

<script>
let MODE='novel',ABL=6,P=1,TP=1,PS=6,AF=[],T={},FI=[],ZM=200;
let VF=new Set();
const RS=['mesh_fitting_failure','gs_lrm_artifact','misalignment','occlusion','missing_data','other'];
const NV=['bottom','top','front_low','side_low'];
const GC=['cam_000','cam_001','cam_002','cam_003','cam_004','cam_005'];

async function init(){
  const st=document.getElementById('stats');
  try{
    st.textContent='Loading config...';
    const c=await(await fetch('/api/config')).json();
    T=c.tiers||{};
    st.textContent=`Loading frame index (${Object.keys(T).length} tiers)...`;
    const f=await(await fetch('/api/frame_indices')).json();
    FI=f.indices;
    st.textContent=`${FI.length} frames found. Rendering...`;
    NV.forEach(v=>VF.add(v));
    updAbl();updVF();await ld(1);
  }catch(e){
    st.textContent='\u274c '+e.message;st.style.color='#e53935';
    document.getElementById('ct').innerHTML=
      '<div style="padding:40px;text-align:center;color:#e53935;font-size:14px">'
      +'<p><b>Connection failed</b></p>'
      +'<p style="margin-top:8px;color:#888">Server must run on gpu03 (where data exists).</p>'
      +'<pre style="margin-top:12px;text-align:left;display:inline-block;background:#f5f5f5;'
      +'padding:12px;border-radius:4px;font-size:12px;color:#333">'
      +'# 1. gpu03\uc5d0\uc11c \uc11c\ubc84 \uc2dc\uc791\n'
      +'ssh gpu03\ncd /home/joon/dev/FaceLift\n'
      +'python -m mouse_extensions.scripts.novel_view.qa_viewer\n\n'
      +'# 2. Mac\uc5d0\uc11c SSH \ud130\ub110\n'
      +'ssh -L 8899:localhost:8899 gpu03\n'
      +'open http://localhost:8899</pre></div>';
  }
}

/* === Mode / Ablation / View Filter === */
function setMode(m){
  MODE=m;
  document.querySelectorAll('.mb').forEach(b=>b.classList.toggle('on',b.dataset.mode===m));
  document.getElementById('as').style.display=m==='gt'?'':'none';
  VF.clear();(m==='novel'?NV:GC).forEach(v=>VF.add(v));
  updVF();
  PS=m==='gt'?4:6;document.getElementById('psi').value=PS;
  ld(1);
}
function setAbl(n){
  ABL=n;
  document.querySelectorAll('.ab').forEach(b=>b.classList.toggle('on',+b.dataset.n===n));
  ld(P);
}
function setPS(n){PS=Math.max(1,Math.min(n,48));ld(1);}
function setZoom(v){
  ZM=v;
  document.body.style.setProperty('--iw',v+'px');
  document.getElementById('zv').textContent=v===384?'384 (native)':v+'px';
}
function updAbl(){
  for(let n=1;n<=6;n++){
    const b=document.getElementById('a'+n);
    if(b){const t=n===6?'gt_views':`ablation_${n}view`;b.disabled=!T[t];}
  }
}
function updVF(){
  const all=MODE==='novel'?NV:GC;
  let h='';
  for(const v of all)
    h+=`<button class="vb${VF.has(v)?' on':''}" data-v="${v}" onclick="togV('${v}')">${v}</button>`;
  document.getElementById('vfb').innerHTML=h;
}
function togV(v){
  if(VF.has(v)){if(VF.size>1)VF.delete(v);}else VF.add(v);
  updVF();ld(P);
}

/* === Columns / Views === */
function cols(){
  if(MODE==='novel'){
    const c=[];
    if(T.tier0_raw)c.push({t:'tier0_raw',l:'GS-LRM 6v'});
    if(T.pseudo_gt)c.push({t:'pseudo_gt',l:'MAMMAL'});
    if(T.pseudo_gt_textured)c.push({t:'pseudo_gt_textured',l:'MAMMAL (tex)'});
    return c;
  }else{
    const c=[];
    if(T.gt_rgb)c.push({t:'gt_rgb',l:'GT RGB'});
    const g=ABL===6?'gt_views':`ablation_${ABL}view`;
    if(T[g])c.push({t:g,l:`GS-LRM ${ABL}v`});
    return c;
  }
}
function views(){return (MODE==='novel'?NV:GC).filter(v=>VF.has(v));}

/* === Load & Render === */
async function ld(p){
  P=p;
  const st=document.getElementById('stats');
  st.textContent=`Loading page ${p}...`;
  const r=await(await fetch(`/api/frames?page=${p}&page_size=${PS}`)).json();
  TP=r.total_pages;AF=r.frames;
  document.getElementById('pi').value=p;
  document.getElementById('pi').max=TP;
  document.getElementById('pp').textContent=`/ ${TP}`;
  ren(r.frames);
  const s=await(await fetch('/api/stats')).json();
  st.textContent=`${s.total_frames} frames | ${s.excluded_count} excluded | Page ${p}/${TP}`;
}

function ren(fs){
  const vw=views(),cl=cols();
  if(!cl.length){document.getElementById('ct').innerHTML=
    '<div style="padding:40px;text-align:center;color:#999">No data for this mode</div>';return;}
  let h='';
  for(const f of fs){
    h+=`<div class="fc${f.excluded?' ex':''}" id="f${f.frame_idx}">`;
    h+=`<div class="fh"><b>Frame ${f.frame_idx}</b><div class="fa">`;
    h+=`<select class="sr" id="r${f.frame_idx}"><option value="">--</option>`;
    for(const r of RS)h+=`<option value="${r}"${f.reason===r?' selected':''}>${r}</option>`;
    h+=`</select>`;
    h+=`<button class="bx${f.excluded?' on':''}" onclick="tx(${f.frame_idx})">${f.excluded?'\u2717 Excluded':'Exclude'}</button>`;
    h+=`<button class="be" onclick="expF(${f.frame_idx})">Export</button>`;
    h+=`</div></div>`;
    h+=`<div class="fg" style="grid-template-columns:64px repeat(${cl.length},auto)">`;
    h+=`<div class="co"></div>`;
    for(const c of cl)h+=`<div class="ch">${c.l}</div>`;
    for(const v of vw){
      h+=`<div class="vl">${v}</div>`;
      for(const c of cl)
        h+=`<div class="ic"><img src="/img/${c.t}/${v}/${f.frame_idx}" loading="lazy" `
          +`crossorigin="anonymous" onerror="this.classList.add('err')" onclick="zm(this.src)"></div>`;
    }
    h+=`</div></div>`;
  }
  document.getElementById('ct').innerHTML=h;
}

/* === Exclude === */
async function tx(i){
  const f=AF.find(x=>x.frame_idx===i);
  await fetch('/api/exclude',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({frame_idx:i,exclude:!f.excluded,reason:document.getElementById('r'+i).value})});
  ld(P);
}

/* === Navigation === */
function go(p){p=Math.max(1,Math.min(p,TP));ld(p);scrollTo(0,0);}
function goF(t){
  let lo=0,hi=FI.length-1;
  while(lo<hi){const m=(lo+hi)>>1;if(FI[m]<t)lo=m+1;else hi=m;}
  go(Math.floor(lo/PS)+1);
}

/* === Zoom overlay === */
function zm(s){document.getElementById('zi').src=s;document.getElementById('zo').classList.add('on');}
function cz(){document.getElementById('zo').classList.remove('on');}

/* === Export single frame === */
async function expF(idx){
  const vw=views(),cl=cols();
  const cS=384,lW=64,hH=22,tH=24;
  const W=lW+cl.length*cS,H=tH+hH+vw.length*cS;
  const cv=document.createElement('canvas');cv.width=W;cv.height=H;
  const x=cv.getContext('2d');
  x.fillStyle='#fff';x.fillRect(0,0,W,H);
  x.fillStyle='#333';x.font='bold 14px sans-serif';
  x.fillText(`Frame ${idx} | ${MODE==='novel'?'Novel':'GT'} | ${ABL}v`,6,17);
  x.font='bold 11px sans-serif';x.fillStyle='#1976D2';
  cl.forEach((c,i)=>x.fillText(c.l,lW+i*cS+4,tH+14));
  const ps=[];
  vw.forEach((v,r)=>{
    const y=tH+hH+r*cS;
    x.fillStyle='#666';x.font='11px sans-serif';x.fillText(v,2,y+cS/2+3);
    x.strokeStyle='#eee';x.beginPath();x.moveTo(0,y);x.lineTo(W,y);x.stroke();
    cl.forEach((c,ci)=>{
      ps.push(li(`/img/${c.t}/${v}/${idx}`).then(im=>{
        if(im)x.drawImage(im,lW+ci*cS,y,cS,cS);
      }));
    });
  });
  await Promise.all(ps);
  const a=document.createElement('a');
  a.download=`qa_${String(idx).padStart(5,'0')}_${MODE}_${ABL}v.png`;
  a.href=cv.toDataURL('image/png');a.click();
}

/* === Save Page (full grid export) === */
async function savePage(){
  const vw=views(),cl=cols(),fs=AF;
  if(!fs.length||!cl.length)return;
  const cS=384,lW=64,hH=22,tH=28,fGap=8;
  const gridW=lW+cl.length*cS;
  const fH=tH+hH+vw.length*cS;
  const W=gridW,H=32+fs.length*(fH+fGap);
  const cv=document.createElement('canvas');cv.width=W;cv.height=H;
  const x=cv.getContext('2d');
  x.fillStyle='#f5f5f5';x.fillRect(0,0,W,H);
  x.fillStyle='#1976D2';x.font='bold 14px sans-serif';
  x.fillText(`QA Viewer | ${MODE==='novel'?'Novel View':'GT View'} | ${ABL}v | Page ${P}/${TP}`,8,20);
  const st=document.getElementById('stats');
  const origSt=st.textContent;
  let loaded=0;const total=fs.length*vw.length*cl.length;
  st.textContent=`Exporting page... 0/${total} images`;
  const ps=[];
  fs.forEach((f,fi)=>{
    const fY=32+fi*(fH+fGap);
    x.fillStyle='#fff';x.fillRect(0,fY,W,fH);
    x.fillStyle=f.excluded?'#e53935':'#333';x.font='bold 12px sans-serif';
    x.fillText(`Frame ${f.frame_idx}${f.excluded?' [EXCLUDED]':''}`,8,fY+18);
    x.font='bold 10px sans-serif';x.fillStyle='#1976D2';
    cl.forEach((c,ci)=>x.fillText(c.l,lW+ci*cS+4,fY+tH+14));
    vw.forEach((v,vi)=>{
      const y=fY+tH+hH+vi*cS;
      x.fillStyle='#666';x.font='10px sans-serif';x.fillText(v,2,y+cS/2+3);
      cl.forEach((c,ci)=>{
        ps.push(li(`/img/${c.t}/${v}/${f.frame_idx}`).then(im=>{
          if(im)x.drawImage(im,lW+ci*cS,y,cS,cS);
          loaded++;
          if(loaded%5===0||loaded===total)
            st.textContent=`Exporting page... ${loaded}/${total} images`;
        }));
      });
    });
  });
  await Promise.all(ps);
  st.textContent='Downloading...';
  const a=document.createElement('a');
  a.download=`qa_page${P}_${MODE}_${ABL}v.png`;
  a.href=cv.toDataURL('image/png');a.click();
  st.textContent=origSt;
}

/* === Helpers === */
function li(s){return new Promise(r=>{const i=new Image();i.crossOrigin='anonymous';
  i.onload=()=>r(i);i.onerror=()=>r(null);i.src=s;})}

document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
  if(e.key==='ArrowLeft')go(P-1);
  if(e.key==='ArrowRight')go(P+1);
  if(e.key==='Escape')cz();
});
init();
</script>
</body></html>"""


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Dataset QA Viewer v2 (stdlib, zero dependencies)"
    )
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    cfg["data_dir"] = os.path.join(args.dataset_dir, SPECIES_DATASET)
    cfg["exclude_path"] = os.path.join(args.dataset_dir, "exclude_list.json")

    global exclude_data
    if os.path.exists(cfg["exclude_path"]):
        with open(cfg["exclude_path"]) as f:
            exclude_data = json.load(f)
        print(f"Loaded {len(exclude_data['excluded_frames'])} excluded frames")

    frames = discover_frames()
    tiers = discover_tiers()
    print(f"Dataset : {cfg['data_dir']}")
    print(f"Frames  : {len(frames)}")
    print(f"Tiers   : {', '.join(tiers.keys())}")
    print(f"Exclude : {cfg['exclude_path']}")
    print(f"Server  : http://{args.host}:{args.port}")
    print()
    print("=== Access from Mac ===")
    print(f"  ssh -L {args.port}:localhost:{args.port} gpu03")
    print(f"  open http://localhost:{args.port}")
    print()

    server = HTTPServer((args.host, args.port), QAHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
