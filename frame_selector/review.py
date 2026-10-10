"""A static review page: every selected frame as a thumbnail, grouped by reason."""
import html
import os

from .export import REASON_TEXT, read_manifest


def write_review_page(out_dir):
    """Static review page built from everything in the manifest (thumbnails already on disk)."""
    sections = {r: [] for r in REASON_TEXT}                     # page from everything selected so far
    for row in read_manifest(out_dir):
        if os.path.exists(os.path.join(out_dir, row['image'])):
            tn = os.path.basename(row['image'])
            when = row['clock'] or '%ss' % row['time_s']
            sections[row['reasons'].split('|')[0]].append(
                '<figure data-img="%s"><img src="review/%s" loading="lazy"><figcaption><label><input type="checkbox"> reject</label> '
                '<b>%s</b> frame %d %s<br>%s</figcaption></figure>'
                % (html.escape(row['image']), html.escape(tn), html.escape(row['video']), int(row['frame']), html.escape(when), html.escape(row['notes'])))
    body = ''.join(f'<h2>{html.escape(REASON_TEXT[r])} ({len(figs)})</h2><div class="grid">{"".join(figs)}</div>'
                   for r, figs in sections.items() if figs)
    page = f'''<!doctype html><meta charset="utf-8"><title>Frame selector review</title>
<style>body{{font:14px system-ui;margin:16px;background:#111;color:#ddd}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:12px}}
figure{{margin:0;background:#1c1c1c;padding:6px;border-radius:6px}} img{{width:100%;display:block}} figcaption{{padding:6px 2px;line-height:1.4}}
h2{{margin:24px 0 8px}} button{{font:inherit;padding:6px 12px}} .legend span{{display:inline-block;margin-right:14px;padding-left:18px;border-left:12px solid}}</style>
<h1>Frame selector review</h1>
<p class="legend">Green = pseudo-label to correct. Flagged box: <span style="border-color:#f00">missed</span><span style="border-color:#ffa500">false positive</span>
<span style="border-color:#f0f">wrong class</span><span style="border-color:#dcdc00">low confidence</span></p>
<p><button onclick="dl()">Download rejected list</button> (frames ticked "reject" are written to rejected.json for you to delete)</p>
{body}
<script>function dl(){{const r=[...document.querySelectorAll('figure')].filter(f=>f.querySelector('input').checked).map(f=>f.dataset.img);
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(r,null,1)],{{type:'application/json'}}));a.download='rejected.json';a.click();}}</script>'''
    with open(os.path.join(out_dir, 'review.html'), 'w') as fh:
        fh.write(page)
