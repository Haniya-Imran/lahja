"""Module G — downloadable CSV results and a printable HTML report (print to PDF from the browser)."""
from __future__ import annotations

import csv
import html
import io
from datetime import datetime, timezone

from nlp.language_detection.detect import LABELS as LANG_LABELS

from . import analytics as analytics_service

BRAND = "LingoAI"


def results_csv(items: list[dict]) -> str:
    buf = io.StringIO()
    buf.write("﻿")  # BOM so Excel shows Urdu correctly
    w = csv.writer(buf)
    w.writerow(["review", "language", "sentiment", "confidence", "p_negative", "p_neutral", "p_positive", "low_confidence"])
    for it in items:
        p = it.get("probabilities") or {}
        w.writerow([it.get("text", ""), LANG_LABELS.get(it.get("language") or "", it.get("language") or ""),
                    it.get("sentiment") or "", it.get("confidence") if it.get("confidence") is not None else "",
                    p.get("negative", ""), p.get("neutral", ""), p.get("positive", ""),
                    "yes" if it.get("low_confidence") else "no"])
    return buf.getvalue()


_CSS = """
body{font-family:Inter,system-ui,sans-serif;color:#1a1a2e;max-width:980px;margin:28px auto;padding:0 20px;line-height:1.5}
h1{font-size:28px;margin:0}h2{font-size:19px;margin:28px 0 8px;border-bottom:2px solid #e6e3f5;padding-bottom:6px}
.meta{color:#5d5a78;font-size:13px}.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:14px}
.kpi{border:1px solid #e6e3f5;border-radius:10px;padding:10px 12px}.kpi b{display:block;font-size:22px}.kpi span{font-size:12px;color:#5d5a78}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px 8px;border-bottom:1px solid #eee;text-align:left;vertical-align:top}
th{background:#f6f4fd}td.n{text-align:right;font-variant-numeric:tabular-nums}
.urdu{font-family:'Noto Nastaliq Urdu','Jameel Noori Nastaleeq',serif;line-height:2}
.pos{color:#0f7a55}.neg{color:#b3263c}.neu{color:#6b6880}ul{margin:4px 0 0 18px}.note{font-size:12px;color:#5d5a78}
@media print{body{margin:0}.noprint{display:none}}
"""


def _cls(s):
    return {"positive": "pos", "negative": "neg", "neutral": "neu"}.get(s or "", "")


def html_report(items: list[dict], model: dict, max_rows: int = 500) -> str:
    s = analytics_service.summary(items)
    now = datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")
    rows = []
    for it in items[:max_rows]:
        t = it.get("text", "")
        urdu = any("؀" <= ch <= "ۿ" for ch in t)
        rows.append(f"<tr><td dir='auto' class='{'urdu' if urdu else ''}'>{html.escape(t[:400])}</td>"
                    f"<td>{html.escape(LANG_LABELS.get(it.get('language') or '', '—'))}</td>"
                    f"<td class='{_cls(it.get('sentiment'))}'>{html.escape(it.get('sentiment') or 'skipped')}</td>"
                    f"<td class='n'>{'' if it.get('confidence') is None else f'{it['confidence'] * 100:.1f}%'}</td></tr>")
    words = "".join(
        f"<td><ul>{''.join(f'<li dir=auto>{html.escape(w['term'])} ({w['count']})</li>' for w in s['text']['words'][lab][:10])}</ul></td>"
        for lab in ("positive", "neutral", "negative"))
    lang_rows = "".join(f"<tr><td>{html.escape(r['label'])}</td><td class='n'>{r['positive']}</td><td class='n'>{r['neutral']}</td>"
                        f"<td class='n'>{r['negative']}</td></tr>" for r in s["sentiment_by_language"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{BRAND} review report</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu&display=swap"><style>{_CSS}</style></head><body>
<p class="noprint note">Use your browser's Print → Save as PDF to keep this report as a PDF.</p>
<h1>{BRAND} — review analysis report</h1>
<div class="meta">Generated {now} · Model: {html.escape(str(model.get('name') or ''))} v{html.escape(str(model.get('version') or ''))}</div>
<div class="kpis">
<div class="kpi"><span>Reviews analysed</span><b>{s['scored']:,}</b></div>
<div class="kpi"><span>Positive</span><b class="pos">{s['sentiment_pct']['positive']}%</b></div>
<div class="kpi"><span>Neutral</span><b class="neu">{s['sentiment_pct']['neutral']}%</b></div>
<div class="kpi"><span>Negative</span><b class="neg">{s['sentiment_pct']['negative']}%</b></div></div>
<h2>Key insights</h2><ul>{''.join(f'<li>{html.escape(b)}</li>' for b in s['business_insights'])}</ul>
<h2>Sentiment by language</h2><table><tr><th>Language / script</th><th>Positive</th><th>Neutral</th><th>Negative</th></tr>{lang_rows}</table>
<h2>Distinctive words</h2><table><tr><th>Positive</th><th>Neutral</th><th>Negative</th></tr><tr>{words}</tr></table>
<p class="note">{html.escape(s['text']['note'])}</p>
<h2>Reviews{' (first ' + str(max_rows) + ')' if len(items) > max_rows else ''}</h2>
<table><tr><th>Review</th><th>Language</th><th>Sentiment</th><th>Confidence</th></tr>{''.join(rows)}</table>
<p class="note">Predictions come from a fine-tuned Transformer. Confidence is the model's calibrated probability, not a guarantee;
low-confidence reviews deserve a human read. Sarcasm, slang and very short reviews are common error sources.</p>
</body></html>"""
