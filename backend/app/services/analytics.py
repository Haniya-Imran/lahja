"""Review analytics workspace: statistics over results the user has analysed in this session.

Stateless by design: the browser sends the results it holds; nothing is stored on the server.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from nlp.language_detection.detect import LABELS as LANG_LABELS

from . import registry

SENTIMENTS = ("positive", "neutral", "negative")


def summary(items: list[dict]) -> dict:
    scored = [it for it in items if it.get("sentiment") in SENTIMENTS]
    n = len(scored)
    sent = Counter(it["sentiment"] for it in scored)
    lang = Counter(it.get("language") or "uncertain" for it in items)
    cross = defaultdict(Counter)
    for it in scored:
        cross[it.get("language") or "uncertain"][it["sentiment"]] += 1
    conf = [float(it.get("confidence") or 0) for it in scored]
    buckets = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    conf_hist = [{"range": f"{int(a * 100)}–{min(int(b * 100), 100)}%", "count": sum(a <= c < b for c in conf)} for a, b in buckets]
    low = sum(bool(it.get("low_confidence")) for it in scored)
    insights = registry.get("insights").analyse(
        [{"text": it["text"], "sentiment": it["sentiment"], "language": it.get("language")} for it in scored])
    pct = (lambda k: round(sent[k] / n * 100, 1)) if n else (lambda k: 0.0)
    net = round(pct("positive") - pct("negative"), 1) if n else None
    business = []
    if n:
        business.append(f"{pct('positive')}% positive, {pct('neutral')}% neutral and {pct('negative')}% negative across {n:,} reviews.")
        if low:
            business.append(f"{low:,} reviews ({round(low / n * 100, 1)}%) have low model confidence; read these yourself before acting on them.")
        worst = max(cross.items(), key=lambda kv: kv[1]["negative"] / max(sum(kv[1].values()), 1), default=None)
        if worst and sum(worst[1].values()) >= 10 and worst[1]["negative"]:
            share = round(worst[1]["negative"] / sum(worst[1].values()) * 100, 1)
            business.append(f"{LANG_LABELS.get(worst[0], worst[0])} reviews have the highest negative share ({share}%).")
        neg_terms = [t["term"] for t in insights["words"].get("negative", [])[:5]]
        if neg_terms:
            business.append("Words that stand out in negative reviews: " + ", ".join(neg_terms) + ".")
    return {
        "total": len(items), "scored": n, "skipped": len(items) - n,
        "sentiment": {k: sent[k] for k in SENTIMENTS},
        "sentiment_pct": {k: pct(k) for k in SENTIMENTS},
        "net_sentiment": net,
        "language": [{"language": k, "label": LANG_LABELS.get(k, k), "count": v} for k, v in lang.most_common()],
        "sentiment_by_language": [{"language": k, "label": LANG_LABELS.get(k, k), **{s: c[s] for s in SENTIMENTS}}
                                  for k, c in sorted(cross.items(), key=lambda kv: -sum(kv[1].values()))],
        "confidence": {"mean": round(sum(conf) / n, 4) if n else None, "histogram": conf_hist, "low_confidence": low},
        "text": insights,
        "business_insights": business,
    }
