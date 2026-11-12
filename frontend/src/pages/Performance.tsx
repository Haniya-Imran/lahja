import { LANG_LABEL, SENT_LABEL, pct } from '../brand'
import Plot from '../components/Plot'
import { ErrorState, Info, Kpi, Loading, PageHead } from '../components/ui'
import { useModelInfo } from '../lib/useModel'

/* eslint-disable @typescript-eslint/no-explicit-any */
export default function Performance() {
  const q = useModelInfo()
  if (q.loading) return <Loading label="Loading the model card…" />
  if (q.error) return <ErrorState message={q.error} onRetry={q.retry} />
  const { metadata: m, card: c } = q.data
  const t = c.served_model_test
  const base = c.baseline
  const cm = t.confusion_matrix
  const tr = m.training || {}
  const hist = tr.history || []
  const comp = c.model_comparison
  const langs = Object.entries(c.test_by_language as Record<string, any>).sort((a, b) => b[1].n - a[1].n)
  const reliable = langs.filter(([, v]) => v.reliable)

  return (
    <>
      <PageHead eyebrow="Evaluation" title="Model Performance"
        lead={<>Every number on this page was measured on the held-out test split ({t.n.toLocaleString()} reviews) that was never used for training, early stopping or calibration. It is the exact compressed model that this site serves.</>} />
      <div className="stack">
        <div className="kpis">
          <Kpi ink label="Macro-F1 (served)" value={t.macro_f1.toFixed(3)} info="Average of the F1 score of the three classes, so each sentiment counts equally." hint={c.transformer_only_test ? `Transformer alone ${c.transformer_only_test.macro_f1.toFixed(3)}` : undefined} />
          <Kpi label="Accuracy" value={pct(t.accuracy)} hint={base ? `baseline ${pct(base.test.accuracy)}` : undefined} />
          <Kpi label="Weighted F1" value={t.weighted_f1.toFixed(3)} />
          <Kpi label="Calibration error" value={t.ece.toFixed(3)} info="Expected calibration error: gap between confidence and actual accuracy. Lower is better; 0 is perfect." hint={`calibrated on validation (T=${c.ensemble?.temperature ?? m.temperature})`} />
          <Kpi label="Model size" value={`${m.size_mb} MB`} hint={`Transformer (int8 ONNX) + small TF-IDF model · ${c.throughput_reviews_per_sec_2cpu} reviews/s on 2 CPUs`} />
        </div>

        {c.ensemble && c.transformer_only_test && base && (
          <div className="card">
            <div className="card-head"><div><h3>How the served model is built<Info text="Both models score every review; their probabilities are mixed with fixed weights chosen on the validation split, then re-calibrated." /></h3>
              <p className="sub">{Math.round(c.ensemble.weight_transformer * 100)}% fine-tuned Transformer + {Math.round(c.ensemble.weight_tfidf * 100)}% TF-IDF character model. Weights chosen on {c.ensemble.chosen_on}.</p></div></div>
            <div className="table-wrap"><table className="tbl">
              <thead><tr><th>Model (same {t.n.toLocaleString()} test reviews)</th><th className="n">Macro-F1</th><th className="n">Accuracy</th>{reliable.map(([k]) => <th key={k} className="n">{LANG_LABEL[k]}</th>)}</tr></thead>
              <tbody>
                <tr><td>Fine-tuned Transformer only</td><td className="n">{c.transformer_only_test.macro_f1.toFixed(3)}</td><td className="n">{pct(c.transformer_only_test.accuracy)}</td>{reliable.map(([k]) => <td key={k} className="n">{c.transformer_only_by_language[k]?.macro_f1?.toFixed(3) ?? '—'}</td>)}</tr>
                <tr><td>TF-IDF model only</td><td className="n">{base.test.macro_f1.toFixed(3)}</td><td className="n">{pct(base.test.accuracy)}</td>{reliable.map(([k]) => <td key={k} className="n">{base.test_by_language[k]?.macro_f1?.toFixed(3) ?? '—'}</td>)}</tr>
                <tr style={{ fontWeight: 700 }}><td>LingoAI blend (served)</td><td className="n">{t.macro_f1.toFixed(3)}</td><td className="n">{pct(t.accuracy)}</td>{reliable.map(([k, v]) => <td key={k} className="n">{v.macro_f1.toFixed(3)}</td>)}</tr>
              </tbody></table></div>
            <p className="small muted" style={{ marginTop: 10 }}>The Transformer understands meaning and word order; the character model copes better with the many spellings of Roman Urdu. Together they beat each one alone on the overall test score.</p>
          </div>
        )}

        <div className="card">
          <div className="card-head"><div><h3>The model</h3><p className="sub">{m.display_name} · version {m.version}</p></div></div>
          <div className="grid g3 small">
            <div><div className="label">Base model</div><p>{m.base_model} ({m.base_model_license})</p></div>
            <div><div className="label">Training</div><p className="num">{(tr.train_rows || 0).toLocaleString()} reviews · {tr.history?.length || 0} evaluations · best epoch {tr.best?.epoch} · {tr.minutes_total} min on CPU</p></div>
            <div><div className="label">Trainable parameters</div><p className="num">{(tr.trainable_params || 0).toLocaleString()} — every weight was fine-tuned{c.vocab_trimming.before_training ? ` (down from ${(c.vocab_trimming.before_training.params_before / 1e6).toFixed(0)}M after trimming the vocabulary)` : ''}</p></div>
            <div><div className="label">Compression</div><p className="num">Vocabulary {(c.vocab_trimming.before_training?.original_pieces ?? c.vocab_trimming.original_pieces).toLocaleString()} → {c.vocab_trimming.kept_pieces.toLocaleString()} pieces (test text never used); int8 weights. Agrees with the full-precision model on {pct(c.served_vs_pytorch_agreement)} of test reviews.</p></div>
            <div><div className="label">Transformer test macro-F1</div><p className="num">full precision {c.pytorch_fp32_test.macro_f1.toFixed(3)}, int8 {(c.transformer_only_test ?? t).macro_f1.toFixed(3)}</p></div>
            <div><div className="label">Low confidence (&lt; {Math.round(c.low_confidence.threshold * 100)}%)</div><p className="num">{pct(c.low_confidence.share)} of test reviews; accuracy {pct(c.low_confidence.accuracy_low)} vs {pct(c.low_confidence.accuracy_rest)} for the rest</p></div>
          </div>
        </div>

        <div className="grid g2">
          <div className="card"><div className="card-head"><h3>Confusion matrix<Info text="Rows are the true label, columns the model's prediction. The diagonal is correct." /></h3></div>
            <Plot label="Confusion matrix heatmap" height={300} data={[{ type: 'heatmap', z: cm.matrix, x: cm.labels.map((l: string) => SENT_LABEL[l]), y: cm.labels.map((l: string) => SENT_LABEL[l]),
              colorscale: [[0, '#FBF8F3'], [1, '#3B36A8']], showscale: false, text: cm.matrix, texttemplate: '%{text}', hovertemplate: 'true %{y} → predicted %{x}: %{z}<extra></extra>' }]}
              layout={{ xaxis: { title: { text: 'Predicted' }, side: 'bottom' }, yaxis: { title: { text: 'True' }, autorange: 'reversed' } }} /></div>
          <div className="card"><div className="card-head"><h3>Per sentiment</h3></div>
            <div className="table-wrap"><table className="tbl"><thead><tr><th>Sentiment</th><th className="n">Precision</th><th className="n">Recall</th><th className="n">F1</th><th className="n">Reviews</th></tr></thead>
              <tbody>{Object.entries(t.per_class as Record<string, any>).map(([k, v]) => <tr key={k}><td>{SENT_LABEL[k]}</td><td className="n">{v.precision.toFixed(3)}</td><td className="n">{v.recall.toFixed(3)}</td><td className="n">{v.f1.toFixed(3)}</td><td className="n">{v.support}</td></tr>)}</tbody></table></div>
            {hist.length > 0 && <>
              <div className="label" style={{ margin: '16px 0 4px' }}>Validation macro-F1 during training</div>
              <Plot label="Validation macro-F1 per epoch" height={180} data={[{ type: 'scatter', mode: 'lines+markers', x: hist.map((h: any) => `epoch ${h.epoch}`), y: hist.map((h: any) => h.val_macro_f1), line: { color: '#3B36A8', width: 3 }, marker: { size: 8 } }]}
                layout={{ margin: { l: 48, r: 16, t: 6, b: 30 } }} /></>}
          </div>
        </div>

        <div className="card">
          <div className="card-head"><div><h3>By language<Info text="Language as detected by LingoAI's language detector. Groups with fewer than 30 test reviews are marked unreliable." /></h3><p className="sub">Served LingoAI model vs the TF-IDF + logistic-regression baseline on the same test reviews.</p></div></div>
          <Plot label="Macro-F1 by language, Transformer vs baseline" height={280} data={[
            { type: 'bar', name: 'LingoAI (served)', x: langs.map(([k]) => LANG_LABEL[k]), y: langs.map(([, v]) => v.macro_f1), marker: { color: '#3B36A8' } },
            ...(base ? [{ type: 'bar', name: 'TF-IDF baseline', x: langs.map(([k]) => LANG_LABEL[k]), y: langs.map(([k]) => base.test_by_language[k]?.macro_f1 ?? null), marker: { color: '#E8A33D' } }] : []),
          ]} layout={{ barmode: 'group', yaxis: { range: [0, 1], title: { text: 'Macro-F1' } } }} />
          <div className="table-wrap" style={{ marginTop: 10 }}><table className="tbl">
            <thead><tr><th>Language</th><th className="n">Test reviews</th><th className="n">Accuracy</th><th className="n">Macro-F1</th><th className="n">Baseline F1</th><th>Note</th></tr></thead>
            <tbody>{langs.map(([k, v]) => <tr key={k}><td>{LANG_LABEL[k]}</td><td className="n">{v.n}</td><td className="n">{pct(v.accuracy)}</td><td className="n">{v.macro_f1.toFixed(3)}</td>
              <td className="n">{base?.test_by_language[k]?.macro_f1?.toFixed(3) ?? '—'}</td><td className="small muted">{v.reliable ? (v.labels_present.length < 3 ? `Only ${v.labels_present.join(', ')} labels present` : '') : v.note}</td></tr>)}</tbody>
          </table></div>
        </div>

        {comp && (
          <div className="card"><div className="card-head"><div><h3>Why this base model</h3><p className="sub">{comp.protocol}</p></div></div>
            <div className="table-wrap"><table className="tbl"><thead><tr><th>Candidate</th><th>License</th><th className="n">Val macro-F1</th><th className="n">Minutes (8k)</th><th className="n">Unknown tokens</th><th className="n">Tokens / word (Urdu)</th></tr></thead>
              <tbody>{comp.candidates.map((x: any) => <tr key={x.key}><td>{x.name}{x.key === comp.chosen && <span className="badge b-pos" style={{ marginLeft: 6 }}>chosen</span>}</td><td>{x.license}</td>
                <td className="n">{x.val_macro_f1.toFixed(3)}</td><td className="n">{x.train_minutes_8k}</td>
                <td className="n">{Object.values(x.tokenizer_by_language as Record<string, any>).reduce((a: number, v: any) => a + v.unknown_tokens, 0)}</td>
                <td className="n">{x.tokenizer_by_language.urdu?.tokens_per_word ?? '—'}</td></tr>)}</tbody></table></div>
            <p className="small" style={{ marginTop: 10 }}>{comp.reason}</p></div>
        )}

        <div className="card">
          <div className="card-head"><div><h3>Error analysis</h3><p className="sub">Accuracy on test reviews with specific traits, and real misclassified test examples.</p></div></div>
          <div className="grid g2">
            <div className="table-wrap"><table className="tbl"><thead><tr><th>Trait</th><th className="n">Reviews</th><th className="n">Accuracy</th></tr></thead>
              <tbody><tr><td>All test reviews</td><td className="n">{t.n}</td><td className="n">{pct(t.accuracy)}</td></tr>
                {Object.entries(c.error_analysis.by_category as Record<string, any>).map(([k, v]) => <tr key={k}><td>{k.replace(/_/g, ' ')}</td><td className="n">{v.n}</td><td className="n">{pct(v.accuracy)}</td></tr>)}</tbody></table></div>
            <div className="table-wrap"><table className="tbl"><thead><tr><th>True</th><th>Predicted as</th><th className="n">Count</th></tr></thead>
              <tbody>{c.error_analysis.top_confusions.map((x: any, i: number) => <tr key={i}><td>{SENT_LABEL[x.true]}</td><td>{SENT_LABEL[x.predicted]}</td><td className="n">{x.count}</td></tr>)}</tbody></table></div>
          </div>
          <div className="table-wrap" style={{ marginTop: 12 }}><table className="tbl"><thead><tr><th>Misclassified review (test set)</th><th>Language</th><th>True</th><th>Predicted</th><th className="n">Conf.</th></tr></thead>
            <tbody>{c.error_analysis.examples.map((x: any, i: number) => <tr key={i}><td className="clip" dir="auto">{x.text}</td><td>{LANG_LABEL[x.language]}</td><td>{SENT_LABEL[x.true]}</td><td>{SENT_LABEL[x.predicted]}</td><td className="n">{pct(x.confidence, 0)}</td></tr>)}</tbody></table></div>
          <ul className="small" style={{ marginTop: 12, paddingLeft: 20 }}>{c.error_analysis.notes.map((n: string, i: number) => <li key={i}>{n}</li>)}</ul>
        </div>
      </div>
    </>
  )
}
