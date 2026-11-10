export const BRAND = {
  name: 'Lahja',
  meaning: 'لہجہ — tone of voice',
  tagline: 'Hear the tone in every review — Urdu, Roman Urdu, English or all three at once.',
  author: 'Haniya Imran',
}

export const SENT_COLOR: Record<string, string> = { positive: '#1F9D7A', neutral: '#7C7997', negative: '#D84B61' }
export const SENT_LABEL: Record<string, string> = { positive: 'Positive', neutral: 'Neutral', negative: 'Negative' }
export const LANG_LABEL: Record<string, string> = {
  urdu: 'Urdu', roman_urdu: 'Roman Urdu', english: 'English',
  mixed_roman_english: 'Mixed Roman Urdu + English', mixed_urdu_english: 'Mixed Urdu + English', uncertain: 'Uncertain',
}
export const LANG_COLOR: Record<string, string> = {
  urdu: '#3B36A8', roman_urdu: '#E8A33D', english: '#4F9FD8', mixed_roman_english: '#1F9D7A', mixed_urdu_english: '#B05FB8', uncertain: '#9C99B3',
}
export const pct = (x: number | null | undefined, d = 1) => (x == null ? '—' : `${(x * 100).toFixed(d)}%`)
export const fmt = (x: number | null | undefined) => (x == null ? '—' : x.toLocaleString())
