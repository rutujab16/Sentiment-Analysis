# merge_and_prepare.py
"""
Robust merge script: tries multiple encodings and delimiter sniffing so
files with non-UTF8 encodings don't crash the merge.
Saves combined_dataset.csv with standardized labels.
"""
import os, glob, csv
import pandas as pd
import numpy as np
import re
from io import StringIO

DATA_FOLDER = "datasets"
OUT_CSV = "combined_dataset.csv"

def try_read_csv(path, encodings=None):
    """Try to read CSV using several encodings and delimiter sniffing."""
    if encodings is None:
        encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1', 'iso-8859-1']
    last_exc = None
    for enc in encodings:
        try:
            # Use python engine and sep=None to allow delimiter sniffing
            df = pd.read_csv(path, encoding=enc, engine='python', sep=None, header=0, on_bad_lines='skip', low_memory=False)
            print(f"  -> Read {os.path.basename(path)} with encoding={enc}, shape={df.shape}")
            return df, enc
        except Exception as e:
            last_exc = e
            # continue trying other encodings
    # last resort: open in binary, decode as latin1 and let pandas read from string
    try:
        with open(path, 'rb') as fh:
            raw = fh.read()
        text = raw.decode('latin-1', errors='replace')
        # Let pandas parse this string (sniffing separators)
        df = pd.read_csv(StringIO(text), sep=None, engine='python', header=0, on_bad_lines='skip')
        print(f"  -> Read {os.path.basename(path)} by forcing latin-1 with replacement.")
        return df, 'latin-1-replaced'
    except Exception as e:
        print("Failed reading file with all encodings. Last error:", last_exc)
        raise last_exc

def normalize_label_series(s):
    s_nonnull = s.dropna().astype(str).str.strip().str.lower()
    unique_vals = sorted(list(set(s_nonnull.tolist())))
    mapped = {}
    # numeric detection
    numeric = True
    as_ints = []
    for v in unique_vals:
        try:
            as_ints.append(int(float(v)))
        except Exception:
            numeric = False
            break
    if numeric and len(unique_vals) > 1:
        mn, mx = min(as_ints), max(as_ints)
        if mn >= 1 and mx <= 5:
            def map_star(x):
                ix = int(float(x))
                if ix <= 2: return 'negative'
                if ix == 3: return 'neutral'
                return 'positive'
            return {v: map_star(v) for v in unique_vals}
        if set(as_ints) <= {0,1}:
            return {v: ('negative' if int(float(v)) == 0 else 'positive') for v in unique_vals}
        med = np.median(as_ints)
        return {v: ('negative' if int(float(v)) < med else ('neutral' if int(float(v)) == med else 'positive')) for v in unique_vals}
    # textual heuristics
    for v in unique_vals:
        lv = str(v).lower()
        if lv in {'neg','negative','bad','0','-1','n','false'}:
            mapped[v] = 'negative'
        elif lv in {'pos','positive','good','2','+1','p','true'}:
            mapped[v] = 'positive'
        elif 'neutral' in lv or lv in {'neu','ntrl','3','meh','okay','ok','average'}:
            mapped[v] = 'neutral'
        else:
            if any(k in lv for k in ['hate','terrible','awful','worst','bad','boring','disappoint','sucks']):
                mapped[v] = 'negative'
            elif any(k in lv for k in ['love','great','excellent','amazing','best','fun','good','brilliant','loved']):
                mapped[v] = 'positive'
            elif any(k in lv for k in ['ok','fine','average','so-so','so so','meh']):
                mapped[v] = 'neutral'
            else:
                mapped[v] = None
    none_vals = [k for k, vv in mapped.items() if vv is None]
    if none_vals and len(unique_vals) == 2:
        uniq_sorted = sorted(unique_vals)
        for i, v in enumerate(uniq_sorted):
            mapped[v] = 'negative' if i == 0 else 'positive'
        return mapped
    for k in mapped:
        if mapped[k] is None:
            mapped[k] = 'neutral'
    return mapped

def find_text_label_columns(df):
    text_cols = [c for c in df.columns if c.lower() in ['text','review','sentence','comment','content','review_text','reviewtext']]
    label_cols = [c for c in df.columns if c.lower() in ['label','labels','sentiment','rating','star','stars','class']]
    if text_cols and label_cols:
        return text_cols[0], label_cols[0]
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if obj_cols:
        maybe_text = max(obj_cols, key=lambda c: df[c].astype(str).map(len).median())
        maybe_label = [c for c in df.columns if c != maybe_text][0]
        return maybe_text, maybe_label
    return df.columns[0], df.columns[1]

def main():
    files = sorted(glob.glob(os.path.join(DATA_FOLDER, "*.csv")))
    if not files:
        print("No CSV files found in", DATA_FOLDER)
        return
    parts = []
    summary_maps = {}
    for f in files:
        print("\nLoading", f)
        try:
            df_raw, used_enc = try_read_csv(f)
        except Exception as e:
            print(f"ERROR: Could not read {f}. Skipping file. Error:", e)
            continue
        tcol, lcol = find_text_label_columns(df_raw)
        print(" Text col:", tcol, "Label col:", lcol)
        df = df_raw[[tcol, lcol]].rename(columns={tcol: 'text', lcol: 'label_raw'})
        df['label_raw'] = df['label_raw'].astype(str).str.strip()
        mapping = normalize_label_series(df['label_raw'])
        print(" Mapping sample:", dict(list(mapping.items())[:10]))
        df['label'] = df['label_raw'].map(mapping)
        unmapped = df['label'].isna().sum()
        if unmapped:
            print(f"  {unmapped} rows unmapped -> setting to 'neutral'")
            df['label'] = df['label'].fillna('neutral')
        parts.append(df[['text','label']])
        summary_maps[os.path.basename(f)] = {'encoding_used': used_enc, 'mapping_sample': dict(list(mapping.items())[:8])}

    if not parts:
        print("No files loaded successfully. Exiting.")
        return

    combined = pd.concat(parts, ignore_index=True)
    combined['text'] = combined['text'].astype(str)
    combined = combined[combined['text'].str.strip() != ""].reset_index(drop=True)

    label_to_id = {'negative': 0, 'neutral': 1, 'positive': 2}
    combined['label'] = combined['label'].str.lower().str.strip()
    combined.loc[~combined['label'].isin(label_to_id.keys()), 'label'] = 'neutral'
    combined['label_id'] = combined['label'].map(label_to_id)

    print("\nCombined size:", len(combined))
    print("Label counts:\n", combined['label'].value_counts())
    combined.to_csv(OUT_CSV, index=False)
    print("\nSaved", OUT_CSV)
    print("Per-file summary (encoding + sample mapping):")
    for k, v in summary_maps.items():
        print(" -", k, "enc:", v['encoding_used'], " mapping-sample:", v['mapping_sample'])

if __name__ == "__main__":
    main()
