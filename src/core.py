"""Dataset loading, profiling, cleaning, verification, and export helpers."""
from __future__ import annotations
import csv, io, json, math, re, warnings
from collections import Counter
from datetime import datetime, timezone
from typing import Any
import numpy as np
import pandas as pd

MISSING_DEFAULTS = ["", "N/A", "NA", "null"]

def read_csv_bytes(data: bytes, name: str = "upload.csv") -> pd.DataFrame:
    if not data:
        raise ValueError("The uploaded file is empty.")
    err = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            frame = pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding=enc, on_bad_lines="error")
            return parse_date_columns(frame)
        except (UnicodeDecodeError, pd.errors.ParserError, csv.Error) as exc:
            err = exc
    raise ValueError(f"Could not read {name}: {err}")

def load_upload(name: str, data: bytes) -> dict[str, pd.DataFrame]:
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if ext == "csv": return {name: read_csv_bytes(data, name)}
    if ext in ("xlsx", "xls"):
        try:
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, engine="openpyxl" if ext == "xlsx" else None)
            return {sheet: parse_date_columns(frame) for sheet, frame in sheets.items()}
        except Exception as exc: raise ValueError(f"Could not read workbook {name}: {exc}") from exc
    raise ValueError("Supported formats are CSV, XLSX, and XLS.")

def parse_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert strongly date-like text columns to pandas datetime64 timestamps."""
    out=df.copy()
    for col in out.columns:
        if (pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col])) and classify(out[col]) == "Date/time (inferred)":
            parsed=pd.to_datetime(out[col],errors="coerce")
            if parsed.notna().any(): out[col]=parsed
    return out

def classify(s: pd.Series) -> str:
    d=s.dtype
    if pd.api.types.is_bool_dtype(d): return "Boolean"
    if pd.api.types.is_datetime64_any_dtype(d): return "Date/time"
    if pd.api.types.is_numeric_dtype(d): return "Numerical"
    if pd.api.types.is_object_dtype(d) or pd.api.types.is_string_dtype(d):
        non=s.dropna()
        if len(non) and not pd.api.types.is_numeric_dtype(d):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed=pd.to_datetime(non, errors="coerce")
            if parsed.notna().mean() >= .85: return "Date/time (inferred)"
        if len(non) and non.nunique()/len(non) > .85 and non.nunique() > 10: return "Potential identifier / text"
        if len(non) and non.astype(str).str.len().median() > 60: return "Free text"
        return "Categorical"
    return "Other"

def imputation_methods(s: pd.Series) -> list[str]:
    """Return safe imputation choices for a column's detected type."""
    kind=classify(s)
    common=["Leave unchanged","Mode","Constant","Forward fill","Backward fill"]
    if kind.startswith("Date/time"):
        return common+["Interpolate (date/time)"]
    if kind=="Numerical":
        return ["Leave unchanged","Mean","Median","Mode","Constant","Forward fill","Backward fill","Interpolate"]
    if kind in ("Categorical","Boolean","Free text","Potential identifier / text"):
        return common
    return ["Leave unchanged","Constant","Forward fill","Backward fill"]

def normalize_missing(df: pd.DataFrame, placeholders=None, whitespace=True) -> pd.DataFrame:
    out=df.copy(); markers=MISSING_DEFAULTS if placeholders is None else placeholders
    for c in out.select_dtypes(include=["object", "string"]).columns:
        vals=out[c].astype("string")
        if whitespace: out.loc[vals.str.strip().eq("").fillna(False), c]=pd.NA
        for marker in markers:
            if marker != "": out.loc[vals.str.casefold().eq(str(marker).casefold()).fillna(False), c]=pd.NA
    return out

def profile(df: pd.DataFrame) -> dict[str, Any]:
    # Keep quality metrics consistent with the default missing-value rules used
    # by the UI: nulls, blank/whitespace-only cells, and common NA placeholders.
    df=normalize_missing(df)
    nrows,ncols=df.shape; miss=df.isna().sum(); total=nrows*ncols
    cols=[]; numeric=[]; categorical=[]
    for c in df.columns:
        s=df[c]; typ=classify(s); vc=s.value_counts(dropna=True)
        item={"column":str(c),"type":typ,"rows":nrows,"valid":int(s.notna().sum()),"missing":int(s.isna().sum()),"missing_pct":float(s.isna().mean()*100) if nrows else 0.0,"unique":int(s.nunique(dropna=True)),"whitespace_values":int(s.astype("string").str.contains(r"^\s|\s$",regex=True,na=False).sum()) if typ in ("Categorical","Free text","Potential identifier / text") else 0,"top":str(vc.index[0]) if len(vc) else None,"top_count":int(vc.iloc[0]) if len(vc) else 0}
        cols.append(item)
        if typ=="Numerical":
            q=s.quantile([.25,.75]) if s.notna().any() else pd.Series(dtype=float); iqr=float(q.iloc[1]-q.iloc[0]) if len(q) else 0
            lo=float(q.iloc[0]-1.5*iqr) if len(q) else None; hi=float(q.iloc[1]+1.5*iqr) if len(q) else None
            numeric.append({"column":str(c),"count":int(s.count()),"missing":int(s.isna().sum()),"mean":_num(s.mean()),"median":_num(s.median()),"mode":str(s.mode().iloc[0]) if len(s.mode()) else None,"min":_num(s.min()),"max":_num(s.max()),"std":_num(s.std()),"variance":_num(s.var()),"q1":_num(q.iloc[0]) if len(q) else None,"q3":_num(q.iloc[1]) if len(q) else None,"iqr":iqr,"skew":_num(s.skew()),"kurtosis":_num(s.kurt()),"outliers_iqr":int(((s<lo)|(s>hi)).sum()) if lo is not None else 0,"lower_fence":lo,"upper_fence":hi,"unique":int(s.nunique())})
        elif typ in ("Categorical","Boolean","Free text","Potential identifier / text"):
            categorical.append({"column":str(c),"valid":int(s.count()),"missing":int(s.isna().sum()),"missing_pct":item["missing_pct"],"unique":int(s.nunique()),"top":item["top"],"top_count":item["top_count"],"rare_values_lt_5":int((vc<5).sum()),"whitespace_values":item["whitespace_values"],"frequencies":{str(k):int(v) for k,v in vc.head(100).items()}})
    corr=df.select_dtypes(include=[np.number]).corr(method="pearson")
    pairs=[]
    for i,a in enumerate(corr.columns):
        for b in corr.columns[i+1:]:
            v=corr.loc[a,b]
            if pd.notna(v): pairs.append({"column_a":str(a),"column_b":str(b),"pearson":float(v),"spearman":float(df[[a,b]].corr(method="spearman").iloc[0,1])})
    dups=int(df.duplicated().sum())
    return {"rows":nrows,"columns":ncols,"memory_bytes":int(df.memory_usage(deep=True).sum()),"missing_cells":int(miss.sum()),"missing_by_column":{str(k):int(v) for k,v in miss.items()},"missing_pct_by_column":{str(c):float(df[c].isna().mean()*100) if nrows else 0.0 for c in df.columns},"columns_with_missing":int((miss>0).sum()),"rows_with_missing":int(df.isna().any(axis=1).sum()),"complete_rows_pct":float((~df.isna().any(axis=1)).mean()*100) if nrows else 0.0,"completeness_pct":float(100*(1-miss.sum()/total)) if total else 100.0,"duplicate_rows":dups,"types":{str(c):classify(df[c]) for c in df.columns},"column_profile":cols,"numerical":numeric,"categorical":categorical,"correlation":corr,"correlation_pairs":pairs,"generated_at":datetime.now(timezone.utc).isoformat()}

def _num(v):
    return float(v) if pd.notna(v) and math.isfinite(float(v)) else None

def categorical_associations(df: pd.DataFrame, max_categories=30) -> pd.DataFrame:
    try:
        from scipy.stats import chi2_contingency
    except ImportError: return pd.DataFrame(columns=["column_a","column_b","cramers_v","p_value"])
    cols=[c for c in df.columns if classify(df[c]) in ("Categorical","Boolean") and df[c].nunique(dropna=True)<=max_categories]
    rows=[]
    for i,a in enumerate(cols):
        for b in cols[i+1:]:
            tab=pd.crosstab(df[a],df[b])
            if tab.shape[0]<2 or tab.shape[1]<2: continue
            chi2,p,_,_=chi2_contingency(tab, correction=False); n=tab.to_numpy().sum(); denom=n*min(tab.shape[0]-1,tab.shape[1]-1)
            rows.append({"column_a":str(a),"column_b":str(b),"cramers_v":float(np.sqrt(chi2/denom)) if denom else 0.0,"p_value":float(p),"rows":int(n)})
    return pd.DataFrame(rows)

def impute(df: pd.DataFrame, config: dict[str,dict]) -> tuple[pd.DataFrame,list[dict]]:
    placeholders=config.get("_missing_placeholders",MISSING_DEFAULTS)
    out=normalize_missing(df,placeholders); audit=[]
    for col,cfg in config.items():
        if col=="_missing_placeholders": continue
        if col not in out: continue
        method=cfg.get("method","Leave unchanged"); before=int(out[col].isna().sum()); value=cfg.get("value","")
        is_date = pd.api.types.is_datetime64_any_dtype(out[col]) or classify(out[col]).startswith("Date/time")
        if is_date and not pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col]=pd.to_datetime(out[col],errors="coerce")
            before=int(out[col].isna().sum())
        if method=="Mean" and is_date: raise ValueError(f"Mean imputation is not defined for timestamp column: {col}")
        elif method=="Median" and is_date: raise ValueError(f"Median imputation is not defined for timestamp column: {col}")
        elif method=="Mean": fill=out[col].mean()
        elif method=="Median": fill=out[col].median()
        elif method=="Mode": fill=out[col].mode(dropna=True).iloc[0] if out[col].notna().any() else pd.NA
        elif method=="Constant":
            try: fill=pd.to_datetime(value,errors="raise") if is_date else value
            except (ValueError,TypeError) as exc: raise ValueError(f"Enter a valid timestamp for {col}, such as 2025-04-18 13:45:00.") from exc
        elif method=="Forward fill": out[col]=out[col].ffill(); fill=None
        elif method=="Backward fill": out[col]=out[col].bfill(); fill=None
        elif method in ("Interpolate","Interpolate (date/time)"):
            if is_date:
                timestamps=pd.to_datetime(out[col],errors="coerce")
                try: time_unit=np.datetime_data(timestamps.dtype)[0]
                except TypeError: time_unit="ns"
                ticks=timestamps.astype("int64").astype("float64")
                ticks[timestamps.isna()]=np.nan
                interpolated=ticks.interpolate(limit_area="inside").round()
                out[col]=pd.to_datetime(interpolated,unit=time_unit,errors="coerce")
            elif pd.api.types.is_numeric_dtype(out[col]): out[col]=out[col].interpolate(limit_area="inside")
            else: raise ValueError(f"Interpolation requires a numeric or timestamp column: {col}")
            fill=None
        else: fill=None
        if fill is not None and method not in ("Leave unchanged",):
            if method == "Constant" and is_date:
                out[col]=out[col].fillna(fill)
            elif method == "Constant":
                # User-entered constants may change a numeric column to mixed/text data.
                # Cast deliberately so pandas nullable dtypes do not reject the fill.
                out[col] = out[col].astype("object").fillna(fill)
            else:
                out[col]=out[col].fillna(fill)
        after=int(out[col].isna().sum()); audit.append({"column":str(col),"method":method,"missing_before":before,"missing_after":after,"filled":before-after,"value":str(fill) if fill is not None else ""})
    return out,audit

def clean(df: pd.DataFrame, config: dict, operations: dict | None=None) -> tuple[pd.DataFrame,list[dict]]:
    out,audit=impute(df,config); ops=operations or {}
    if ops.get("drop_duplicates"):
        subset=ops.get("duplicate_columns") or None; keep=ops.get("keep","first"); n=int(out.duplicated(subset=subset,keep=keep).sum()); out=out.drop_duplicates(subset=subset,keep=keep); audit.append({"operation":"remove_duplicates","removed":n,"subset":subset,"keep":keep})
    for col,rule in ops.get("trim_columns",{}).items():
        if col in out and (pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col])):
            n=int(out[col].astype("string").ne(out[col].astype("string").str.strip()).sum()); out[col]=out[col].str.strip(); audit.append({"operation":"trim_whitespace","column":col,"changed":n})
    for col,dtype in ops.get("conversions",{}).items():
        if col in out:
            converted=pd.to_numeric(out[col],errors="coerce") if dtype=="Numeric" else pd.to_datetime(out[col],errors="coerce")
            audit.append({"operation":"convert_type","column":col,"target":dtype,"new_missing":int(converted.isna().sum()-out[col].isna().sum())}); out[col]=converted
    return out,audit

def verify(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    before_profile=profile(before); after_profile=profile(after)
    for label,df in (("Before",before),("After",after)):
        p=profile(df); rows.append({"stage":label,"rows":p["rows"],"columns":p["columns"],"missing_cells":p["missing_cells"],"duplicate_rows":p["duplicate_rows"],"completeness_pct":p["completeness_pct"]})
    # The Change row reports how many units changed, independent of direction.
    # This keeps the summary useful as an affected-count view (e.g. 8 missing
    # cells filled is shown as 8) while Before and After retain the direction.
    rows.append({"stage":"Change","rows":abs(len(after)-len(before)),"columns":abs(after.shape[1]-before.shape[1]),"missing_cells":abs(after_profile["missing_cells"]-before_profile["missing_cells"]),"duplicate_rows":abs(after_profile["duplicate_rows"]-before_profile["duplicate_rows"]),"completeness_pct":abs(after_profile["completeness_pct"]-before_profile["completeness_pct"])})
    return pd.DataFrame(rows)

def safe_name(name: str) -> str: return re.sub(r"[^A-Za-z0-9._-]+","_",name).strip("._") or "dataset"
def json_safe(value):
    if isinstance(value,pd.DataFrame): return json_safe(value.to_dict(orient="index"))
    if isinstance(value,(np.integer,)): return int(value)
    if isinstance(value,(np.floating,)): return _num(value)
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,dict): return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_safe(v) for v in value]
    if isinstance(value,(float,np.floating)) and not math.isfinite(float(value)): return None
    if value is pd.NA or value is pd.NaT: return None
    return value

def eda_html(name: str, result: dict) -> str:
    tables="".join(f"<h2>{title}</h2>{pd.DataFrame(result[key]).to_html(index=False,escape=True)}" for title,key in (("Column profile","column_profile"),("Numerical statistics","numerical"),("Categorical statistics","categorical"),("Correlation pairs","correlation_pairs")))
    return f"<!doctype html><meta charset='utf-8'><title>EDA - {name}</title><style>body{{font:15px system-ui;max-width:1100px;margin:2rem auto;color:#182230}}table{{border-collapse:collapse}}td,th{{padding:.4rem;border:1px solid #ccd}}</style><h1>EDA report: {name}</h1><p>Generated {result['generated_at']}. Completeness: {result['completeness_pct']:.2f}%.</p><pre>{json.dumps({k:result[k] for k in ('rows','columns','memory_bytes','missing_cells','duplicate_rows','types')},indent=2)}</pre>{tables}"
