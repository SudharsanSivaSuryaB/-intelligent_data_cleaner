from __future__ import annotations
import io, json, zipfile
import pandas as pd
import streamlit as st
import plotly.express as px
from src.core import (load_upload, profile, normalize_missing, categorical_associations, classify, imputation_methods,
                      clean, verify, safe_name, json_safe, eda_html)

st.set_page_config(page_title="Intelligent Data Cleaner", page_icon="🧹", layout="wide")
st.markdown("""<style>
    [data-testid="stSidebar"] { border-right: 1px solid rgba(120,130,150,.18); }
    [data-testid="stMetric"] { background: linear-gradient(145deg, rgba(70,110,170,.08), rgba(70,110,170,.02)); padding: 14px 16px; border: 1px solid rgba(120,130,150,.15); border-radius: 14px; }
    div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
    .block-container { padding-top: 2rem; }
</style>""", unsafe_allow_html=True)
st.title("🧹 Intelligent Data Cleaner")
st.caption("A workspace for profiling, exploring, cleaning, verifying, and exporting your data.")
for key, default in {"datasets":{},"cleaned":{},"history":{},"eda":{},"verification":{},"configs":{}}.items():
    if key not in st.session_state: st.session_state[key]=default
if st.session_state.pop("navigate_to_dashboard",False):
    st.session_state.nav_group="Workspace"
    st.session_state.nav_page="Dashboard"
if "nav_group" not in st.session_state: st.session_state.nav_group="Workspace"
if "nav_page" not in st.session_state: st.session_state.nav_page="Dashboard"

navigation={
    "Workspace":["Dashboard"],
    "Data": ["Upload & Dataset Selection","Data Preview","Data Quality Analysis"],
    "Explore": ["Exploratory Data Analysis","Relationships & Correlations"],
    "Clean & Verify": ["Imputation Methods","Data Cleaning","Verification","Cleaning History"],
    "Export & Help": ["Reports & Export","About & Methodology"],
}
page_icons={"Dashboard":"⌂","Upload & Dataset Selection":"＋","Data Preview":"▤","Data Quality Analysis":"◉","Exploratory Data Analysis":"▥","Relationships & Correlations":"⇄","Imputation Methods":"≈","Data Cleaning":"✦","Verification":"✓","Reports & Export":"⇩","Cleaning History":"◷","About & Methodology":"ⓘ"}
def _sync_navigation():
    group=st.session_state.get("nav_group","Workspace")
    if st.session_state.get("nav_page") not in navigation[group]:
        st.session_state.nav_page=navigation[group][0]
st.sidebar.markdown("### Workspace")
st.sidebar.selectbox("Section",list(navigation),key="nav_group",on_change=_sync_navigation,label_visibility="collapsed")
group=st.session_state.get("nav_group","Workspace")
st.sidebar.selectbox("Page",navigation[group],key="nav_page",format_func=lambda item:f"{page_icons.get(item,'')}  {item}",label_visibility="collapsed")
page=st.session_state.nav_page
if st.sidebar.button("Clear all uploaded data",type="secondary"):
    for k in ("datasets","cleaned","history","eda","verification","configs"): st.session_state[k]={}
    st.rerun()
if not st.session_state.datasets: st.sidebar.info("Upload CSV or Excel files to begin.")

def names(): return list(st.session_state.datasets)
def choose(label="Dataset"):
    return st.selectbox(label,names()) if names() else None
def data_for(name,version="Original"):
    return st.session_state.cleaned.get(name,st.session_state.datasets[name]) if version=="Cleaned" else st.session_state.datasets[name]
def metric_row(name, df):
    p=profile(df); a,b,c,d=st.columns(4)
    a.metric("Rows",f"{p['rows']:,}"); b.metric("Columns",p["columns"]); c.metric("Missing cells",f"{p['missing_cells']:,}"); d.metric("Completeness",f"{p['completeness_pct']:.1f}%")
def csv_button(label, df, filename): st.download_button(label,df.to_csv(index=False).encode("utf-8-sig"),filename,"text/csv")
def classify_label(s): return classify(s)

if page=="Upload & Dataset Selection":
    st.header(page); st.write("Each CSV is a dataset. Each workbook worksheet is a separate dataset.")
    uploads=st.file_uploader("Upload CSV, XLSX, or XLS files",type=["csv","xlsx","xls"],accept_multiple_files=True)
    if st.button("Add uploaded files",disabled=not uploads):
        ok=0
        for f in uploads:
            try:
                loaded=load_upload(f.name,f.getvalue())
                for sheet,df in loaded.items():
                    key=f.name if len(loaded)==1 else f"{f.name} — {sheet}"
                    if df.columns.duplicated().any(): df.columns=[f"{c}_{i}" if n else c for i,(c,n) in enumerate(zip(df.columns,df.columns.duplicated()))]
                    st.session_state.datasets[key]=df.copy(); st.session_state.configs[key]={}; ok+=1
            except Exception as e: st.error(f"{f.name}: {e}")
        if ok: st.success(f"Loaded {ok} dataset(s).")
    if names():
        rows=[]
        for n,df in st.session_state.datasets.items():
            p=profile(df); rows.append({"Dataset":n,"Source":n.split(" — ")[0],"Rows":len(df),"Columns":df.shape[1],"Missing":p["missing_cells"],"Duplicates":p["duplicate_rows"],"Cleaned":n in st.session_state.cleaned,"Verified":n in st.session_state.verification})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        st.multiselect("Selected datasets",names(),default=names(),key="selected_datasets")

elif page=="Dashboard":
    if st.session_state.pop("dashboard_notice",False): st.success("Cleaning applied. Dashboard statistics now reflect the cleaned data.")
    st.header(page); st.write("Upload → Preview → Analyze → EDA → Configure Cleaning → Apply Cleaning → Verify → Export")
    if names():
        pds=[st.session_state.datasets[n] for n in st.session_state.get("selected_datasets",names()) if n in names()]
        total_rows=sum(map(len,pds)); total_cols=sum(df.shape[1] for df in pds)
        num=sum(sum(t=="Numerical" for t in profile(df)["types"].values()) for df in pds)
        cat=sum(sum(t in ("Categorical","Free text","Potential identifier / text") for t in profile(df)["types"].values()) for df in pds)
        dates=sum(sum(t.startswith("Date/time") for t in profile(df)["types"].values()) for df in pds)
        bools=sum(sum(t=="Boolean" for t in profile(df)["types"].values()) for df in pds)
        metrics=[("Files",len(set(n.split(" — ")[0] for n in names())),"Datasets / worksheets",len(pds)),("Rows",total_rows,"Columns",total_cols),("Numerical",num,"Categorical",cat),("Date/time",dates,"Boolean",bools),("Missing cells",sum(profile(d)["missing_cells"] for d in pds),"Duplicate rows",sum(profile(d)["duplicate_rows"] for d in pds)),("Cleaned datasets",len(st.session_state.cleaned),"Verified datasets",len(st.session_state.verification))]
        for i in range(0,len(metrics),2):
            a,b,c,d=st.columns(4); a.metric(metrics[i][0],metrics[i][1]); b.metric(metrics[i][2],metrics[i][3]); c.metric(metrics[i+1][0],metrics[i+1][1]); d.metric(metrics[i+1][2],metrics[i+1][3])
        st.dataframe(pd.DataFrame([{"Dataset":n,"Rows":len(df),"Columns":df.shape[1],"Missing":profile(df)["missing_cells"],"Duplicates":profile(df)["duplicate_rows"],"EDA":"Run" if n in st.session_state.eda else "Not run","Cleaning":"Applied" if n in st.session_state.cleaned else "Pending","Verification":"Run" if n in st.session_state.verification else "Pending"} for n,df in st.session_state.datasets.items()]),use_container_width=True,hide_index=True)
        cleaned_selected={n:st.session_state.cleaned[n] for n in st.session_state.get("selected_datasets",names()) if n in st.session_state.cleaned}
        if cleaned_selected:
            st.divider(); st.subheader("Cleaned dataset statistics")
            cleaned_rows=sum(len(df) for df in cleaned_selected.values())
            cleaned_cols=sum(df.shape[1] for df in cleaned_selected.values())
            cleaned_missing=sum(profile(df)["missing_cells"] for df in cleaned_selected.values())
            cleaned_duplicates=sum(profile(df)["duplicate_rows"] for df in cleaned_selected.values())
            cleaned_cells=sum(df.shape[0]*df.shape[1] for df in cleaned_selected.values())
            cleaned_completeness=100*(1-cleaned_missing/cleaned_cells) if cleaned_cells else 100.0
            a,b,c,d,e=st.columns(5)
            a.metric("Cleaned datasets",len(cleaned_selected)); b.metric("Rows",f"{cleaned_rows:,}"); c.metric("Columns",cleaned_cols); d.metric("Missing cells",f"{cleaned_missing:,}"); e.metric("Completeness",f"{cleaned_completeness:.1f}%")
            st.dataframe(pd.DataFrame([{"Dataset":n,"Rows":len(df),"Columns":df.shape[1],"Missing cells":profile(df)["missing_cells"],"Duplicates":profile(df)["duplicate_rows"],"Completeness %":profile(df)["completeness_pct"],"Rows removed":max(0,len(st.session_state.datasets[n])-len(df)),"Missing values filled":max(0,profile(st.session_state.datasets[n])["missing_cells"]-profile(df)["missing_cells"])} for n,df in cleaned_selected.items()]),use_container_width=True,hide_index=True)
    else: st.info("Upload datasets to see live quality metrics.")

elif page=="Data Preview":
    st.header(page); n=choose()
    if n:
        version=st.radio("Version",["Original","Cleaned"],horizontal=True,disabled=n not in st.session_state.cleaned); df=data_for(n,version)
        rows=st.slider("Preview rows",min_value=5,max_value=500,value=min(50,max(5,len(df) or 5)))
        cols=st.multiselect("Columns",list(df.columns),default=list(df.columns)); st.caption(f"Showing {min(rows,len(df)):,} of {len(df):,} rows.")
        st.dataframe(df.loc[:,cols].head(rows),use_container_width=True)
        st.dataframe(pd.DataFrame([{"Column":c,"Type":str(df[c].dtype),"Missing":int(df[c].isna().sum()),"Unique":int(df[c].nunique(dropna=True))} for c in df]),use_container_width=True,hide_index=True)
        if n in st.session_state.cleaned:
            st.subheader("Before and after"); st.dataframe(pd.concat({"Original":st.session_state.datasets[n].head(rows),"Cleaned":st.session_state.cleaned[n].head(rows)},axis=1),use_container_width=True)

elif page in ("Data Quality Analysis","Exploratory Data Analysis","Relationships & Correlations"):
    st.header(page); n=choose()
    if n:
        version="Original"
        if page=="Exploratory Data Analysis" and n in st.session_state.cleaned: version=st.radio("Analyze version",["Original","Cleaned"],horizontal=True)
        df=data_for(n,version); placeholders=st.text_input("Missing value placeholders (comma separated)","N/A,NA,null")
        if st.button("Analyze data" if page!="Exploratory Data Analysis" else "Run EDA"):
            analyzed=normalize_missing(df,[x.strip() for x in placeholders.split(",") if x.strip()]); result=profile(analyzed)
            assoc=categorical_associations(analyzed)
            result["categorical_associations"]=assoc.to_dict(orient="records")
            result["observations"]=[f"{c['column']} has {c['missing_pct']:.1f}% missing values." for c in result["column_profile"] if c["missing"]]
            result["observations"] += [f"{x['column']} has {x['outliers_iqr']} potential IQR outliers." for x in result["numerical"] if x["outliers_iqr"]]
            st.session_state.eda[n]=result; st.session_state.eda[n+"__data"]=analyzed
        result=st.session_state.eda.get(n)
        if page=="Relationships & Correlations":
            nums=df.select_dtypes(include="number").columns.tolist()
            if len(nums)>1:
                selected=st.multiselect("Numerical columns",nums,default=nums[:20]); method=st.selectbox("Correlation method",["pearson","spearman","kendall"])
                corr=df[selected].corr(method=method) if len(selected)>1 else pd.DataFrame()
                if not corr.empty: st.plotly_chart(px.imshow(corr,text_auto=".2f",color_continuous_scale="RdBu_r",zmin=-1,zmax=1),use_container_width=True); st.dataframe(corr,use_container_width=True)
                assoc=categorical_associations(df); st.subheader("Categorical association (Cramér’s V)"); st.dataframe(assoc,use_container_width=True,hide_index=True)
            else: st.info("At least two numerical columns are needed for correlation analysis.")
        elif result:
            metric_row(n,st.session_state.eda[n+"__data"]); st.dataframe(pd.DataFrame(result["column_profile"]),use_container_width=True,hide_index=True)
            tabs=st.tabs(["Numerical","Categorical","Missingness","Duplicates / outliers","Observations"])
            with tabs[0]: st.dataframe(pd.DataFrame(result["numerical"]),use_container_width=True,hide_index=True)
            with tabs[1]:
                st.dataframe(pd.DataFrame(result["categorical"]),use_container_width=True,hide_index=True)
                if result["categorical"]:
                    c=st.selectbox("Category distribution",[x["column"] for x in result["categorical"]]); st.plotly_chart(px.bar(st.session_state.eda[n+"__data"][c].value_counts().head(30).rename_axis(c).reset_index(name="count"),x=c,y="count"),use_container_width=True)
            with tabs[2]:
                st.dataframe(pd.DataFrame({"Column":result["missing_by_column"].keys(),"Missing":result["missing_by_column"].values(),"Missing %":result["missing_pct_by_column"].values()}),use_container_width=True,hide_index=True)
                st.caption(f"{result['rows_with_missing']} rows contain missing values; {result['complete_rows_pct']:.1f}% of rows are complete.")
                mask=st.session_state.eda[n+"__data"].isna().astype(int).head(500); st.plotly_chart(px.imshow(mask,aspect="auto",color_continuous_scale="Reds"),use_container_width=True)
            with tabs[3]:
                x=st.session_state.eda[n+"__data"]; st.write(f"Exact duplicate rows: {x.duplicated().sum()}"); st.dataframe(x[x.duplicated(keep=False)].head(100),use_container_width=True)
                st.dataframe(pd.DataFrame(result["numerical"])[["column","outliers_iqr","lower_fence","upper_fence"]] if result["numerical"] else pd.DataFrame(),use_container_width=True,hide_index=True)
                st.dataframe(pd.DataFrame(result["correlation_pairs"]),use_container_width=True,hide_index=True)
            with tabs[4]: st.write("Completeness measures non-missing cells only; it does not measure correctness."); st.write(result["observations"])
            st.subheader("Independent EDA downloads")
            stem=safe_name(n)
            st.download_button("Full EDA HTML",eda_html(n,result),f"{stem}_eda.html","text/html")
            st.download_button("EDA summary JSON",json.dumps(json_safe(result),indent=2,allow_nan=False),f"{stem}_eda.json","application/json")
            for key,label in [("column_profile","Column profile"),("numerical","Numerical statistics"),("categorical","Categorical statistics"),("correlation_pairs","Correlation pairs"),("categorical_associations","Categorical associations")]:
                csv_button(f"{label} CSV",pd.DataFrame(result[key]),f"{stem}_{key}.csv")
            csv_button("Missing report CSV",pd.DataFrame([{"column":c,"missing":v,"missing_pct":result["missing_pct_by_column"][c]} for c,v in result["missing_by_column"].items()]),f"{stem}_missing.csv")
            csv_button("Duplicate rows CSV",st.session_state.eda[n+"__data"][st.session_state.eda[n+"__data"].duplicated(keep=False)],f"{stem}_duplicates.csv")

elif page=="Imputation Methods":
    st.header(page); n=choose()
    if n:
        df=st.session_state.datasets[n]
        marker_text=st.text_input("Treat these text values as missing",value="N/A, NA, null",help="Whitespace-only cells are always treated as missing. Values such as 0 and unknown are preserved.",key=f"impute_markers_{n}")
        markers=[x.strip() for x in marker_text.split(",") if x.strip()]
        imputation_data=normalize_missing(df,markers)
        miss=[c for c in imputation_data if imputation_data[c].isna().any()]
        if not miss: st.success("No missing values detected in the original dataset.")
        cfg={"_missing_placeholders":markers}
        for c in miss:
            kind=classify(imputation_data[c]); is_date=kind.startswith("Date/time")
            methods=imputation_methods(imputation_data[c])
            type_label="Timestamp" if is_date else "Numeric" if kind=="Numerical" else "Categorical" if kind in ("Categorical","Boolean","Free text","Potential identifier / text") else kind
            label=f"{c} · {type_label}"
            a,b=st.columns([2,1]); method=a.selectbox(label,methods,key=f"imp_{n}_{c}")
            value=b.text_input("Timestamp constant",placeholder="2025-04-18 13:45:00",help="Enter a parseable date or timestamp.",key=f"val_{n}_{c}") if method=="Constant" and is_date else b.text_input("Constant value",key=f"val_{n}_{c}") if method=="Constant" else ""
            cfg[c]={"method":method,"value":value}
        st.session_state.configs[n]=cfg
        st.info("Date columns are stored as pandas timestamps. Date interpolation fills gaps between surrounding timestamps; forward/backward fill follows row order. Mean and median are not offered for dates.")

elif page=="Data Cleaning":
    st.header(page); n=choose()
    if n:
        df=st.session_state.datasets[n]; st.caption("Cleaning always starts from the preserved original dataset; applying again replaces the previous cleaned version.")
        cfg=st.session_state.configs.get(n,{})
        dup=st.checkbox("Remove duplicate rows when applying",key=f"dup_{n}"); dupcols=st.multiselect("Duplicate key columns",list(df.columns),key=f"dupcols_{n}"); keep=st.radio("Keep duplicate",["first","last"],horizontal=True,key=f"keep_{n}")
        trim=st.multiselect("Trim leading/trailing whitespace",list(df.select_dtypes(include=["object","string"]).columns),key=f"trim_{n}")
        conversions={}
        for c in st.multiselect("Convert columns",list(df.columns),key=f"conv_{n}"):
            conversions[c]=st.selectbox(f"Convert {c} to",["Numeric","Datetime"],key=f"type_{n}_{c}")
        if dup:
            duplicate_subset=dupcols or None
            proposed=df[df.duplicated(subset=duplicate_subset,keep=keep)]
            st.caption(f"Review: {len(proposed):,} row(s) would be removed using {', '.join(dupcols) if dupcols else 'all columns'}; keeping the {keep} record.")
            if len(proposed): st.dataframe(proposed.head(100),use_container_width=True,hide_index=True)
        if st.button("Apply cleaning",type="primary"):
            try:
                out,audit=clean(df,cfg,{"drop_duplicates":dup,"duplicate_columns":dupcols,"keep":keep,"trim_columns":{c:True for c in trim},"conversions":conversions})
                st.session_state.cleaned[n]=out; st.session_state.history.setdefault(n,[]).append({"at":pd.Timestamp.utcnow().isoformat(),"audit":audit,"rows_before":len(df),"rows_after":len(out)})
                st.session_state.verification[n]=verify(df,out)
                st.session_state.dashboard_notice=True
                st.session_state.navigate_to_dashboard=True
                st.rerun()
            except Exception as e: st.error(f"Cleaning failed: {e}")
        if n in st.session_state.cleaned:
            cleaned=st.session_state.cleaned[n]
            a,b,c=st.columns(3); a.metric("Rows after cleaning",f"{len(cleaned):,}",delta=f"{len(cleaned)-len(df):,}"); b.metric("Missing cells",f"{int(cleaned.isna().sum().sum()):,}",delta=f"{int(cleaned.isna().sum().sum()-df.isna().sum().sum()):,}"); c.metric("Exact duplicates",f"{int(cleaned.duplicated().sum()):,}",delta=f"{int(cleaned.duplicated().sum()-df.duplicated().sum()):,}")
            st.subheader("Cleaned preview")
            st.dataframe(cleaned.head(50),use_container_width=True,hide_index=True)
            st.subheader("Verification summary")
            st.dataframe(verify(df,cleaned),use_container_width=True,hide_index=True)
            csv_button("Download cleaned CSV",cleaned,f"{safe_name(n)}_cleaned.csv")
        if n in st.session_state.cleaned and st.button("Reset to original",key=f"reset_{n}"):
            del st.session_state.cleaned[n]; st.session_state.verification.pop(n,None); st.success("Working data restored from original.")

elif page=="Verification":
    st.header(page); n=choose()
    if n:
        if n in st.session_state.cleaned:
            st.session_state.verification[n]=verify(st.session_state.datasets[n],st.session_state.cleaned[n]); st.dataframe(st.session_state.verification[n],use_container_width=True,hide_index=True)
            st.write("Unresolved missing cells are reported in the after counts; verification does not label them clean automatically.")
        else: st.info("Apply cleaning to generate before-and-after verification.")

elif page=="Reports & Export":
    st.header(page)
    if names():
        n=choose(); df=st.session_state.cleaned.get(n,st.session_state.datasets[n]); csv_button("Download cleaned dataset CSV",df,f"{safe_name(n)}_cleaned.csv")
        if st.session_state.cleaned:
            buf=io.BytesIO()
            with pd.ExcelWriter(buf,engine="openpyxl") as writer:
                for key,frame in st.session_state.cleaned.items(): frame.to_excel(writer,sheet_name=safe_name(key)[:31] or "Data",index=False)
            st.download_button("Download cleaned datasets Excel",buf.getvalue(),"cleaned_datasets.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        if st.session_state.eda and n in st.session_state.eda:
            res=st.session_state.eda[n]; st.download_button("EDA HTML",eda_html(n,res),f"{safe_name(n)}_eda.html","text/html"); st.download_button("EDA JSON",json.dumps(json_safe(res),indent=2,allow_nan=False),f"{safe_name(n)}_eda.json","application/json")
        if n in st.session_state.verification: csv_button("Verification CSV",st.session_state.verification[n],f"{safe_name(n)}_verification.csv")
        if n in st.session_state.history: st.download_button("Cleaning history JSON",json.dumps(json_safe(st.session_state.history[n]),indent=2),f"{safe_name(n)}_history.json","application/json")
        st.download_button("Cleaning configuration JSON",json.dumps(json_safe(st.session_state.configs.get(n,{})),indent=2),f"{safe_name(n)}_config.json","application/json")
        if len(st.session_state.cleaned)>1:
            if st.button("Prepare ZIP of cleaned datasets"):
                buf=io.BytesIO()
                with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
                    for key,frame in st.session_state.cleaned.items(): z.writestr(safe_name(key)+".csv",frame.to_csv(index=False).encode("utf-8-sig"))
                st.download_button("Download cleaned ZIP",buf.getvalue(),"cleaned_datasets.zip","application/zip")
    else: st.info("Upload data first. EDA exports are available before cleaning.")

elif page=="Cleaning History":
    st.header(page)
    for n,items in st.session_state.history.items():
        with st.expander(n):
            st.dataframe(pd.DataFrame([{"timestamp":x["at"],"rows_before":x["rows_before"],"rows_after":x["rows_after"],"operations":json.dumps(x["audit"])} for x in items]),use_container_width=True,hide_index=True)

elif page=="About & Methodology":
    st.header(page)
    st.markdown("""Uploads are processed in this local Streamlit session and are not intentionally sent to an external service. Session state is temporary; export results before the session ends. CSV files become individual datasets; each Excel worksheet is a dataset. XLS support depends on a compatible reader.

**Methods.** Numeric summaries use pandas sample standard deviation and quartiles. IQR outliers lie below Q1−1.5×IQR or above Q3+1.5×IQR; they are possible anomalies, not errors. Correlations are pairwise Pearson, Spearman, or Kendall for numeric columns. Cramér’s V summarizes categorical association and does not imply causation. Completeness is 100×(1−missing cells/total cells) and is not a correctness score. Missing placeholders are configurable and values such as 0, False, and “unknown” are not marked missing by default.

Mean imputation is sensitive to outliers; median can suit skewed data; mode can fill dates with the most frequent timestamp; date interpolation fills only gaps between surrounding timestamps. Forward/backward fill requires meaningful row ordering. KNN is not currently exposed in this interface. Original data is retained separately. Data starts in memory and large files are bounded by available RAM and Streamlit upload configuration. Formula cells are read as values; macros are not executed.""")

