import io, json
import pandas as pd  # pyright: ignore[reportMissingModuleSource]
import unittest
from src.core import (read_csv_bytes, load_upload, classify, normalize_missing,
                      profile, impute, clean, verify, json_safe, eda_html, imputation_methods)

def test_csv_load_and_malformed():
    df=read_csv_bytes(b"a;b\n1;2\n3;4\n")
    assert df.shape==(2,2)
    with unittest.TestCase().assertRaises(ValueError): read_csv_bytes(b"")

def test_excel_sheets():
    b=io.BytesIO()
    with pd.ExcelWriter(b,engine="openpyxl") as w:
        pd.DataFrame({"x":[1]}).to_excel(w,sheet_name="one",index=False)
        pd.DataFrame({"y":[2]}).to_excel(w,sheet_name="two",index=False)
    assert set(load_upload("book.xlsx",b.getvalue()))=={"one","two"}

def test_types_missing_profile_and_duplicate_stats():
    df=pd.DataFrame({"n":[1.,2.,None,2.],"c":["A","B","NA","A"],"b":[True,False,True,True]})
    assert classify(df.n)=="Numerical" and classify(df.b)=="Boolean"
    norm=normalize_missing(df,["NA"]); p=profile(norm)
    assert p["missing_cells"]==2 and p["duplicate_rows"]==0
    assert p["numerical"][0]["median"]==2

def test_cleaning_and_original_values():
    df=pd.DataFrame({"x":[1.,None,3.],"c":["a ","b","b"]})
    out,audit=clean(df,{"x":{"method":"Median"}},{"trim_columns":{"c":True}})
    assert out.x.tolist()==[1.,2.,3.] and out.c.iloc[0]=="a"
    assert pd.isna(df.x.iloc[1])
    assert verify(df,out).iloc[1].missing_cells==0

def test_duplicate_removal_by_key_is_explicit():
    df=pd.DataFrame({"id":[1,1,2],"v":["old","new","x"]})
    unchanged,_=clean(df,{}, {})
    removed,audit=clean(df,{}, {"drop_duplicates":True,"duplicate_columns":["id"],"keep":"last"})
    assert len(unchanged)==3 and removed.v.tolist()==["new","x"]
    assert audit[0]["removed"]==1

def test_imputation_edge_empty_column():
    df=pd.DataFrame({"x":[None,None],"y":[1,2]})
    out,a=impute(df,{"x":{"method":"Mode"}})
    assert out.x.isna().all() and a[0]["filled"]==0

def test_constant_imputation_numeric_column():
    df=pd.DataFrame({"x":[1.,None,3.]})
    out,_=impute(df,{"x":{"method":"Constant","value":"missing"}})
    assert out.x.tolist()==[1.0,"missing",3.0]

def test_date_columns_load_as_timestamps_and_impute():
    frame=read_csv_bytes(b"date,value\n2025-04-18,1\n,2\n2025-04-20,3\n")
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])
    filled,_=impute(frame,{"date":{"method":"Interpolate (date/time)"}})
    assert filled["date"].iloc[1]==pd.Timestamp("2025-04-19")
    assert pd.api.types.is_datetime64_any_dtype(filled["date"])

def test_date_constant_imputation():
    frame=pd.DataFrame({"date":pd.to_datetime(["2025-01-01",None])})
    filled,_=impute(frame,{"date":{"method":"Constant","value":"2025-01-03 12:00:00"}})
    assert filled.date.iloc[1]==pd.Timestamp("2025-01-03 12:00:00")
    assert pd.api.types.is_datetime64_any_dtype(filled.date)

def test_imputation_options_follow_column_type():
    numeric=imputation_methods(pd.Series([1.0,None,3.0]))
    category=imputation_methods(pd.Series(["red",None,"blue"]))
    dates=imputation_methods(pd.Series(pd.to_datetime(["2025-01-01",None])))
    assert "Mean" in numeric and "Interpolate" in numeric
    assert "Mean" not in category and "Mode" in category
    assert "Interpolate (date/time)" in dates and "Median" not in dates

def test_verification_change_row_reports_positive_magnitudes():
    before=pd.DataFrame({"x":[1,None,None],"y":["a","b","b"]})
    after=pd.DataFrame({"x":[1,2],"y":["a","b"]})
    change=verify(before,after).query("stage == 'Change'").iloc[0]
    assert change.missing_cells==2
    assert change.duplicate_rows==1
    assert change.rows==1
    assert (change[["rows","columns","missing_cells","duplicate_rows","completeness_pct"]]>=0).all()

def test_blank_and_placeholder_names_count_as_missing_and_can_be_filled():
    frame=pd.DataFrame({"name":["Alice","   ","NA"]})
    assert profile(frame)["missing_cells"]==2
    filled,_=clean(frame,{"_missing_placeholders":["NA"],"name":{"method":"Constant","value":"Unknown"}})
    assert filled.name.tolist()==["Alice","Unknown","Unknown"]
    assert profile(filled)["completeness_pct"]==100.0

def test_reports_json_safe():
    result=profile(pd.DataFrame({"x":[1,None]}))
    encoded=json.dumps(json_safe(result),allow_nan=False)
    assert "missing_cells" in encoded and "<!doctype html>" in eda_html("demo",result).lower()
