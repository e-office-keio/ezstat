"""
APA Style Statistical Analysis Web Application using Streamlit.
(Features: Instant Sidebar Refresh on Queue Add, Scrollable Plot Containers, Japanese Interface)
"""

import os
import io
import importlib
import streamlit as st
import pandas as pd
import numpy as np

# カスタムモジュールの読み込み & 自動リロード
import stats_engine
import apa_excel
importlib.reload(stats_engine)
importlib.reload(apa_excel)

st.set_page_config(
    page_title="ezstat - APAスタイル 統計解析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# セッション状態の初期化
if "analysis_queue" not in st.session_state:
    st.session_state["analysis_queue"] = []
if "results" not in st.session_state:
    st.session_state["results"] = {}
if "sem_latents" not in st.session_state:
    st.session_state["sem_latents"] = []
if "sem_paths" not in st.session_state:
    st.session_state["sem_paths"] = []
if "sem_covs" not in st.session_state:
    st.session_state["sem_covs"] = []

# カスタムCSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1e3a8a;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .main-title span.app-tag {
        font-size: 0.9rem;
        font-weight: 600;
        background-color: #e0e7ff;
        color: #3730a3;
        padding: 4px 10px;
        border-radius: 9999px;
        letter-spacing: normal;
        vertical-align: middle;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        line-height: 1.6;
        margin-bottom: 1.4rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f5f9;
        border-radius: 6px 6px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e3a8a !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)





def load_data_file(uploaded_file, encoding_choice="自動判定 (Auto)", sheet_name=0):
    """CSVまたはExcelファイル (.xlsx, .xls) の読み込み"""
    file_name = uploaded_file.name.lower()
    
    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        return df, f"Excel ({sheet_name})"
    else:  # CSV
        if encoding_choice == "自動判定 (Auto)":
            encodings_to_try = ["utf-8", "cp932", "shift_jis", "euc-jp", "utf-8-sig"]
            for enc in encodings_to_try:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc)
                    return df, enc
                except Exception:
                    continue
            raise ValueError("CSVファイルのエンコーディングを自動判定できませんでした。手動で選択してください。")
        else:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=encoding_choice)
            return df, encoding_choice


def main():
    st.markdown("""
    <div class="main-title">
        ezstat <span class="app-tag">APA 7th Edition Ready</span>
    </div>
    <div class="sub-title">
        Python統計ライブラリ（SciPy / statsmodels / semopy 等）による高精度な分析と、APA第7版スタイルに準拠した図表（Excel / 画像）を出力するWebアプリケーション
    </div>
    """, unsafe_allow_html=True)
    
    # ------------------------------------------------------------------
    # サイドバー：ファイルアップロード & 出力レポート管理
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("📂 データアップロード")
        uploaded_file = st.file_uploader("ファイルを選択 (CSV / Excel)", type=["csv", "xlsx", "xls"])
        
        encoding_choice = "自動判定 (Auto)"
        selected_sheet = 0
        if uploaded_file is not None:
            fname = uploaded_file.name.lower()
            if fname.endswith(".csv"):
                encoding_choice = st.selectbox(
                    "文字コード (CSV Encoding)",
                    ["自動判定 (Auto)", "utf-8", "cp932 (Windows Shift-JIS)", "shift_jis", "utf-8-sig"]
                )
            elif fname.endswith(".xlsx") or fname.endswith(".xls"):
                try:
                    excel_file = pd.ExcelFile(uploaded_file)
                    sheet_names = excel_file.sheet_names
                    if len(sheet_names) > 1:
                        selected_sheet = st.selectbox("読み込むシートを選択:", sheet_names)
                    else:
                        selected_sheet = sheet_names[0]
                except Exception:
                    selected_sheet = 0
                    
        if "df" in st.session_state and st.session_state["df"] is not None:
            curr_df = st.session_state["df"]
            with st.expander("💾 加工済みデータの保存", expanded=False):
                st.caption(f"現在のデータ: {curr_df.shape[0]}行 × {curr_df.shape[1]}列")
                c_s_dl1, c_s_dl2 = st.columns(2)
                with c_s_dl1:
                    s_csv = curr_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                    st.download_button("CSV (.csv)", data=s_csv, file_name="加工済みデータ.csv", mime="text/csv", key="sb_dl_csv", use_container_width=True)
                with c_s_dl2:
                    s_io = io.BytesIO()
                    with pd.ExcelWriter(s_io, engine="openpyxl") as writer:
                        curr_df.to_excel(writer, index=False, sheet_name="加工済みデータ")
                    st.download_button("Excel (.xlsx)", data=s_io.getvalue(), file_name="加工済みデータ.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="sb_dl_xlsx", use_container_width=True)
            
            if st.button("🔄 データをリセットしてトップに戻る", use_container_width=True):
                for k in ["df", "raw_df", "last_uploaded", "results"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()
        
        st.divider()
        st.header("📋 出力レポート管理")
        queue_count = len(st.session_state["analysis_queue"])
        st.info(f"追加済みレポート: **{queue_count}** 件")
        
        if queue_count > 0:
            if st.button("📥 全結果をAPAスタイルExcelで作成", type="primary", use_container_width=True):
                with st.spinner("⏳ APAスタイルのExcelレポートを作成中..."):
                    excel_bytes = apa_excel.build_full_excel_report(st.session_state["analysis_queue"])
                st.toast("🎉 Excelレポートの作成が完了しました！", icon="✅")
                st.download_button(
                    label="💾 Excelファイルを保存 (.xlsx)",
                    data=excel_bytes,
                    file_name="APA_統計解析結果レポート.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            if st.button("🗑 レポートリストをクリア", use_container_width=True):
                st.session_state["analysis_queue"] = []
                st.toast("レポートリストをクリアしました", icon="🧹")
                st.rerun()

    # データフレームの初期ロード処理
    if uploaded_file is None:
        if "df" not in st.session_state:
            # ロゴ画像の読み込み（ローカルファイル・複数パス・GitHub Raw URLへの確実なフォールバック）
            logo_candidates = [
                os.path.join(os.path.dirname(__file__), "Logo.png"),
                os.path.join(os.getcwd(), "Logo.png"),
                "Logo.png",
                os.path.join(os.path.dirname(__file__), "logo.png"),
                "logo.png",
            ]
            logo_src = None
            for p in logo_candidates:
                if os.path.exists(p):
                    logo_src = p
                    break
            if not logo_src:
                logo_src = "https://raw.githubusercontent.com/e-office-keio/ezstat/main/Logo.png"

            try:
                st.image(logo_src, use_container_width=True)
            except Exception:
                try:
                    st.image("https://raw.githubusercontent.com/e-office-keio/ezstat/main/Logo.png", use_container_width=True)
                except Exception:
                    pass

            st.info("👈 サイドバーから分析したいCSVまたはExcelファイル（.xlsx / .xls）をアップロードしてください。")
            with st.expander("💡 演習用サンプルデータで試す", expanded=True):
                st.caption("ファイルをお持ちでない場合、すぐに全機能を試せる実践演習用データ（N=150）をワンクリックで読み込めます。")
                
                c_samp1, c_samp2 = st.columns(2)
                with c_samp1:
                    if st.button("🎲 ランダムな演習用サンプルデータを生成", type="primary", use_container_width=True):
                        with st.spinner("実践演習用サンプルデータを生成中..."):
                            sample_data = stats_engine.generate_sample_dataset(n=150, seed=None)
                            st.session_state["raw_df"] = sample_data.copy()
                            st.session_state["df"] = sample_data.copy()
                            st.session_state["last_uploaded"] = "sample_data_random"
                        st.toast("演習用サンプルデータを読み込みました！", icon="🚀")
                        st.rerun()
                with c_samp2:
                    if st.button("🎯 再現可能な標準サンプルデータを生成 (Seed=42)", use_container_width=True):
                        with st.spinner("標準サンプルデータを生成中..."):
                            sample_data = stats_engine.generate_sample_dataset(n=150, seed=42)
                            st.session_state["raw_df"] = sample_data.copy()
                            st.session_state["df"] = sample_data.copy()
                            st.session_state["last_uploaded"] = "sample_data_fixed"
                        st.toast("標準サンプルデータを読み込みました！", icon="🎯")
                        st.rerun()
            return
        else:
            df = st.session_state["df"]
    else:
        file_identifier = f"{uploaded_file.name}_{selected_sheet}_{uploaded_file.size}"
        if "last_uploaded" not in st.session_state or st.session_state["last_uploaded"] != file_identifier:
            try:
                with st.spinner("データを読み込み中..."):
                    df_raw, load_info = load_data_file(uploaded_file, encoding_choice=encoding_choice, sheet_name=selected_sheet)
                    st.session_state["raw_df"] = df_raw.copy()
                    st.session_state["df"] = df_raw.copy()
                    st.session_state["last_uploaded"] = file_identifier
                st.toast(f"✅ {uploaded_file.name} ({load_info}) を読み込みました！", icon="📂")
            except Exception as e:
                st.error(f"ファイルの読み込みエラー: {e}")
                return
        df = st.session_state["df"]

    st.caption(f"現在の分析対象データ: {df.shape[0]}行 × {df.shape[1]}列")

    # 変数列の分類 (前処理・分析共通)
    all_columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # ------------------------------------------------------------------
    # データ前処理・リコード・フィルタリング セクション
    # ------------------------------------------------------------------
    with st.expander("🛠️ データの加工・Recode（再符号化）・特定値の除外・欠損値処理", expanded=False):
        c_p_cat, c_p_tool = st.columns([1.2, 2.8])
        with c_p_cat:
            proc_category = st.radio(
                "前処理カテゴリー:",
                [
                    "🧼 クリーニング・除外",
                    "🔄 変数の変換・符号化",
                    "📐 尺度・スコア作成"
                ],
                key="proc_cat_radio"
            )
            
        with c_p_tool:
            if "クリーニング" in proc_category:
                proc_tool = st.selectbox(
                    "実行するツールを選択:",
                    [
                        "① 特定値の除外 (無効値 -99等の行を一括削除)",
                        "⑦ 欠損値の処理 (リストワイズ削除)",
                        "⑧ 複数条件データ抽出 (AND / OR フィルタ)"
                    ],
                    key="proc_tool_clean"
                )
            elif "変換・符号化" in proc_category:
                proc_tool = st.selectbox(
                    "実行するツールを選択:",
                    [
                        "② 値のリコード (値の置き換え・再符号化)",
                        "⑤ 逆転項目の反転 (アンケート尺度反転)",
                        "⑨ ダミー変数化 (One-Hot Encoding)"
                    ],
                    key="proc_tool_trans"
                )
            else:  # 尺度・スコア作成
                proc_tool = st.selectbox(
                    "実行するツールを選択:",
                    [
                        "③ 数値変数のカテゴリ化 (ビン分割)",
                        "④ 合成スコア作成 (複数変数の合計・平均)",
                        "⑥ 標準化 / 正規化 (Zスコア / 0-1 スケーリング)"
                    ],
                    key="proc_tool_scale"
                )

        st.divider()

        # --- ① 特定値の除外 (複数変数対応) ---
        if "① 特定値の除外" in proc_tool:
            st.markdown("**指定した複数変数から、無効値（例: -99, 99, '無回答' など）を含む行を一括で除外します。**")
            col_ex1, col_ex2 = st.columns(2)
            with col_ex1:
                target_ex_vars = st.multiselect("対象の変数を選択 (複数選択可):", all_columns, default=[], key="ex_vars")
            with col_ex2:
                if target_ex_vars:
                    # 選択された全変数のユニーク値を集約
                    all_uniques = []
                    for v in target_ex_vars:
                        all_uniques.extend(df[v].dropna().unique().tolist())
                    unique_vals = sorted(list(set(all_uniques)), key=lambda x: str(x))
                else:
                    unique_vals = []
                vals_to_exclude = st.multiselect("除外したい値を選択 (該当する行が削除されます):", unique_vals, default=[], key="ex_vals")
                
            if st.button("🚫 指定した値の行を除外してデータを更新", key="btn_apply_ex"):
                if target_ex_vars and vals_to_exclude:
                    new_df = stats_engine.filter_exclude_values(df, target_ex_vars, vals_to_exclude)
                    st.session_state["df"] = new_df
                    st.toast(f"{len(target_ex_vars)} 変数から {vals_to_exclude} を除外しました ({len(df)}行 → {len(new_df)}行)", icon="✂️")
                    st.rerun()
                else:
                    st.warning("対象の変数と除外する値をそれぞれ1つ以上選択してください。")

        # --- ② 値のリコード (複数変数対応 & 動的変数名) ---
        elif "② 値のリコード" in proc_tool:
            st.markdown("**変数の特定の値を別の値に置き換えて新しい変数を作成します（複数変数への一括適用に対応）。**")
            rec_vars = st.multiselect("リコード元の変数を選択 (複数選択可):", all_columns, default=[], key="rec_vars")
            
            c_r_opt1, c_r_opt2 = st.columns(2)
            with c_r_opt1:
                rec_suffix = st.text_input("新変数の末尾 (接尾辞):", value="_recoded", key="rec_suffix")
            with c_r_opt2:
                if rec_vars:
                    preview_names = [f"`{v}{rec_suffix}`" for v in rec_vars]
                    st.caption(f"✨ 作成される新変数名:\n{', '.join(preview_names)}")
                else:
                    st.caption("変数を1つ以上選択してください。")

            if rec_vars:
                # 選択された全変数のユニーク値を集約
                all_rec_uniques = []
                for v in rec_vars:
                    all_rec_uniques.extend(df[v].dropna().unique().tolist())
                unique_rec_vals = sorted(list(set(all_rec_uniques)), key=lambda x: str(x))
                
                st.write("各値の置き換えルールを設定してください:")
                mapping_dict = {}
                # 2列グリッドで配置
                col_a, col_b = st.columns(2)
                for i, val in enumerate(unique_rec_vals):
                    with col_a if i % 2 == 0 else col_b:
                        new_val_str = st.text_input(f"旧値 `{val}` の新値:", value=str(val), key=f"rec_val_{i}_{str(val)}")
                        try:
                            if "." in new_val_str:
                                val_conv = float(new_val_str)
                            else:
                                val_conv = int(new_val_str)
                        except ValueError:
                            val_conv = new_val_str
                        mapping_dict[val] = val_conv
                        
                if st.button("🔄 リコードを実行して新変数を作成", key="btn_apply_rec"):
                    new_df, created_names = stats_engine.recode_values(df, rec_vars, mapping_dict, suffix=rec_suffix)
                    st.session_state["df"] = new_df
                    st.toast(f"{len(created_names)} 個の新変数を作成しました！ ({', '.join(created_names)})", icon="✨")
                    st.rerun()

        # --- ③ 数値変数のカテゴリ化 (複数変数対応 & 動的変数名) ---
        elif "③ 数値変数のカテゴリ化" in proc_tool:
            st.markdown("**連続数値変数（例: 年齢、得点）を区切り値でカテゴリ変数（例: 年齢層、得点区分）に変換します。**")
            bin_vars = st.multiselect("カテゴリ化する数値変数 (複数選択可):", numeric_columns, default=[], key="bin_vars")
            
            c_b1, c_b2 = st.columns(2)
            with c_b1:
                bin_suffix = st.text_input("新変数の末尾 (接尾辞):", value="_層", key="bin_suffix")
            with c_b2:
                if bin_vars:
                    b_preview = [f"`{v}{bin_suffix}`" for v in bin_vars]
                    st.caption(f"✨ 作成される新変数名:\n{', '.join(b_preview)}")

            c_b3, c_b4 = st.columns(2)
            with c_b3:
                cuts_input = st.text_input("区切り値をカンマ区切りで入力 (例: 0, 30, 50, 100):", value="0, 30, 50, 100", key="bin_cuts")
            with c_b4:
                labels_input = st.text_input("ラベルをカンマ区切りで入力 (例: 低, 中, 高):", value="低, 中, 高", key="bin_labels")
                
            if st.button("📊 カテゴリ化（ビン分割）を実行", key="btn_apply_bin"):
                if bin_vars:
                    try:
                        cuts = [float(x.strip()) for x in cuts_input.split(",")]
                        labels = [x.strip() for x in labels_input.split(",")]
                        new_df, created_names = stats_engine.create_binned_variable(df, bin_vars, cuts, labels, suffix=bin_suffix)
                        st.session_state["df"] = new_df
                        st.toast(f"{len(created_names)} 個の新変数を作成しました！ ({', '.join(created_names)})", icon="✨")
                        st.rerun()
                    except Exception as e:
                        st.error(f"エラー: {e}")
                else:
                    st.warning("1つ以上の数値変数を選択してください。")

        # --- ④ 合成スコア作成 ---
        elif "④ 合成スコア作成" in proc_tool:
            st.markdown("**複数の数値変数から「平均値」または「合計値」の新変数（尺度得点など）を作成します。**")
            source_vars = st.multiselect("合成する変数を選択 (複数選択):", numeric_columns, default=[], key="comp_vars")
            
            c_c1, c_c2 = st.columns(2)
            with c_c1:
                comp_method = st.radio("計算方法:", ["平均値 (Mean)", "合計値 (Sum)"], horizontal=True, key="comp_method")
            with c_c2:
                default_name = "合成スコア_平均" if "平均値" in comp_method else "合成スコア_合計"
                comp_new_name = st.text_input("作成する新変数名:", value=default_name, key="comp_new_name")
                if source_vars:
                    st.caption(f"✨ 投入変数: {len(source_vars)} 項目 ({', '.join(source_vars)})")
                
            if st.button("➕ 合成スコアを作成", key="btn_apply_comp"):
                if len(source_vars) >= 2:
                    func_type = "mean" if "平均値" in comp_method else "sum"
                    new_df, created_name = stats_engine.create_composite_score(df, source_vars, func=func_type, new_var_name=comp_new_name)
                    st.session_state["df"] = new_df
                    st.toast(f"新変数 『{created_name}』 を作成しました！", icon="✨")
                    st.rerun()
                else:
                    st.warning("合成には2つ以上の変数を選択してください。")

        # --- ⑤ 逆転項目の反転 ---
        elif "⑤ 逆転項目の反転" in proc_tool:
            st.markdown("**アンケートの逆転項目（例: 1~5件法で 1↔5, 2↔4 に反転）を一括処理します。**")
            rev_vars = st.multiselect("反転する変数を選択 (複数選択可):", numeric_columns, default=[], key="rev_vars")
            
            c_r1, c_r2, c_r3 = st.columns(3)
            with c_r1:
                min_val = st.number_input("尺度の最小値 (例: 1):", value=1.0, key="rev_min")
            with c_r2:
                max_val = st.number_input("尺度の最大値 (例: 5や7):", value=5.0, key="rev_max")
            with c_r3:
                rev_suffix = st.text_input("新変数の末尾 (接尾辞):", value="_rev", key="rev_suffix")
                
            if rev_vars:
                r_preview = [f"`{v}{rev_suffix}`" for v in rev_vars]
                st.caption(f"✨ 作成される新変数名: {', '.join(r_preview)} | 計算式: 新値 = ({min_val} + {max_val}) - 旧値")
            
            if st.button("🔄 逆転項目の反転を実行", key="btn_apply_rev"):
                if rev_vars:
                    new_df, created_names = stats_engine.reverse_code_values(df, rev_vars, min_val, max_val, rev_suffix)
                    st.session_state["df"] = new_df
                    st.toast(f"{len(created_names)} 個の反転変数を作成しました！ ({', '.join(created_names)})", icon="✨")
                    st.rerun()
                else:
                    st.warning("1つ以上の変数を選択してください。")

        # --- ⑥ 標準化 / 正規化 ---
        elif "⑥ 標準化 / 正規化" in proc_tool:
            st.markdown("**数値変数を「標準化（Zスコア: 平均0, 分散1）」または「正規化（0〜1スケーリング）」します。**")
            std_vars = st.multiselect("変換する数値変数を選択 (複数選択可):", numeric_columns, default=[], key="std_vars")
            
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                std_method = st.radio("変換方法:", ["標準化 (Zスコア化: 平均0, SD 1)", "正規化 (0-1 Min-Maxスケーリング)"], key="std_method")
            with c_s2:
                method_type = "standardize" if "標準化" in std_method else "normalize"
                default_suffix = "_z" if method_type == "standardize" else "_norm"
                std_suffix = st.text_input("新変数の末尾 (接尾辞):", value=default_suffix, key="std_suffix")
                
            if std_vars:
                s_preview = [f"`{v}{std_suffix}`" for v in std_vars]
                st.caption(f"✨ 作成される新変数名: {', '.join(s_preview)}")
                
            if st.button("📐 標準化 / 正規化を実行", key="btn_apply_std"):
                if std_vars:
                    new_df, created_names = stats_engine.standardize_normalize_variables(df, std_vars, method=method_type, suffix=std_suffix)
                    st.session_state["df"] = new_df
                    st.toast(f"{len(created_names)} 個の変換変数を作成しました！ ({', '.join(created_names)})", icon="✨")
                    st.rerun()
                else:
                    st.warning("1つ以上の変数を選択してください。")

        # --- ⑦ 欠損値処理 ---
        elif "⑦ 欠損値の処理" in proc_tool:
            st.markdown("**データ内の欠損値（NaN/空白）を含む行をリストワイズ削除します。**")
            na_target_vars = st.multiselect("対象とする変数 (未指定の場合は全変数で欠損行を削除):", all_columns, default=[], key="na_vars")
            st.caption("※ 学術研究・論文執筆では、回答に欠損（空白）があるサンプルを除外するリストワイズ削除が標準的で安全な手法です。")
            if st.button("🧹 リストワイズ削除を実行", key="btn_apply_na"):
                target_v = na_target_vars if na_target_vars else None
                new_df = stats_engine.handle_missing_values(df, strategy="listwise", target_vars=target_v)
                st.session_state["df"] = new_df
                st.toast(f"リストワイズ削除を適用しました ({len(df)}行 → {len(new_df)}行)", icon="🧼")
                st.rerun()

        # --- ⑧ 複数条件フィルタ ---
        elif "⑧ 複数条件データ抽出" in proc_tool:
            st.markdown("**複数条件（AND / OR）を組み合わせて分析対象データを抽出します。**")
            filt_logic = st.radio("条件の結合方法:", ["すべての条件を満たす (AND)", "いずれかの条件を満たす (OR)"], horizontal=True, key="filt_logic")
            logic_code = "AND" if "AND" in filt_logic else "OR"
            
            if "num_conditions" not in st.session_state:
                st.session_state["num_conditions"] = 1
                
            c_f_btn1, c_f_btn2, _ = st.columns([1, 1, 3])
            with c_f_btn1:
                if st.button("➕ 条件を追加", key="btn_add_cond"):
                    st.session_state["num_conditions"] += 1
                    st.rerun()
            with c_f_btn2:
                if st.button("➖ 条件を減らす", key="btn_sub_cond") and st.session_state["num_conditions"] > 1:
                    st.session_state["num_conditions"] -= 1
                    st.rerun()
                    
            conditions = []
            for c_idx in range(st.session_state["num_conditions"]):
                cf1, cf2, cf3 = st.columns([2, 1.5, 2.5])
                with cf1:
                    col_name = st.selectbox(f"条件{c_idx+1} 変数:", all_columns, index=None, placeholder="変数を選択...", key=f"f_col_{c_idx}")
                with cf2:
                    op = st.selectbox(f"条件{c_idx+1} 演算子:", ["==", "!=", ">", ">=", "<", "<=", "contains", "in"], key=f"f_op_{c_idx}")
                with cf3:
                    val_str = st.text_input(f"条件{c_idx+1} 比較値 (カンマ区切り可):", value="", key=f"f_val_{c_idx}")
                if val_str and col_name:
                    conditions.append((col_name, op, val_str))
                    
            if st.button("🔍 フィルタを実行してデータを抽出", key="btn_apply_filter"):
                if conditions:
                    new_df = stats_engine.filter_advanced(df, conditions, logic=logic_code)
                    st.session_state["df"] = new_df
                    st.toast(f"フィルタを適用しました ({len(df)}行 → {len(new_df)}行)", icon="🎯")
                    st.rerun()
                else:
                    st.warning("変数と比較値を入力してください。")

        # --- ⑨ ダミー変数化 (複数変数対応) ---
        elif "⑨ ダミー変数化" in proc_tool:
            st.markdown("**カテゴリ変数（名義尺度）を 0 と 1 のダミー変数（One-Hot Encoding）に変換します（複数変数の一括変換に対応）。**")
            dummy_vars = st.multiselect("ダミー変数化するカテゴリ変数 (複数選択可):", all_columns, default=[], key="dummy_vars")
            
            c_d1, c_d2 = st.columns(2)
            with c_d1:
                drop_first = st.checkbox("各変数の最初のカテゴリを除外 (参照カテゴリ・多重共線性対策)", value=False, key="dummy_drop_first")
                st.caption("※ 回帰分析等で多重共線性（マルチコ）を防ぐ場合はチェックを推奨します。")
            with c_d2:
                if dummy_vars:
                    st.caption(f"✨ 変換対象: {len(dummy_vars)} 個のカテゴリ変数 ({', '.join(dummy_vars)})")
                    
            if st.button("🏷️ ダミー変数を作成", key="btn_apply_dummy"):
                if dummy_vars:
                    new_df, created_cols = stats_engine.create_dummy_variables(df, dummy_vars, drop_first=drop_first)
                    st.session_state["df"] = new_df
                    st.toast(f"{len(created_cols)} 個のダミー変数を作成しました！ ({', '.join(created_cols)})", icon="✨")
                    st.rerun()
                else:
                    st.warning("1つ以上のカテゴリ変数を選択してください。")
            


        st.divider()
        st.markdown("#### 💾 加工・変更済みデータの保存 / リセット")
        c_dl_csv, c_dl_xlsx, c_rst = st.columns([1.2, 1.2, 1.5])
        
        with c_dl_csv:
            csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                label="📥 CSV形式で保存 (.csv)",
                data=csv_bytes,
                file_name="加工済みデータ.csv",
                mime="text/csv",
                use_container_width=True,
                help="Excelでも文字化けしないUTF-8 (BOM付き) CSV形式で保存します。"
            )
        with c_dl_xlsx:
            excel_io = io.BytesIO()
            with pd.ExcelWriter(excel_io, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="加工済みデータ")
            excel_bytes = excel_io.getvalue()
            st.download_button(
                label="📥 Excel形式で保存 (.xlsx)",
                data=excel_bytes,
                file_name="加工済みデータ.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                help="加工後の全データを含むExcelファイルとして保存します。"
            )
        with c_rst:
            if st.button("↩️ データを初期状態に戻す", use_container_width=True):
                st.session_state["df"] = st.session_state["raw_df"].copy()
                st.toast("データを初期状態にリセットしました", icon="🔄")
                st.rerun()


    # データプレビュー
    with st.expander("🔍 現在のデータプレビュー & 変数一覧", expanded=False):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.dataframe(df.head(10), use_container_width=True)
        with col2:
            st.write("**データ型一覧:**")
            st.dataframe(pd.DataFrame(df.dtypes, columns=["データ型"]), use_container_width=True)
            
        c_p_dl1, c_p_dl2 = st.columns(2)
        with c_p_dl1:
            csv_bytes_prev = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                label="📥 現在のデータをCSVで保存 (.csv)",
                data=csv_bytes_prev,
                file_name="加工済みデータ.csv",
                mime="text/csv",
                key="dl_preview_csv",
                use_container_width=True
            )
        with c_p_dl2:
            excel_io_prev = io.BytesIO()
            with pd.ExcelWriter(excel_io_prev, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="加工済みデータ")
            excel_bytes_prev = excel_io_prev.getvalue()
            st.download_button(
                label="📥 現在のデータをExcelで保存 (.xlsx)",
                data=excel_bytes_prev,
                file_name="加工済みデータ.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_preview_xlsx",
                use_container_width=True
            )

    # ------------------------------------------------------------------
    # メイン分析セクション (4大カテゴリー階層ナビゲーション)
    # ------------------------------------------------------------------
    st.markdown("### 📊 統計分析メニュー")
    cat_tab1, cat_tab2, cat_tab3, cat_tab4 = st.tabs([
        "1. 📊 基本・記述統計",
        "2. ⚖️ 平均値の差の検定",
        "3. 📈 関連・回帰モデル",
        "4. 🧩 尺度・多変量解析"
    ])

    # ==================================================================
    # カテゴリー 1: 基本・記述統計
    # ==================================================================
    with cat_tab1:
        stat_method_1 = st.radio(
            "分析手法を選択:",
            ["1. 単純集計 (Frequency)", "2. 基本統計量 (Descriptive)", "3. クロス集計 (Crosstab & Chi-Square)"],
            horizontal=True,
            key="method_cat1"
        )
        st.divider()

        # 1. 単純集計
        if "1. 単純集計" in stat_method_1:
            st.subheader("1. 単純集計 (Frequency Analysis)")
            target_vars = st.multiselect("集計したいカテゴリ変数を選択 (複数選択可):", all_columns, default=[], key="freq_vars")
            
            if st.button("🚀 単純集計を実行", key="run_freq", type="primary"):
                if target_vars:
                    results_dict = {}
                    with st.spinner(f"{len(target_vars)} 個の変数の単純集計を計算中..."):
                        for var in target_vars:
                            res_df, note, fig_bytes = stats_engine.analyze_frequency(df, var)
                            results_dict[var] = {
                                "res_df": res_df, "note": note, "fig_bytes": fig_bytes, "target_var": var
                            }
                    st.session_state["results"]["freq"] = {
                        "results_dict": results_dict
                    }
                    st.toast(f"{len(results_dict)} 個の変数の単純集計が完了しました！", icon="✅")
                else:
                    st.warning("1つ以上の変数を選択してください。")
                
            if "freq" in st.session_state["results"]:
                res_data = st.session_state["results"]["freq"]
                results_dict = res_data.get("results_dict", {})
                if not results_dict and "res_df" in res_data:
                    results_dict = {res_data["target_var"]: res_data}
                    
                if results_dict:
                    c_btn1, _ = st.columns([2, 1])
                    with c_btn1:
                        if st.button(f"➕ 選択した全変数 ({len(results_dict)}件) をExcelレポートに追加", key="btn_add_freq"):
                            for var, res in results_dict.items():
                                st.session_state["analysis_queue"].append({
                                    "sheet_name": f"単純集計_{var}",
                                    "title": f"{var} の度数分布表",
                                    "df": res["res_df"],
                                    "note": res["note"],
                                    "fig_bytes": res["fig_bytes"]
                                })
                            st.toast(f"『単純集計 ({len(results_dict)}件)』をレポートリストに追加しました！", icon="📋")
                            st.rerun()

                    active_var = st.selectbox("表示する変数の切替:", list(results_dict.keys()), key="select_freq_display")
                    if active_var in results_dict:
                        res = results_dict[active_var]
                        c1, c2 = st.columns([2.2, 2])
                        with c1:
                            st.write(f"**集計表: {res['target_var']}**")
                            st.dataframe(res["res_df"], use_container_width=True)
                            st.caption(res["note"])
                        with c2:
                            st.write("**度数分布グラフ (APA Style)**")
                            with st.container(height=450):
                                st.image(res["fig_bytes"], use_column_width=True)

        # 2. 基本統計量
        elif "2. 基本統計量" in stat_method_1:
            st.subheader("2. 基本統計量 (Descriptive Statistics)")
            selected_num_vars = st.multiselect("分析する数値変数を選択:", numeric_columns, default=[], key="desc_vars")
            
            if st.button("🚀 基本統計量を計算", key="run_desc", type="primary"):
                if selected_num_vars:
                    with st.spinner("基本統計量と変数ごとの分布プロットを作成中..."):
                        desc_res = stats_engine.analyze_descriptives(df, selected_num_vars)
                        if len(desc_res) == 5:
                            res_df, hist_dict, note, fig_bytes, fig_dict = desc_res
                        else:
                            res_df, hist_dict, note, fig_bytes = desc_res
                            fig_dict = {selected_num_vars[0]: fig_bytes} if selected_num_vars else {}
                            
                        st.session_state["results"]["desc"] = {
                            "res_df": res_df, "hist_dict": hist_dict, "note": note,
                            "fig_bytes": fig_bytes, "fig_dict": fig_dict,
                            "selected_num_vars": selected_num_vars
                        }
                    st.toast("基本統計量の計算が完了しました！", icon="✅")
                else:
                    st.warning("1つ以上の数値変数を選択してください。")
                    
            if "desc" in st.session_state["results"]:
                res = st.session_state["results"]["desc"]
                fig_dict = res.get("fig_dict", {})
                hist_dict = res.get("hist_dict", {})
                num_vars_list = res.get("selected_num_vars", list(res["res_df"].index))
                
                c_btn_d1, c_btn_d2 = st.columns([1.5, 1.5])
                with c_btn_d1:
                    if st.button("➕ 変数ごとに個別シートでExcelに追加 (度数表 & 分布図)", key="btn_add_desc_separate"):
                        for var_name in num_vars_list:
                            var_hist_df = hist_dict.get(var_name, pd.DataFrame())
                            var_fig = fig_dict.get(var_name, res["fig_bytes"])
                            st.session_state["analysis_queue"].append({
                                "sheet_name": f"分布_{var_name}",
                                "title": f"{var_name} の度数分布 & 基本統計量",
                                "df": var_hist_df if not var_hist_df.empty else res["res_df"].loc[[var_name]],
                                "note": res["note"],
                                "fig_bytes": var_fig
                            })
                        st.toast(f"『基本統計量 (変数ごと {len(num_vars_list)} シート)』をレポートリストに追加しました！", icon="📋")
                        st.rerun()
                with c_btn_d2:
                    if st.button("➕ 全変数の基本統計量一覧表を1シートに追加", key="btn_add_desc_summary"):
                        st.session_state["analysis_queue"].append({
                            "sheet_name": "基本統計量_サマリー",
                            "title": "選択変数の基本統計量一覧表",
                            "df": res["res_df"],
                            "note": res["note"],
                            "fig_bytes": res["fig_bytes"]
                        })
                        st.toast("『基本統計量一覧表』をレポートリストに追加しました！", icon="📋")
                        st.rerun()

                c1, c2 = st.columns([2.2, 2.2])
                with c1:
                    st.write("**📋 基本統計量一覧 (APA Style)**")
                    st.dataframe(res["res_df"], use_container_width=True)
                    st.caption(res["note"])
                    
                with c2:
                    st.write("**📈 変数ごとの分布プロット (ヒストグラム & 確率密度)**")
                    if fig_dict:
                        active_plot_var = st.selectbox(
                            "表示する変数を選択:",
                            list(fig_dict.keys()),
                            key="select_desc_plot_var"
                        )
                        if active_plot_var in fig_dict:
                            st.image(fig_dict[active_plot_var], use_container_width=True)
                            
                            c_dl_v1, _ = st.columns([2, 1])
                            with c_dl_v1:
                                st.download_button(
                                    label=f"📥 『{active_plot_var}』の分布図を保存 (.png)",
                                    data=fig_dict[active_plot_var],
                                    file_name=f"分布プロット_{active_plot_var}.png",
                                    mime="image/png",
                                    use_container_width=True,
                                    key=f"dl_plot_{active_plot_var}"
                                )
                            
                            if active_plot_var in hist_dict:
                                with st.expander(f"📊 『{active_plot_var}』の度数分布テーブル", expanded=False):
                                    st.dataframe(hist_dict[active_plot_var], use_container_width=True)
                    else:
                        st.image(res["fig_bytes"], use_container_width=True)

        # 3. クロス集計
        elif "3. クロス集計" in stat_method_1:
            st.subheader("3. クロス集計 & カイ二乗検定 (Crosstab & Chi-Square)")
            c1, c2 = st.columns(2)
            with c1:
                row_var = st.selectbox("行変数 (Row):", all_columns, index=None, placeholder="行変数を選択...", key="ct_row")
            with c2:
                col_options = [c for c in all_columns if c != row_var]
                col_var = st.selectbox("列変数 (Column):", col_options, index=None, placeholder="列変数を選択...", key="ct_col")
                
            if st.button("🚀 クロス集計を実行", key="run_ct", type="primary"):
                if row_var and col_var:
                    with st.spinner("クロス集計とカイ二乗検定を計算中..."):
                        ct_formatted_df, pct_df, note, fig_bytes = stats_engine.analyze_crosstab(df, row_var, col_var)
                        st.session_state["results"]["ct"] = {
                            "ct_formatted_df": ct_formatted_df, "pct_df": pct_df, "note": note, "fig_bytes": fig_bytes,
                            "row_var": row_var, "col_var": col_var
                        }
                    st.toast("クロス集計が完了しました！", icon="✅")
                else:
                    st.warning("行変数と列変数をそれぞれ選択してください。")
                    
            if "ct" in st.session_state["results"]:
                res = st.session_state["results"]["ct"]
                
                if st.button("➕ この結果をExcelレポートに追加", key="btn_add_ct"):
                    st.session_state["analysis_queue"].append({
                        "sheet_name": f"クロス_{res['row_var']}_vs_{res['col_var']}",
                        "title": f"クロス集計表 ({res['row_var']} × {res['col_var']})",
                        "df": res["ct_formatted_df"],
                        "note": res["note"]
                    })
                    st.session_state["analysis_queue"].append({
                        "sheet_name": f"構成比_{res['row_var']}_vs_{res['col_var']}",
                        "title": f"構成比 (%) 表 ({res['row_var']} × {res['col_var']})",
                        "df": res["pct_df"],
                        "note": "注. 数値は列方向の構成比 (%) を表す。"
                    })
                    st.toast("『クロス集計表 & 構成比グラフ』をレポートリストに追加しました！", icon="📋")
                    st.rerun()

                col_left, col_right = st.columns([2.5, 2])
                with col_left:
                    st.write(f"**クロス度数表 [度数 (列%)] ({res['row_var']} × {res['col_var']})**")
                    st.dataframe(res["ct_formatted_df"], use_container_width=True)
                    st.caption(res["note"])
                with col_right:
                    st.write("**構成比グラフ (100% 積み上げ)**")
                    with st.container(height=450):
                        st.image(res["fig_bytes"], use_column_width=True)

    # ==================================================================
    # カテゴリー 2: 平均値の差の検定
    # ==================================================================
    with cat_tab2:
        stat_method_2 = st.radio(
            "分析手法を選択:",
            ["4. 独立2群t検定", "5. 対応のあるt検定", "6. 一元配置ANOVA", "7. 二元配置ANOVA"],
            horizontal=True,
            key="method_cat2"
        )
        st.divider()

        # 4. 独立2群のt検定
        if "4. 独立2群t検定" in stat_method_2:
            st.subheader("4. 独立2群の平均値の差の検定 (Independent Samples t-Test / Welch)")
            c1, c2, c3 = st.columns([1.5, 2, 1.5])
            with c1:
                group_var = st.selectbox("グループ変数 (2カテゴリ):", all_columns, index=None, placeholder="グループ変数を選択...", key="tt_group")
            with c2:
                num_vars = st.multiselect("比較する従属変数 (複数選択可):", numeric_columns, default=[], key="tt_nums")
            with c3:
                equal_var_opt = st.selectbox("等分散性の仮定:", ["Welchのt検定 (推奨: 等分散非仮定)", "Studentのt検定 (等分散仮定)"], key="tt_eq_opt")
                equal_var = True if "Student" in equal_var_opt else False
                
            if st.button("🚀 独立t検定を実行", key="run_tt", type="primary"):
                if group_var and num_vars:
                    results_dict = {}
                    errors = []
                    with st.spinner(f"{len(num_vars)} 個の従属変数のt検定を計算中..."):
                        for nv in num_vars:
                            try:
                                res_df, note, fig_bytes = stats_engine.analyze_ttest(df, group_var, nv, equal_var=equal_var)
                                results_dict[nv] = {
                                    "res_df": res_df, "note": note, "fig_bytes": fig_bytes, "num_var": nv, "group_var": group_var
                                }
                            except Exception as e:
                                errors.append(f"{nv}: {e}")
                    
                    if results_dict:
                        st.session_state["results"]["tt"] = {
                            "results_dict": results_dict, "group_var": group_var
                        }
                        st.toast(f"{len(results_dict)} 個の従属変数のt検定が完了しました！", icon="✅")
                    if errors:
                        for err in errors:
                            st.error(f"分析エラー ({err})")
                else:
                    st.warning("グループ変数と1つ以上の従属変数を選択してください。")
                        
            if "tt" in st.session_state["results"]:
                res_data = st.session_state["results"]["tt"]
                results_dict = res_data.get("results_dict", {})
                if not results_dict and "res_df" in res_data:
                    results_dict = {res_data["num_var"]: res_data}
                    
                if results_dict:
                    grp_v = res_data.get("group_var", "")
                    combined_df = pd.concat([res["res_df"] for res in results_dict.values()])
                    combined_notes = " | ".join([f"{nv}: {res['note']}" for nv, res in results_dict.items()]) if len(results_dict) > 1 else list(results_dict.values())[0]["note"]
                    
                    c_btn1, _ = st.columns([2, 1])
                    with c_btn1:
                        if st.button(f"➕ 選択した全従属変数 ({len(results_dict)}件) の結果をExcelレポートに追加", key="btn_add_tt"):
                            st.session_state["analysis_queue"].append({
                                "sheet_name": f"t検定_{grp_v}",
                                "title": f"2群の平均値の比較 (t検定: {grp_v})",
                                "df": combined_df,
                                "note": combined_notes,
                                "fig_bytes": list(results_dict.values())[0]["fig_bytes"]
                            })
                            st.toast(f"『t検定一括結果 ({len(results_dict)}件)』をレポートリストに追加しました！", icon="📋")
                            st.rerun()

                    st.markdown(f"**📊 t検定 要約一覧表 (グループ変数: {grp_v})**")
                    st.dataframe(combined_df, use_container_width=True)

                    st.divider()
                    active_nv = st.selectbox("個別グラフ表示の従属変数を選択:", list(results_dict.keys()), key="select_tt_display")
                    if active_nv in results_dict:
                        res = results_dict[active_nv]
                        col_left, col_right = st.columns([2.5, 2])
                        with col_left:
                            st.write(f"**t検定詳細 ({res['num_var']} × {res['group_var']})**")
                            st.dataframe(res["res_df"], use_container_width=True)
                            st.caption(res["note"])
                        with col_right:
                            st.write("**平均値比較グラフ (95%信頼区間)**")
                            with st.container(height=450):
                                st.image(res["fig_bytes"], use_column_width=True)

        # 5. 対応のあるt検定
        elif "5. 対応のあるt検定" in stat_method_2:
            st.subheader("5. 対応のある2群の平均値の差の検定 (Paired Samples t-Test)")
            st.caption("同一の被験者における前後比較（例: 事前テスト vs 事後テスト）や対応する2条件間の平均値の差を検定します。")
            c1, c2 = st.columns(2)
            with c1:
                pair_var1 = st.selectbox("条件1 (例: 事前):", numeric_columns, index=None, placeholder="条件1の変数を選択...", key="pt_v1")
            with c2:
                pair_options = [c for c in numeric_columns if c != pair_var1]
                pair_var2 = st.selectbox("条件2 (例: 事後):", pair_options, index=None, placeholder="条件2の変数を選択...", key="pt_v2")
                
            if st.button("🚀 対応のあるt検定を実行", key="run_pt", type="primary"):
                if pair_var1 and pair_var2:
                    try:
                        with st.spinner("対応のあるt検定を計算中..."):
                            pt_df, note, fig_bytes = stats_engine.analyze_paired_ttest(df, pair_var1, pair_var2)
                            st.session_state["results"]["pt"] = {
                                "pt_df": pt_df, "note": note, "fig_bytes": fig_bytes,
                                "pair_var1": pair_var1, "pair_var2": pair_var2
                            }
                        st.toast("対応のあるt検定が完了しました！", icon="✅")
                    except Exception as e:
                        st.error(f"分析エラー: {e}")
                else:
                    st.warning("2つの数値変数を選択してください。")
                    
            if "pt" in st.session_state["results"]:
                res = st.session_state["results"]["pt"]
                
                if st.button("➕ この結果をExcelレポートに追加", key="btn_add_pt"):
                    st.session_state["analysis_queue"].append({
                        "sheet_name": f"対応ありt_{res['pair_var1']}_vs_{res['pair_var2']}",
                        "title": f"対応のあるt検定結果表 ({res['pair_var1']} vs. {res['pair_var2']})",
                        "df": res["pt_df"],
                        "note": res["note"],
                        "fig_bytes": res["fig_bytes"]
                    })
                    st.toast("『対応のあるt検定結果』をレポートリストに追加しました！", icon="📋")
                    st.rerun()

                c_l, c_r = st.columns([2.5, 2])
                with c_l:
                    st.write(f"**対応のあるt検定 結果表 ({res['pair_var1']} vs. {res['pair_var2']})**")
                    st.dataframe(res["pt_df"], use_container_width=True)
                    st.caption(res["note"])
                with c_r:
                    st.write("**平均値の変化プロット (95%信頼区間)**")
                    with st.container(height=450):
                        st.image(res["fig_bytes"], use_column_width=True)

        # 6. 一元配置ANOVA
        elif "6. 一元配置ANOVA" in stat_method_2:
            st.subheader("6. 一元配置分散分析 & 多重比較 (One-Way ANOVA)")
            c1, c2 = st.columns([1.5, 2])
            with c1:
                anova_group = st.selectbox("要因 (グループ変数):", all_columns, index=None, placeholder="要因変数を選択...", key="anova_grp")
            with c2:
                anova_nums = st.multiselect("従属変数 (複数選択可):", numeric_columns, default=[], key="anova_nums")
                
            if st.button("🚀 分散分析(ANOVA)を実行", key="run_anova", type="primary"):
                if anova_group and anova_nums:
                    results_dict = {}
                    errors = []
                    with st.spinner(f"{len(anova_nums)} 個の従属変数の分散分析を実行中..."):
                        for nv in anova_nums:
                            try:
                                anova_df, tukey_df, note, fig_bytes = stats_engine.analyze_anova(df, anova_group, nv)
                                results_dict[nv] = {
                                    "anova_df": anova_df, "tukey_df": tukey_df, "note": note, "fig_bytes": fig_bytes,
                                    "anova_num": nv, "anova_group": anova_group
                                }
                            except Exception as e:
                                errors.append(f"{nv}: {e}")
                    
                    if results_dict:
                        st.session_state["results"]["anova"] = {
                            "results_dict": results_dict, "anova_group": anova_group
                        }
                        st.toast(f"{len(results_dict)} 個の従属変数の分散分析が完了しました！", icon="✅")
                    if errors:
                        for err in errors:
                            st.error(f"分析エラー ({err})")
                else:
                    st.warning("要因変数と1つ以上の従属変数を選択してください。")
                    
            if "anova" in st.session_state["results"]:
                res_data = st.session_state["results"]["anova"]
                results_dict = res_data.get("results_dict", {})
                if not results_dict and "anova_df" in res_data:
                    results_dict = {res_data["anova_num"]: res_data}
                    
                if results_dict:
                    grp_v = res_data.get("anova_group", "")
                    combined_anova_df = pd.concat([res["anova_df"] for res in results_dict.values()])
                    combined_notes = " | ".join([f"{nv}: {res['note']}" for nv, res in results_dict.items()]) if len(results_dict) > 1 else list(results_dict.values())[0]["note"]
                    
                    tukey_list = []
                    for nv, res in results_dict.items():
                        tdf = res["tukey_df"].reset_index()
                        tdf.insert(0, "従属変数", nv)
                        tukey_list.append(tdf)
                    combined_tukey_df = pd.concat(tukey_list).set_index(["従属変数", "グループ1", "グループ2"]) if tukey_list else None
                    
                    c_btn1, _ = st.columns([2, 1])
                    with c_btn1:
                        if st.button(f"➕ 選択した全従属変数 ({len(results_dict)}件) の結果をExcelレポートに追加", key="btn_add_anova"):
                            st.session_state["analysis_queue"].append({
                                "sheet_name": f"分散分析_{grp_v}",
                                "title": f"一元配置分散分析表 ({grp_v})",
                                "df": combined_anova_df,
                                "note": combined_notes,
                                "fig_bytes": list(results_dict.values())[0]["fig_bytes"]
                            })
                            if combined_tukey_df is not None:
                                st.session_state["analysis_queue"].append({
                                    "sheet_name": f"多重比較_{grp_v}",
                                    "title": f"Tukey HSD 多重比較結果 ({grp_v})",
                                    "df": combined_tukey_df,
                                    "note": "注. 有意水準 alpha = .05 におけるTukeyのHSD検定結果。"
                                })
                            st.toast(f"『分散分析 & 多重比較一括結果 ({len(results_dict)}件)』をレポートリストに追加しました！", icon="📋")
                            st.rerun()

                    st.markdown(f"**📊 分散分析 要約一覧表 (要因: {grp_v})**")
                    st.dataframe(combined_anova_df, use_container_width=True)

                    st.divider()
                    active_nv = st.selectbox("個別結果表示の従属変数を選択:", list(results_dict.keys()), key="select_anova_display")
                    if active_nv in results_dict:
                        res = results_dict[active_nv]
                        c_l, c_r = st.columns([2.5, 2])
                        with c_l:
                            st.write(f"**分散分析詳細 ({res['anova_num']} × {res['anova_group']})**")
                            st.dataframe(res["anova_df"], use_container_width=True)
                            st.caption(res["note"])
                            
                            st.write("**Tukey HSD 多重比較結果**")
                            st.dataframe(res["tukey_df"], use_container_width=True)
                        with c_r:
                            st.write("**平均値比較プロット**")
                            with st.container(height=450):
                                st.image(res["fig_bytes"], use_column_width=True)

        # 7. 二元配置ANOVA
        elif "7. 二元配置ANOVA" in stat_method_2:
            st.subheader("7. 二元配置分散分析 (Two-Way ANOVA)")
            st.caption("2つの要因（独立変数）による主効果および交互作用効果を検定します。")
            c1, c2, c3 = st.columns(3)
            with c1:
                two_f1 = st.selectbox("要因1 (因子A):", all_columns, index=None, placeholder="要因1を選択...", key="two_f1")
            with c2:
                two_f2_options = [c for c in all_columns if c != two_f1]
                two_f2 = st.selectbox("要因2 (因子B):", two_f2_options, index=None, placeholder="要因2を選択...", key="two_f2")
            with c3:
                two_dep = st.selectbox("従属変数 (目的変数):", numeric_columns, index=None, placeholder="従属変数を選択...", key="two_dep")
                
            if st.button("🚀 二元配置分散分析を実行", key="run_two_anova", type="primary"):
                if two_f1 and two_f2 and two_dep:
                    try:
                        with st.spinner("二元配置分散分析と交互作用プロットを生成中..."):
                            anova_df, cell_desc, note, fig_bytes = stats_engine.analyze_two_way_anova(df, two_f1, two_f2, two_dep)
                            st.session_state["results"]["two_anova"] = {
                                "anova_df": anova_df, "cell_desc": cell_desc, "note": note, "fig_bytes": fig_bytes,
                                "two_f1": two_f1, "two_f2": two_f2, "two_dep": two_dep
                            }
                        st.toast("二元配置分散分析が完了しました！", icon="✅")
                    except Exception as e:
                        st.error(f"分析エラー: {e}")
                else:
                    st.warning("要因1、要因2、および従属変数を選択してください。")
                    
            if "two_anova" in st.session_state["results"]:
                res = st.session_state["results"]["two_anova"]
                
                if st.button("➕ この結果をExcelレポートに追加", key="btn_add_two_anova"):
                    st.session_state["analysis_queue"].append({
                        "sheet_name": f"二元配置ANOVA_{res['two_dep']}",
                        "title": f"二元配置分散分析表 (従属変数: {res['two_dep']})",
                        "df": res["anova_df"],
                        "note": res["note"],
                        "fig_bytes": res["fig_bytes"],
                        "extra_df": res["cell_desc"],
                        "extra_title": f"セル別平均値 (M (SD): {res['two_f1']} × {res['two_f2']})"
                    })
                    st.toast("『二元配置分散分析結果』をレポートリストに追加しました！", icon="📋")
                    st.rerun()

                c_l, c_r = st.columns([2.5, 2])
                with c_l:
                    st.write(f"**二元配置分散分析表 (従属変数: {res['two_dep']})**")
                    st.dataframe(res["anova_df"], use_container_width=True)
                    st.caption(res["note"])
                    
                    st.write(f"**セル別記述統計量 [M (SD)] ({res['two_f1']} × {res['two_f2']})**")
                    st.dataframe(res["cell_desc"], use_container_width=True)
                with c_r:
                    st.write("**交互作用プロット (Interaction Plot)**")
                    with st.container(height=450):
                        st.image(res["fig_bytes"], use_column_width=True)

    # ==================================================================
    # カテゴリー 3: 関連・回帰モデル
    # ==================================================================
    with cat_tab3:
        stat_method_3 = st.radio(
            "分析手法を選択:",
            ["8. 相関分析", "9. 重回帰分析", "10. ロジスティック回帰"],
            horizontal=True,
            key="method_cat3"
        )
        st.divider()

        # 8. 相関分析
        if "8. 相関分析" in stat_method_3:
            st.subheader("8. 相関分析 (Correlation Analysis)")
            c1, c2 = st.columns([2.5, 1])
            with c1:
                corr_vars = st.multiselect("相関を計算する数値変数 (複数選択):", numeric_columns, default=[], key="corr_vars")
            with c2:
                corr_method = st.radio("相関係数の種類:", ["pearson", "spearman"], key="corr_method_opt")
                
            if st.button("🚀 相関分析を実行", key="run_corr", type="primary"):
                if len(corr_vars) >= 2:
                    with st.spinner("相関係数行列とヒートマップを生成中..."):
                        res_df, note, fig_bytes = stats_engine.analyze_correlation(df, corr_vars, method=corr_method)
                        st.session_state["results"]["corr"] = {
                            "res_df": res_df, "note": note, "fig_bytes": fig_bytes, "corr_method": corr_method
                        }
                    st.toast("相関分析が完了しました！", icon="✅")
                else:
                    st.warning("相関分析には2つ以上の変数を選択してください。")
                    
            if "corr" in st.session_state["results"]:
                res = st.session_state["results"]["corr"]
                
                if st.button("➕ この結果をExcelレポートに追加", key="btn_add_corr"):
                    method_jp = "ピアソン" if res["corr_method"] == "pearson" else "スピアマン"
                    st.session_state["analysis_queue"].append({
                        "sheet_name": f"相関分析_{res['corr_method']}",
                        "title": f"{method_jp} 相関係数行列",
                        "df": res["res_df"],
                        "note": res["note"],
                        "fig_bytes": res["fig_bytes"]
                    })
                    st.toast("『相関係数行列』をレポートリストに追加しました！", icon="📋")
                    st.rerun()

                c_l, c_r = st.columns([2.5, 2])
                method_jp = "ピアソン" if res["corr_method"] == "pearson" else "スピアマン"
                with c_l:
                    st.write(f"**{method_jp} 相関係数行列**")
                    st.dataframe(res["res_df"], use_container_width=True)
                    st.caption(res["note"])
                with c_r:
                    st.write("**相関ヒートマップ**")
                    with st.container(height=450):
                        st.image(res["fig_bytes"], use_column_width=True)

        # 9. 重回帰分析
        elif "9. 重回帰分析" in stat_method_3:
            st.subheader("9. 重回帰分析 (Multiple Linear Regression)")
            c1, c2 = st.columns([1, 2])
            with c1:
                target_var = st.selectbox("目的変数 (Y):", numeric_columns, index=None, placeholder="目的変数を選択...", key="reg_target")
            with c2:
                avail_features = [c for c in numeric_columns if c != target_var]
                feature_vars = st.multiselect("説明変数 (X):", avail_features, default=[], key="reg_features")
                
            if st.button("🚀 回帰分析を実行", key="run_reg", type="primary"):
                if target_var and feature_vars:
                    with st.spinner("重回帰モデルを推定中..."):
                        res_df, note, fig_bytes = stats_engine.analyze_regression(df, target_var, feature_vars)
                        st.session_state["results"]["reg"] = {
                            "res_df": res_df, "note": note, "fig_bytes": fig_bytes, "target_var": target_var
                        }
                    st.toast("重回帰分析が完了しました！", icon="✅")
                else:
                    st.warning("目的変数と1つ以上の説明変数を選択してください。")
                    
            if "reg" in st.session_state["results"]:
                res = st.session_state["results"]["reg"]
                
                if st.button("➕ この結果をExcelレポートに追加", key="btn_add_reg"):
                    st.session_state["analysis_queue"].append({
                        "sheet_name": f"回帰分析_{res['target_var']}",
                        "title": f"{res['target_var']} を目的変数とする重回帰分析表",
                        "df": res["res_df"],
                        "note": res["note"],
                        "fig_bytes": res["fig_bytes"]
                    })
                    st.toast("『重回帰分析結果』をレポートリストに追加しました！", icon="📋")
                    st.rerun()

                c_l, c_r = st.columns([2.5, 2])
                with c_l:
                    st.write(f"**重回帰モデル分析結果表 (目的変数: {res['target_var']})**")
                    st.dataframe(res["res_df"], use_container_width=True)
                    st.caption(res["note"])
                with c_r:
                    st.write("**実測値 vs. 予測値プロット**")
                    with st.container(height=450):
                        st.image(res["fig_bytes"], use_column_width=True)

        # 10. ロジスティック回帰
        elif "10. ロジスティック回帰" in stat_method_3:
            st.subheader("10. 二項ロジスティック回帰分析 (Binary Logistic Regression)")
            st.caption("二値カテゴリ変数（例: 購買あり/なし、合格/不合格、0/1）の生起確率を予測し、オッズ比を算出します。")
            c1, c2 = st.columns([1, 2])
            with c1:
                logit_target = st.selectbox("目的変数 (二値カテゴリ):", all_columns, index=None, placeholder="目的変数を選択...", key="logit_target")
            with c2:
                logit_avail_features = [c for c in numeric_columns if c != logit_target]
                logit_features = st.multiselect("説明変数 (連続またはダミー変数):", logit_avail_features, default=[], key="logit_features")
                
            if st.button("🚀 ロジスティック回帰を実行", key="run_logit", type="primary"):
                if logit_target and logit_features:
                    try:
                        with st.spinner("ロジスティック回帰モデルを推定中..."):
                            logit_df, note, fig_bytes = stats_engine.analyze_logistic_regression(df, logit_target, logit_features)
                            st.session_state["results"]["logit"] = {
                                "logit_df": logit_df, "note": note, "fig_bytes": fig_bytes,
                                "logit_target": logit_target
                            }
                        st.toast("ロジスティック回帰分析が完了しました！", icon="✅")
                    except Exception as e:
                        st.error(f"分析エラー: {e}")
                else:
                    st.warning("目的変数と1つ以上の説明変数を選択してください。")
                    
            if "logit" in st.session_state["results"]:
                res = st.session_state["results"]["logit"]
                
                if st.button("➕ この結果をExcelレポートに追加", key="btn_add_logit"):
                    st.session_state["analysis_queue"].append({
                        "sheet_name": f"ロジスティック_{res['logit_target']}",
                        "title": f"ロジスティック回帰分析結果表 (目的変数: {res['logit_target']})",
                        "df": res["logit_df"],
                        "note": res["note"],
                        "fig_bytes": res["fig_bytes"]
                    })
                    st.toast("『ロジスティック回帰結果』をレポートリストに追加しました！", icon="📋")
                    st.rerun()

                c_l, c_r = st.columns([2.5, 2])
                with c_l:
                    st.write(f"**ロジスティック回帰パラメータ表 (オッズ比・95%CI)**")
                    st.dataframe(res["logit_df"], use_container_width=True)
                    st.caption(res["note"])
                with c_r:
                    st.write("**オッズ比 フォレストプロット (95%信頼区間)**")
                    with st.container(height=450):
                        st.image(res["fig_bytes"], use_column_width=True)

    # ==================================================================
    # カテゴリー 4: 尺度・多変量解析
    # ==================================================================
    with cat_tab4:
        stat_method_4 = st.radio(
            "分析手法を選択:",
            ["11. 因子分析", "12. 信頼性分析 (α)", "13. 共分散構造分析 / パス解析 (SEM)"],
            horizontal=True,
            key="method_cat4"
        )
        st.divider()

        # 11. 因子分析
        if "11. 因子分析" in stat_method_4:
            st.subheader("11. 探索的因子分析 (Exploratory Factor Analysis)")
            st.caption("負荷量の大きさで項目をソートし、共通性 (h²) の算出および主因子負荷量の太字表示に対応しています。")
            c1, c2, c3 = st.columns(3)
            with c1:
                fa_vars = st.multiselect("因子分析に投入する観測変数:", numeric_columns, default=[], key="fa_vars")
            with c2:
                max_f_possible = max(1, len(fa_vars) - 1) if fa_vars else 5
                n_factors = st.number_input("抽出する因子数:", min_value=1, max_value=max_f_possible, value=min(2, max_f_possible))
            with c3:
                rotation = st.selectbox("因子回転法:", ["promax", "varimax"], key="fa_rot")
                
            if st.button("🚀 因子分析を実行", key="run_fa", type="primary"):
                if len(fa_vars) >= 3:
                    with st.spinner("因子分析とスクリープロットを実行中..."):
                        res_df, note, fig_bytes, scree_df = stats_engine.analyze_factor_analysis(df, fa_vars, n_factors=int(n_factors), rotation=rotation)
                        st.session_state["results"]["fa"] = {
                            "res_df": res_df, "note": note, "fig_bytes": fig_bytes, "scree_df": scree_df, "rotation": rotation
                        }
                    st.toast("因子分析が完了しました！ (項目を負荷量順にソートしました)", icon="✅")
                else:
                    st.warning("因子分析には3つ以上の観測変数を選択してください。")
                    
            if "fa" in st.session_state["results"]:
                res = st.session_state["results"]["fa"]
                
                if st.button("➕ この結果をExcelレポートに追加 (編集可能スクリープロット付き)", key="btn_add_fa"):
                    rot_jp = "プロマックス回転" if res["rotation"] == "promax" else "バリマックス回転"
                    st.session_state["analysis_queue"].append({
                        "sheet_name": "因子分析",
                        "title": f"探索的因子分析結果 ({rot_jp})",
                        "df": res["res_df"],
                        "note": res["note"],
                        "fig_bytes": res["fig_bytes"],
                        "scree_df": res.get("scree_df", None)
                    })
                    st.toast("『因子分析結果』をレポートリストに追加しました！", icon="📋")
                    st.rerun()

                c_l, c_r = st.columns([2.5, 2])
                with c_l:
                    rot_jp = "プロマックス回転" if res["rotation"] == "promax" else "バリマックス回転"
                    st.write(f"**因子負荷量行列・共通性 (h²) & 寄与率 ({rot_jp})**")
                    st.dataframe(res["res_df"], use_container_width=True)
                    st.caption(res["note"])
                with c_r:
                    st.write("**スクリープロット (固有値の推移)**")
                    with st.container(height=450):
                        st.image(res["fig_bytes"], use_column_width=True)

        # 12. 尺度信頼性分析
        elif "12. 信頼性分析" in stat_method_4:
            st.subheader("12. 尺度信頼性分析 (Scale Reliability / Cronbach's Alpha)")
            st.caption("アンケート尺度などの内的一貫性をクロンバックのα係数で評価し、逆転項目（負の相関）を自動判別します。")
            rel_vars = st.multiselect("信頼性を分析する尺度項目 (数値変数):", numeric_columns, default=[], key="rel_vars")
            
            if st.button("🚀 信頼性分析を実行", key="run_rel", type="primary"):
                if len(rel_vars) >= 2:
                    try:
                        with st.spinner("クロンバックのα係数と逆転項目の判定を実行中..."):
                            rel_df, note, fig_bytes, alpha_val, alpha_corr, rev_candidates = stats_engine.analyze_reliability(df, rel_vars)
                            st.session_state["results"]["rel"] = {
                                "rel_df": rel_df, "note": note, "fig_bytes": fig_bytes, 
                                "alpha_val": alpha_val, "alpha_corr": alpha_corr, 
                                "rev_candidates": rev_candidates, "rel_vars": rel_vars
                            }
                        if rev_candidates:
                            st.toast(f"信頼性分析完了: 逆転項目の可能性あり ({', '.join(rev_candidates)})", icon="⚠️")
                        else:
                            st.toast(f"信頼性分析が完了しました！ (全体 α = {alpha_val:.3f})", icon="✅")
                    except Exception as e:
                        st.error(f"分析エラー: {e}")
                else:
                    st.warning("信頼性分析には2つ以上の変数を選択してください。")
                    
            if "rel" in st.session_state["results"]:
                res = st.session_state["results"]["rel"]
                
                # 逆転項目のアラートと反転時αの提示
                if res.get("rev_candidates"):
                    rev_list_str = ", ".join([f"`{v}`" for v in res["rev_candidates"]])
                    st.warning(
                        f"⚠️ **逆転項目（負の項目-全体相関）が検出されました**: {rev_list_str}\n\n"
                        f"- 現行の全体 α: **`{res['alpha_val']:.3f}`**\n"
                        f"- 該当項目を**反転補正した場合の推定 α**: **`{res.get('alpha_corr', 0.0):.3f}`** （大幅に向上します）\n"
                        f"- ※上部のデータ前処理メニュー『⑤ 逆転項目の反転』で反転変数を作成して再分析することを推奨します。"
                    )
                else:
                    st.success(f"✅ 全ての項目が正の相関を示しています。(全体 α = **`{res['alpha_val']:.3f}`**)")
                
                if st.button("➕ この結果をExcelレポートに追加 (編集可能グラフ付き)", key="btn_add_rel"):
                    st.session_state["analysis_queue"].append({
                        "sheet_name": "信頼性分析",
                        "title": f"尺度項目の信頼性分析表 (全体 α = {res['alpha_val']:.3f})",
                        "df": res["rel_df"],
                        "note": res["note"],
                        "fig_bytes": res["fig_bytes"]
                    })
                    st.toast("『尺度信頼性分析結果』をレポートリストに追加しました！", icon="📋")
                    st.rerun()

                c_l, c_r = st.columns([2.5, 2])
                with c_l:
                    st.write(f"**項目統計量および項目削除時クロンバックのα係数 (全体 α = {res['alpha_val']:.3f})**")
                    st.dataframe(res["rel_df"], use_container_width=True)
                    st.caption(res["note"])
                with c_r:
                    st.write("**項目削除時αプロット**")
                    with st.container(height=450):
                        st.image(res["fig_bytes"], use_column_width=True)

        # 13. 共分散構造分析 / パス解析 (SEM)
        elif "13. 共分散構造分析" in stat_method_4:
            st.subheader("13. 共分散構造分析 / パス解析 (SEM: Structural Equation Modeling)")
            st.markdown("""
            **直感的な「矢印（From ➔ To）」のペアリングで、媒介分析・パス解析から潜在変数を含む共分散構造分析まで自由にモデリングできます。**
            - **潜在変数（因子）の作成**: 複数のアンケート項目をまとめる因子を定義（省略して観測変数のみのパス解析も可能）
            - **構造モデル（パス）の指定**: 原因変数（From）➔ 結果変数（To）の矢印をペアで追加
            - **共分散（相関）**: 原因となる外生変数同士の相関は自動考慮（個別指定も可能）
            """)

            # モデルプリセット読み込み (サンプルデータ等向け)
            c_pre1, c_pre2, _ = st.columns([1.5, 1.2, 2])
            with c_pre1:
                if st.button("💡 演習用 媒介モデルを設定 (意欲 ➔ 満足度 ➔ 得点)", key="btn_preset_med"):
                    st.session_state["sem_latents"] = []
                    # 存在する列で設定
                    p_list = []
                    if "Q1_意欲" in df.columns and "学習満足度" in df.columns:
                        p_list.append({"from": "Q1_意欲", "to": "学習満足度"})
                    if "学習満足度" in df.columns and "事後テスト得点" in df.columns:
                        p_list.append({"from": "学習満足度", "to": "事後テスト得点"})
                    if "Q1_意欲" in df.columns and "事後テスト得点" in df.columns:
                        p_list.append({"from": "Q1_意欲", "to": "事後テスト得点"})
                    if not p_list and len(numeric_columns) >= 3:
                        p_list = [
                            {"from": numeric_columns[0], "to": numeric_columns[1]},
                            {"from": numeric_columns[1], "to": numeric_columns[2]},
                            {"from": numeric_columns[0], "to": numeric_columns[2]}
                        ]
                    st.session_state["sem_paths"] = p_list
                    st.session_state["sem_covs"] = []
                    st.toast("演習用 媒介分析モデルを設定しました！", icon="💡")
                    st.rerun()
            with c_pre2:
                if st.button("🧹 モデル設定をクリア", key="btn_clear_sem"):
                    st.session_state["sem_latents"] = []
                    st.session_state["sem_paths"] = []
                    st.session_state["sem_covs"] = []
                    st.toast("モデル設定を初期化しました", icon="🧹")
                    st.rerun()

            st.divider()

            # --- ステップ 1: 潜在変数（因子）の作成 ---
            with st.expander("① 潜在変数（因子）の作成 (任意・複数観測項目の合成)", expanded=len(st.session_state["sem_latents"]) > 0):
                st.caption("アンケートの複数項目（観測変数）から構成される「潜在因子」を定義します。観測変数のみのパス解析を行う場合はスキップできます。")
                c_l1, c_l2, c_l3 = st.columns([1.5, 2.5, 1])
                with c_l1:
                    new_latent_name = st.text_input("潜在変数名 (例: 学習意欲):", key="new_latent_name", placeholder="因子名を入力...")
                with c_l2:
                    new_latent_inds = st.multiselect("構成する観測項目 (数値変数):", numeric_columns, default=[], key="new_latent_inds")
                with c_l3:
                    st.write("")
                    st.write("")
                    if st.button("➕ 潜在変数を追加", key="btn_add_latent", use_container_width=True):
                        if new_latent_name and len(new_latent_inds) >= 2:
                            # 重複チェック
                            existing_names = [l["name"] for l in st.session_state["sem_latents"]]
                            if new_latent_name in existing_names:
                                st.warning("同名の潜在変数が既に存在します。")
                            else:
                                st.session_state["sem_latents"].append({
                                    "name": new_latent_name, "indicators": new_latent_inds
                                })
                                st.toast(f"潜在変数 『{new_latent_name}』 を作成しました！", icon="✨")
                                st.rerun()
                        else:
                            st.warning("潜在変数名と2つ以上の観測項目を指定してください。")

                if st.session_state["sem_latents"]:
                    st.markdown("**📋 登録済み 潜在変数一覧:**")
                    for l_idx, l_data in enumerate(st.session_state["sem_latents"]):
                        c_li1, c_li2 = st.columns([4, 1])
                        with c_li1:
                            st.info(f"🔹 **{l_data['name']}** =~ " + " + ".join([f"`{ind}`" for ind in l_data['indicators']]))
                        with c_li2:
                            if st.button("🗑 削除", key=f"del_latent_{l_idx}", use_container_width=True):
                                st.session_state["sem_latents"].pop(l_idx)
                                st.rerun()

            # --- ステップ 2: 矢印（パス: From ➔ To）の指定 ---
            st.markdown("#### ② 構造モデル（矢印パス: 原因 ➔ 結果）の指定")
            latent_names = [l["name"] for l in st.session_state["sem_latents"]]
            all_choice_vars = latent_names + all_columns

            c_p1, c_p2, c_p3 = st.columns([2, 2, 1.2])
            with c_p1:
                from_v = st.selectbox("原因変数 (From ➔):", all_choice_vars, index=None, placeholder="原因変数を選択...", key="sem_from")
            with c_p2:
                to_options = [c for c in all_choice_vars if c != from_v]
                to_v = st.selectbox("結果変数 (➔ To):", to_options, index=None, placeholder="結果変数を選択...", key="sem_to")
            with c_p3:
                st.write("")
                st.write("")
                if st.button("➕ 矢印 (パス) を追加", key="btn_add_path", type="primary", use_container_width=True):
                    if from_v and to_v:
                        # 重複チェック
                        already = any(p["from"] == from_v and p["to"] == to_v for p in st.session_state["sem_paths"])
                        if already:
                            st.warning("既に同じパスが登録されています。")
                        else:
                            st.session_state["sem_paths"].append({"from": from_v, "to": to_v})
                            st.toast(f"パス 『{from_v} ➔ {to_v}』 を追加しました！", icon="➡️")
                            st.rerun()
                    else:
                        st.warning("From と To の変数をそれぞれ選択してください。")

            # 登録済みパスの表示
            if st.session_state["sem_paths"]:
                st.markdown("**📋 登録されたパス一覧 (構造方程式):**")
                cols_grid = st.columns(3)
                for p_idx, p_item in enumerate(st.session_state["sem_paths"]):
                    with cols_grid[p_idx % 3]:
                        cp_a, cp_b = st.columns([3, 1])
                        with cp_a:
                            st.success(f"**`{p_item['from']}`** ➔ **`{p_item['to']}`**")
                        with cp_b:
                            if st.button("✖", key=f"del_path_{p_idx}"):
                                st.session_state["sem_paths"].pop(p_idx)
                                st.rerun()

            # --- ステップ 3: 共分散（相関 ↔）の設定 ---
            with st.expander("③ 共分散・相関（双方向矢印 ↔）の設定", expanded=False):
                auto_exog_cov = st.checkbox(
                    "☑️ 原因となる外生変数（他の変数から矢印が入らない変数）同士の相関（共分散）を自動的に仮定する (推奨: 標準的SEM仕様)",
                    value=True, key="sem_auto_cov"
                )
                st.caption("※ チェックを入れると、原因変数同士の事前相関が自動的にモデルに含まれます。")
                
                st.markdown("**個別に特定の変数・誤差間の相関（↔）を追加する場合:**")
                c_cov1, c_cov2, c_cov3 = st.columns([2, 2, 1.2])
                with c_cov1:
                    cov_v1 = st.selectbox("変数1 (↔):", all_choice_vars, index=None, placeholder="変数1...", key="sem_cov1")
                with c_cov2:
                    cov_v2_opts = [c for c in all_choice_vars if c != cov_v1]
                    cov_v2 = st.selectbox("変数2 (↔):", cov_v2_opts, index=None, placeholder="変数2...", key="sem_cov2")
                with c_cov3:
                    st.write("")
                    st.write("")
                    if st.button("➕ 相関を追加", key="btn_add_cov", use_container_width=True):
                        if cov_v1 and cov_v2:
                            st.session_state["sem_covs"].append({"var1": cov_v1, "var2": cov_v2})
                            st.toast(f"共分散 『{cov_v1} ↔ {cov_v2}』 を追加しました！", icon="↔️")
                            st.rerun()
                        else:
                            st.warning("2つの変数を選択してください。")

                if st.session_state["sem_covs"]:
                    st.markdown("**個別指定された共分散一覧:**")
                    for c_idx, c_item in enumerate(st.session_state["sem_covs"]):
                        cc1, cc2 = st.columns([4, 1])
                        with cc1:
                            st.caption(f"↔️ `{c_item['var1']}` ↔ `{c_item['var2']}`")
                        with cc2:
                            if st.button("✖", key=f"del_cov_{c_idx}"):
                                st.session_state["sem_covs"].pop(c_idx)
                                st.rerun()

            st.divider()

            # --- 分析実行 ---
            if st.button("🚀 共分散構造分析 (SEM) / パス解析を実行", key="run_sem", type="primary"):
                if st.session_state["sem_paths"]:
                    try:
                        with st.spinner("共分散構造分析モデルの最尤推定・適合度計算・パス図描画を実行中..."):
                            param_df, fit_df, indirect_df, note, fig_bytes, model_meta = stats_engine.analyze_sem(
                                df,
                                latent_defs=st.session_state["sem_latents"],
                                paths=st.session_state["sem_paths"],
                                covariances=st.session_state["sem_covs"],
                                auto_exogenous_cov=auto_exog_cov
                            )
                            st.session_state["results"]["sem"] = {
                                "param_df": param_df, "fit_df": fit_df, "indirect_df": indirect_df,
                                "note": note, "fig_bytes": fig_bytes, "model_meta": model_meta
                            }
                        st.toast("共分散構造分析 (SEM) の推定が完了しました！", icon="🎉")
                    except Exception as e:
                        st.error(f"SEM分析エラー: {e}")
                else:
                    st.warning("1つ以上の矢印（パス）を追加してください。")

            # --- 結果の表示 ---
            if "sem" in st.session_state["results"]:
                res = st.session_state["results"]["sem"]
                model_meta = res.get("model_meta", {})
                
                st.markdown("### 📊 分析結果サマリー")
                
                # 適合度のハイライト表示
                cfi_val = res["fit_df"].loc["CFI (適合度指数)", "値"] if "CFI (適合度指数)" in res["fit_df"].index else 0.95
                rmsea_val = res["fit_df"].loc["RMSEA (二乗平均平方根誤差)", "値"] if "RMSEA (二乗平均平方根誤差)" in res["fit_df"].index else 0.05
                chi_df_val = res["fit_df"].loc["χ²/df 比", "値"] if "χ²/df 比" in res["fit_df"].index else 1.5
                
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("CFI (適合度指数)", f"{cfi_val:.3f}", delta="良好 (≥ .95)" if cfi_val >= 0.95 else ("許容 (≥ .90)" if cfi_val >= 0.90 else "要改善"))
                with m2:
                    st.metric("RMSEA (誤差二乗平均)", f"{rmsea_val:.3f}", delta="優秀 (≤ .05)" if rmsea_val <= 0.05 else ("良好 (≤ .08)" if rmsea_val <= 0.08 else "不良"), delta_color="inverse")
                with m3:
                    st.metric("χ² / df 比", f"{chi_df_val:.2f}", delta="良好 (< 2.0)" if chi_df_val < 2.0 else ("許容 (< 3.0)" if chi_df_val < 3.0 else "不良"), delta_color="inverse")

                c_res_l, c_res_r = st.columns([2.5, 2.5])
                with c_res_l:
                    st.markdown("**1. パラメータ推定値表 (APA Style)**")
                    st.dataframe(res["param_df"], use_container_width=True)
                    st.caption(res["note"])
                    
                    st.markdown("**2. モデル適合度指標一覧**")
                    st.dataframe(res["fit_df"], use_container_width=True)
                    
                    if not res["indirect_df"].empty:
                        st.markdown("**3. 媒介効果・間接効果の分解**")
                        st.dataframe(res["indirect_df"], use_container_width=True)
                        st.caption("注. 総合効果 = 直接効果 + 間接効果。媒介比率は総合効果に対する間接効果の割合を表す。")

                with c_res_r:
                    st.markdown("**📐 パスダイアグラム (Path Diagram)**")
                    
                    # パス図のカスタマイズ設定
                    with st.expander("🎨 パス図の表示カスタマイズ設定", expanded=False):
                        cc_1, cc_2 = st.columns(2)
                        with cc_1:
                            theme_choice = st.selectbox(
                                "配色テーマ:",
                                ["APA 標準 (ネイビー/ブルー)", "APA クラシック (白黒/グレー)", "ソフトカラー (パステル)"],
                                key="sem_theme"
                            )
                            coef_choice = st.selectbox(
                                "表示する係数:",
                                ["標準化係数 (β)", "非標準化係数 (B)"],
                                key="sem_coef_type"
                            )
                            show_r2_opt = st.checkbox("決定係数 (R²) を表示", value=True, key="sem_show_r2")
                        with cc_2:
                            font_sz = st.slider("文字サイズ (pt):", min_value=8, max_value=14, value=10, key="sem_font_sz")
                            fig_w = st.slider("図の横幅 (インチ):", min_value=8.0, max_value=14.0, value=10.0, step=0.5, key="sem_fig_w")
                            fig_h = st.slider("図の縦幅 (インチ):", min_value=4.5, max_value=8.5, value=6.0, step=0.5, key="sem_fig_h")
                            show_ns_opt = st.checkbox("非有意なパス (点線) も表示", value=True, key="sem_show_ns")

                    # カスタマイズ設定に基づきリアルタイム再描画
                    # model_meta が前回のセッション等で欠落していても確実に復元
                    if not model_meta:
                        param_df_temp = res.get("param_df", pd.DataFrame())
                        fb_paths = []
                        fb_exog = set()
                        fb_endog = set()
                        for idx_val in param_df_temp.index:
                            if " → " in str(idx_val):
                                f_part, t_part = str(idx_val).split(" → ")
                                fb_paths.append({"from": f_part, "to": t_part})
                                fb_exog.add(f_part)
                                fb_endog.add(t_part)
                        fb_latents = [l["name"] for l in st.session_state.get("sem_latents", [])]
                        fb_all = list(set(fb_exog | fb_endog))
                        fb_pure_exog = list(fb_exog - fb_endog)
                        model_meta = {
                            "latent_names": fb_latents,
                            "paths": fb_paths if fb_paths else st.session_state.get("sem_paths", []),
                            "r2_map": {},
                            "exogenous_vars": fb_pure_exog,
                            "all_model_vars": fb_all
                        }

                    curr_fig_bytes = stats_engine.render_sem_path_diagram(
                        param_df=res["param_df"],
                        fit_df=res["fit_df"],
                        latent_names=model_meta.get("latent_names", []),
                        paths=model_meta.get("paths", []),
                        r2_map=model_meta.get("r2_map", {}),
                        exogenous_vars=model_meta.get("exogenous_vars", []),
                        all_model_vars=model_meta.get("all_model_vars", []),
                        show_r2=show_r2_opt,
                        font_size=font_sz,
                        fig_width=fig_w,
                        fig_height=fig_h,
                        color_theme=theme_choice,
                        show_fit_footer=True,
                        coef_type=coef_choice,
                        show_insignificant=show_ns_opt
                    )

                    # 動的にサイズ変更が反映されるように直接描画
                    st.image(curr_fig_bytes, use_container_width=True)

                    c_dl_img, _ = st.columns([2, 1])
                    with c_dl_img:
                        st.download_button(
                            label="📥 パス図を高解像度画像で保存 (.png)",
                            data=curr_fig_bytes,
                            file_name="SEM_パスダイアグラム.png",
                            mime="image/png",
                            use_container_width=True
                        )

                st.divider()
                if st.button("➕ この結果をExcelレポートに追加 (適合度表 & パラメータ表 & パス図)", key="btn_add_sem", type="primary"):
                    st.session_state["analysis_queue"].append({
                        "sheet_name": "共分散構造分析",
                        "title": "共分散構造分析 (SEM) パラメータ推定値一覧表",
                        "df": res["param_df"],
                        "note": res["note"],
                        "extra_title": "モデル適合度指標サマリー",
                        "extra_df": res["fit_df"],
                        "extra_title_2": "媒介効果・間接効果の分解",
                        "extra_df_2": res["indirect_df"],
                        "fig_bytes": curr_fig_bytes
                    })
                    st.toast("『共分散構造分析 (SEM) 結果』をレポートリストに追加しました！", icon="📋")
                    st.rerun()



if __name__ == "__main__":
    main()
