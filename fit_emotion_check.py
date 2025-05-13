import streamlit as st
import os
import tempfile
import zipfile
from fitparse import FitFile
import pandas as pd
import numpy as np
from datetime import timedelta
import io
import folium
from streamlit_folium import st_folium

st.set_page_config(layout="wide")

# 初始化 session_state
if 'page' not in st.session_state:
    st.session_state.page = 'fit_upload'
if 'df_minute' not in st.session_state:
    st.session_state.df_minute = None
if 'df_all' not in st.session_state:
    st.session_state.df_all = None
if 'df_summary' not in st.session_state:
    st.session_state.df_summary = None
if 'all_coords' not in st.session_state:
    st.session_state.all_coords = None
if 'summary_list' not in st.session_state:
    st.session_state.summary_list = None
if 'csv_filename_all' not in st.session_state:
    st.session_state.csv_filename_all = "df_all.csv"
if 'csv_filename_minute' not in st.session_state:
    st.session_state.csv_filename_minute = "df_all_minute.csv"
if 'valid_files' not in st.session_state:
    st.session_state.valid_files = {}

# 檢查 ZIP 檔案是否有效
def is_valid_zip(file):
    """檢查 ZIP 檔案是否有效"""
    try:
        with zipfile.ZipFile(file) as zip_ref:
            zip_ref.testzip()  # 檢查 ZIP 檔案完整性
        return True
    except zipfile.BadZipFile:
        return False

# 遞迴解壓縮 ZIP 檔案並收集 .fit 檔案
def extract_zip_recursive(zip_file, extract_path, fit_files):
    """遞迴解壓縮 ZIP 檔案並收集 .fit 檔案"""
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            for file_info in zip_ref.infolist():
                file_path = os.path.join(extract_path, file_info.filename)
                # 忽略 macOS 隱藏檔案和資料夾
                if file_info.filename.startswith('__MACOSX/') or file_info.filename.startswith('._'):
                    continue
                # 檢查是否為 ZIP 檔案
                if file_info.filename.endswith('.zip') and os.path.isfile(file_path):
                    if is_valid_zip(file_path):
                        # 遞迴解壓縮內嵌 ZIP
                        nested_extract_path = os.path.join(extract_path, os.path.basename(file_info.filename).replace('.zip', ''))
                        os.makedirs(nested_extract_path, exist_ok=True)
                        extract_zip_recursive(file_path, nested_extract_path, fit_files)
                # 檢查是否為 .fit 檔案
                elif file_info.filename.endswith('.fit'):
                    fit_files.append(file_path)
    except zipfile.BadZipFile as e:
        st.error(f"無法解壓縮檔案 {zip_file.name}：檔案損壞或格式錯誤")
    except Exception as e:
        st.error(f"處理檔案 {zip_file.name} 時發生錯誤：{str(e)}")

# 第一頁：FIT 檔案上傳與處理
if st.session_state.page == 'fit_upload':
    st.title("\U0001F4CA FIT 檔案分析工具")

    uploaded_files = st.file_uploader("請上傳包含 .fit 檔案的 ZIP 資料夾或單個 .fit 檔案", type=["zip", "fit"], accept_multiple_files=True)

    if uploaded_files:
        temp_dir = tempfile.TemporaryDirectory()
        fit_files = []

        for uploaded_file in uploaded_files:
            if uploaded_file.name.endswith('.zip'):
                if not is_valid_zip(uploaded_file):
                    st.error("\U0001F6A8 上傳的 ZIP 檔案無效或損壞，請檢查檔案並重新上傳")
                    temp_dir.cleanup()
                    st.stop()
                # 遞迴解壓縮 ZIP 檔案
                extract_zip_recursive(uploaded_file, temp_dir.name, fit_files)
                zip_name = uploaded_file.name.replace(".zip", "")
                parts = zip_name.split("-")
                if len(parts) >= 2:
                    st.session_state.csv_filename_all = f"{parts[0]}_{parts[1]}_df_all.csv"
                    st.session_state.csv_filename_minute = f"{parts[0]}_{parts[1]}_df_all_minute.csv"
                else:
                    st.session_state.csv_filename_all = "df_all.csv"
                    st.session_state.csv_filename_minute = "df_all_minute.csv"
            elif uploaded_file.name.endswith('.fit'):
                # 儲存單個 .fit 檔案到臨時目錄
                fit_path = os.path.join(temp_dir.name, uploaded_file.name)
                with open(fit_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                fit_files.append(fit_path)
                # 設定檔案名稱
                parts = uploaded_file.name.replace(".fit", "").split("-")
                if len(parts) >= 2:
                    st.session_state.csv_filename_all = f"{parts[0]}_{parts[1]}_df_all.csv"
                    st.session_state.csv_filename_minute = f"{parts[0]}_{parts[1]}_df_all_minute.csv"
                else:
                    st.session_state.csv_filename_all = "df_all.csv"
                    st.session_state.csv_filename_minute = "df_all_minute.csv"

        if not fit_files:
            st.error("\U0001F6A8 未找到任何 .fit 檔案")
            temp_dir.cleanup()
            st.stop()

        summary_list = []
        long_records = []
        all_coords = []

        for fit_path in fit_files:
            fit_filename = os.path.basename(fit_path)
            fit_file = FitFile(fit_path)

            total_distance = next(
                (f.value for m in fit_file.get_messages('session') for f in m.fields if f.name == 'total_distance'), None
            )
            distance_gt_2000 = total_distance > 2000 if total_distance else False
            has_gps = any(m.name == 'gps_metadata' for m in fit_file.get_messages())
            has_hrv = any(m.name == 'hrv' for m in fit_file.get_messages())

            taiwan_time = None
            for m in fit_file.get_messages('record'):
                ts_field = next((f for f in m.fields if f.name == 'timestamp'), None)
                if ts_field and ts_field.value:
                    taiwan_time = ts_field.value + timedelta(hours=8)
                    break
            is_weekday = taiwan_time.weekday() <= 4 if taiwan_time else None

            is_valid = (distance_gt_2000 and has_gps and has_hrv and is_weekday)
            # 初始化 checkbox 狀態，預設不合格檔案為 unchecked
            if fit_filename not in st.session_state.valid_files:
                st.session_state.valid_files[fit_filename] = is_valid

            summary_list.append({
                'file': fit_filename,
                'total_distance': total_distance,
                'distance_gt_2000': distance_gt_2000,
                'has_gps': has_gps,
                'has_hrv': has_hrv,
                'timestamp': taiwan_time,
                'is_weekday': is_weekday
            })

            # 處理所有檔案的數據，不僅限於合格檔案
            hrv_queue = []
            gps_queue = []
            gps_coords = []

            for msg in fit_file.get_messages():
                if msg.name == 'hrv':
                    for field in msg.fields:
                        if field.name == 'time':
                            for rr in field.value:
                                hrv_queue.append(None if rr == 65535 else rr)

                elif msg.name == 'gps_metadata':
                    gps_entry = {f.name: f.value for f in msg.fields}
                    gps_queue.append(gps_entry)

                elif msg.name == 'record':
                    record_data = {f.name: f.value for f in msg.fields}

                    ts = record_data.get('timestamp')
                    record_data['timestamp_taiwan'] = ts + timedelta(hours=8) if ts else None
                    record_data['is_weekday'] = record_data['timestamp_taiwan'].weekday() <= 4 if ts else None
                    record_data['minute'] = record_data['timestamp_taiwan'].replace(second=0, microsecond=0) if ts else None

                    if hrv_queue:
                        record_data['hrv'] = hrv_queue[:5]
                        hrv_queue = hrv_queue[5:]
                    else:
                        record_data['hrv'] = None

                    if gps_queue:
                        gps_data = gps_queue.pop(0)
                        record_data['enhanced_altitude'] = gps_data.get('enhanced_altitude')
                        record_data['enhanced_speed'] = gps_data.get('enhanced_speed')
                    else:
                        record_data['enhanced_altitude'] = None
                        record_data['enhanced_speed'] = None

                    lat = record_data.get('position_lat')
                    lon = record_data.get('position_long')
                    if lat is not None and lon is not None:
                        latitude = lat * (180 / 2**31)
                        longitude = lon * (180 / 2**31)
                        record_data['Latitude'] = latitude
                        record_data['Longitude'] = longitude
                        gps_coords.append([latitude, longitude])
                    else:
                        record_data['Latitude'] = None
                        record_data['Longitude'] = None

                    record_data['ID'] = fit_filename
                    record_data['total_distance'] = total_distance
                    record_data['distance_gt_2000']: distance_gt_2000
                    record_data['has_gps'] = has_gps
                    record_data['has_hrv'] = has_hrv

                    long_records.append(record_data)

            if gps_coords:
                all_coords.append((fit_filename, gps_coords))

        df_summary = pd.DataFrame(summary_list)
        df_all = pd.DataFrame(long_records)

        if df_all.empty:
            st.error("\u274C 無符合條件的 FIT 檔案")
            temp_dir.cleanup()
            st.stop()

        def filter_rr_intervals(rr):
            if isinstance(rr, list) and len(rr) > 1:
                rr_filtered = [x for x in rr if 500 <= x <= 1200]
                return rr_filtered if len(rr_filtered) > 1 else None
            return None

        df_all['hrv_ms'] = df_all['hrv'].apply(
            lambda rr: [x * 1000 for x in rr if x is not None] if isinstance(rr, list) else None
        )
        df_all['hrv_ms_filtered'] = df_all['hrv_ms'].apply(filter_rr_intervals)

        df_all['minute'] = df_all['timestamp_taiwan'].dt.floor('T')

# 除錯
        required_cols = [
            'ID', 'minute', 'hrv_ms_filtered',
            'heart_rate', 'enhanced_speed', 'enhanced_altitude',
            'temperature', 'Latitude', 'Longitude'
        ]
        missing_cols = [col for col in required_cols if col not in df_all.columns]
        for col in missing_cols:
            df_all[col] = np.nan
        if missing_cols:
            st.warning(f"⚠️ 下列欄位在 df_all 中缺失，已自動補齊為 NA：{missing_cols}")


        def get_middle_point(coords):
            if coords and isinstance(coords, list) and len(coords) > 0:
                coords = [c for c in coords if c is not None]
                if len(coords) == 0:
                    return None
                mid_idx = len(coords) // 2
                return coords[mid_idx]
            return None

        df_minute = df_all.groupby(['ID', 'minute']).agg({
            'hrv_ms_filtered': lambda x: [item for sublist in x if isinstance(sublist, list) for item in sublist],
            'heart_rate': ['min', 'max', 'mean'],
            'enhanced_speed': ['min', 'max', 'mean'],
            'enhanced_altitude': ['min', 'max', 'mean'],
            'temperature': 'mean',
            'Latitude': lambda x: get_middle_point(x.tolist()),
            'Longitude': lambda x: get_middle_point(x.tolist())
        }).reset_index()

        df_minute.columns = [
            'ID', 'minute',
            'hrv_ms_filtered',
            'heart_rate_min', 'heart_rate_max', 'heart_rate_mean',
            'enhanced_speed_min', 'enhanced_speed_max', 'enhanced_speed_mean',
            'enhanced_altitude_min', 'enhanced_altitude_max', 'enhanced_altitude_mean',
            'temperature_mean',
            'Latitude', 'Longitude'
        ]

        def calculate_hrv_metrics(rr):
            if isinstance(rr, list) and len(rr) > 1:
                return {
                    'sdnn': np.std(rr),
                    'sdsd': np.sqrt(np.mean(np.diff(rr)**2) - np.mean(np.diff(rr))**2),
                    'rmssd': np.sqrt(np.sum(np.diff(rr)**2) / (len(rr) - 1)),
                    'pnn50': np.sum(np.abs(np.diff(rr)) > 50) / (len(rr) - 1) * 100,
                    'heart_rate_hrv': 60 / (np.mean(rr) / 1000) if np.mean(rr) > 0 else None
                }
            return {'sdnn': None, 'sdsd': None, 'rmssd': None, 'pnn50': None, 'heart_rate_hrv': None}

        hrv_metrics = df_minute['hrv_ms_filtered'].apply(calculate_hrv_metrics)
        df_hrv = pd.DataFrame(hrv_metrics.tolist(), index=df_minute.index)
        df_minute = pd.concat([df_minute, df_hrv], axis=1)

        df_minute['valid'] = df_minute['hrv_ms_filtered'].apply(
            lambda rr: sum(1 for x in rr if x < 500 or x > 1200) <= 1 if isinstance(rr, list) else False
        )
        df_minute = df_minute[df_minute['valid']].drop(columns=['valid', 'hrv_ms_filtered'])

        # 儲存完整數據以便後續動態篩選
        st.session_state.df_all_full = df_all
        st.session_state.df_minute_full = df_minute

        # 根據當前 valid_files 篩選數據
        selected_files = [f for f in st.session_state.valid_files if st.session_state.valid_files[f]]
        df_all = df_all[df_all['ID'].isin(selected_files)]
        df_minute = df_minute[df_minute['ID'].isin(selected_files)]
        all_coords = [(name, coords) for name, coords in all_coords if name in selected_files]

        st.session_state.df_minute = df_minute
        st.session_state.df_all = df_all
        st.session_state.df_summary = df_summary
        st.session_state.all_coords = all_coords
        st.session_state.summary_list = summary_list

        st.subheader("\U0001F4CB FIT 摘要資訊")
        # 準備顯示的 DataFrame，包含 checkbox 欄位
        display_summary = df_summary.rename(columns={
            'file': '檔名',
            'total_distance': '總距離',
            'distance_gt_2000': '是否超過2公里',
            'has_gps': '是否有GPS資料',
            'has_hrv': '是否有HRV資料',
            'timestamp': '時間',
            'is_weekday': '是否為平日'
        }).copy()
        display_summary['有效'] = display_summary['檔名'].map(st.session_state.valid_files)

        # 使用 st.data_editor 顯示表格並允許編輯 checkbox
        edited_df = st.data_editor(
            display_summary,
            column_config={
                '有效': st.column_config.CheckboxColumn(
                    '有效',
                    help="勾選表示該檔案有效",
                    default=True
                )
            },
            disabled=[col for col in display_summary.columns if col != '有效'],
            hide_index=True,
            key="summary_editor"
        )

        # 同步編輯後的 checkbox 狀態到 st.session_state.valid_files
        for idx, row in edited_df.iterrows():
            st.session_state.valid_files[row['檔名']] = row['有效']

        alarm_msgs = []
        for row in summary_list:
            ts = row["timestamp"]
            if not row["distance_gt_2000"]:
                alarm_msgs.append(f"\U0001F6A8 {row['file']} 的資料沒有超過2公里")
            if not row["has_gps"]:
                alarm_msgs.append(f"\U0001F6A8 {row['file']} 的資料沒有 GPS")
            if not row["has_hrv"]:
                alarm_msgs.append(f"\U0001F6A8 {row['file']} 的資料沒有 HRV")
            if not row["is_weekday"]:
                alarm_msgs.append(f"\u26a0\ufe0f {row['file']} 的資料非平日")

        if alarm_msgs:
            st.subheader("\U0001F514 無效路徑")
            for msg in alarm_msgs:
                st.warning(msg)

        valid_records = [row for row in summary_list if st.session_state.valid_files[row['file']]]

        st.markdown(
            f"\u2705 初步有效紀錄為 <span style='font-weight:bold; color:red;'><strong>{len(valid_records)}</strong> 筆</span>，後續仍需確認是否無重複起訖點，實際有效紀錄依本研究室信件通知為準",
            unsafe_allow_html=True
        )

        st.subheader("\U0001F4E5 下載資料")
        # 根據當前 valid_files 篩選完整數據進行下載
        selected_files = [f for f in st.session_state.valid_files if st.session_state.valid_files[f]]
        df_all_download = st.session_state.df_all_full[st.session_state.df_all_full['ID'].isin(selected_files)]
        df_minute_download = st.session_state.df_minute_full[st.session_state.df_minute_full['ID'].isin(selected_files)]

        csv_buffer_all = io.StringIO()
        df_all_download.to_csv(csv_buffer_all, index=False)
        st.download_button(
            label=f"\U0001F4C5 下載原始數據 ({st.session_state.csv_filename_all})",
            data=csv_buffer_all.getvalue(),
            file_name=st.session_state.csv_filename_all,
            mime="text/csv"
        )

        csv_buffer_minute = io.StringIO()
        df_minute_download.to_csv(csv_buffer_minute, index=False)
        st.download_button(
            label=f"\U0001F4C5 下載每分鐘數據 ({st.session_state.csv_filename_minute})",
            data=csv_buffer_minute.getvalue(),
            file_name=st.session_state.csv_filename_minute,
            mime="text/csv"
        )

        st.subheader("🗺️ GPS 路徑地圖")
        map_options = [name for name, _ in all_coords]
        selected_maps = st.multiselect("選擇要顯示的路徑：", map_options, default=map_options)

        if any(len(coords) > 0 for name, coords in all_coords if name in selected_maps):
            first_valid = next((coords for name, coords in all_coords if name in selected_maps and coords), None)
            m = folium.Map(location=first_valid[0], zoom_start=13, tiles=None)

            folium.TileLayer('CartoDB positron', name='Light Map', control=False).add_to(m)

            import hashlib
            import colorsys

            def name_to_color(name):
                hash_digest = hashlib.md5(name.encode()).hexdigest()
                hue = int(hash_digest[:2], 16) / 255
                rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                return '#%02x%02x%02x' % tuple(int(c * 255) for c in rgb)

            for name, coords in all_coords:
                if name in selected_maps:
                    color = name_to_color(name)

                    # 畫線
                    folium.PolyLine(coords, tooltip=name, color=color).add_to(m)

                    # 起點：藍色、info-sign
                    if coords:
                        folium.Marker(
                            coords[0],
                            popup=f"{name} 起點",
                            icon=folium.Icon(icon="play", color="blue")
                        ).add_to(m)

                    # 終點：紅色、stop
                    if coords:
                        folium.Marker(
                            coords[-1],
                            popup=f"{name} 終點",
                            icon=folium.Icon(icon="stop", color="red")
                        ).add_to(m)

            st_folium(m, width=800, height=500)
        else:
            st.warning("❗ 無 GPS 座標可顯示（position_lat / position_long 為 None）")

        if st.button("開始處理情緒資料 (研究者)"):
            st.session_state.page = 'emotion_upload'

        temp_dir.cleanup()

# 第二頁：情緒資料上傳與處理
elif st.session_state.page == 'emotion_upload':
    st.title("📂 合併 FIT 與情緒問卷")

    # 注入 CSS 樣式以放大 checkbox 字體
    st.markdown("""
        <style>
        .stCheckbox > label {
            font-size: 18px !important;  /* 放大 checkbox 標籤字體 */
            margin-bottom: 5px;  /* 增加與下方警訊的間距 */
        }
        </style>
    """, unsafe_allow_html=True)

    # 初始化 session_state 用於追蹤每個 ID 的 valid 狀態
    if 'valid_ids' not in st.session_state:
        st.session_state.valid_ids = {}
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'alerts_dict' not in st.session_state:
        st.session_state.alerts_dict = {}
    if 'merged_all_list' not in st.session_state:
        st.session_state.merged_all_list = []
    if 'all_emotion_data' not in st.session_state:
        st.session_state.all_emotion_data = []
    if 'invalid_ids' not in st.session_state:
        st.session_state.invalid_ids = set()

    uploaded_emotion_zip = st.file_uploader("📦 上傳包含多個 .csv 的 ZIP 檔", type="zip")

    if uploaded_emotion_zip and st.button("🚀 開始處理"):
        if st.session_state.df_minute is None:
            st.error("❌ 無有效的 df_minute 資料，請先完成 FIT 檔案處理")
            st.stop()

        df_minute = st.session_state.df_minute.copy()
        df_all = st.session_state.df_all.copy()
        df_minute['minute'] = pd.to_datetime(df_minute['minute']).dt.tz_localize(None).dt.floor('min')

        temp_dir_emotion = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(uploaded_emotion_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir_emotion.name)

        all_emotion_data = []
        merged_raw_all = []
        merged_all_list = []
        alerts_dict = {}
        valid_file_ids = set(df_minute['ID'].unique())
        invalid_ids = set()

        for file_name in os.listdir(temp_dir_emotion.name):
            if file_name.endswith(".csv"):
                file_path = os.path.join(temp_dir_emotion.name, file_name)
                try:
                    df = pd.read_csv(file_path)
                    first_row = df.iloc[0]
                    first_id = first_row['ID']
                    first_name = first_row['Name']

                    first_different_index = -1
                    for index, row in df.iloc[1:].iterrows():
                        if row['ID'] != first_id or row['Name'] != first_name:
                            first_different_index = index
                            break

                    if first_different_index == -1:
                        continue

                    start_index = first_different_index
                    end_index = min(start_index + 13, len(df) - 1)

                    survey_original = df.iloc[start_index:end_index+1][['ID', 'Name']].copy()
                    survey_transposed = survey_original.T
                    survey_transposed.columns = [f'col_{i}' for i in range(len(survey_transposed.columns))]

                    len_val = first_different_index
                    rows_to_concat = [survey_transposed.iloc[1].values] * len_val
                    df_to_concat = pd.DataFrame(rows_to_concat, columns=survey_transposed.iloc[0].values)

                    df_head = df.iloc[:first_different_index].copy()
                    merged_df = pd.concat([df_head, df_to_concat], axis=1)
                    merged_df['FileID'] = file_name.replace(".csv", "")

                    if 'Time' in merged_df.columns:
                        merged_df['Time'] = pd.to_datetime(merged_df['Time'], errors='coerce')
                        merged_df['Time_TW'] = (merged_df['Time'] + timedelta(hours=8)).dt.tz_localize(None).dt.floor('min')

                    all_emotion_data.append(merged_df.copy())
                    merged_raw_all.append(merged_df)

                except Exception as e:
                    st.warning(f"處理檔案 {file_name} 發生錯誤：{e}")

        if all_emotion_data:
            st.subheader("📃 所有合併後的情緒問卷")
            df_all_emotion = pd.concat(all_emotion_data, ignore_index=True)
            st.dataframe(df_all_emotion)
            st.session_state.all_emotion_data = all_emotion_data

        if merged_raw_all:
            merged_all_df = pd.concat(merged_raw_all, ignore_index=True)
            merged_all_df['Time_TW'] = pd.to_datetime(merged_all_df['Time_TW']).dt.tz_localize(None).dt.floor('min')

            # 初始化 valid_ids，預設所有 ID 為 valid
            for id_val in df_minute['ID'].unique():
                if id_val not in st.session_state.valid_ids:
                    st.session_state.valid_ids[id_val] = True

            for id_val in df_minute['ID'].unique():
                df_minute_part = df_minute[df_minute['ID'] == id_val].copy()
                df_minute_part['minute'] = pd.to_datetime(df_minute_part['minute']).dt.tz_localize(None).dt.floor('min')

                merged = pd.merge(
                    df_minute_part,
                    merged_all_df,
                    how="left",
                    left_on="minute",
                    right_on="Time_TW",
                    suffixes=('', '_merged')
                )

                # 檢查資料問題並記錄警訊
                alerts = []
                is_valid = True
                if 'Q1' in merged.columns:
                    q1_series = merged['Q1']
                    q1_clean = pd.to_numeric(q1_series, errors='coerce')
                    fit_id = merged['ID'].iloc[0] if 'ID' in merged.columns else id_val

                    if q1_clean.isna().rolling(window=5, min_periods=5).sum().ge(5).any():
                        alerts.append(f"⚠️ 檔案 {fit_id} 出現連續 5 個 NA")
                        is_valid = False
                    if q1_clean.head(5).isna().all() or q1_clean.tail(5).isna().all():
                        alerts.append(f"⚠️ 檔案 {fit_id} 問卷可能填答不完整（前或後 5 筆為 NA）")
                        is_valid = False

                alerts_dict[id_val] = alerts
                if not is_valid:
                    invalid_ids.add(id_val)

                # 儲存 merged 資料
                merged_all_list.append((id_val, merged))

            st.session_state.merged_all_list = merged_all_list
            st.session_state.alerts_dict = alerts_dict
            st.session_state.invalid_ids = invalid_ids
            st.session_state.processed_data = True

        temp_dir_emotion.cleanup()

    # 顯示處理後的資料
    if st.session_state.processed_data:
        # 顯示每個 ID 的 DataFrame 及其警訊
        for id_val, merged in st.session_state.merged_all_list:
            # 顯示 checkbox 控制是否 valid
            st.session_state.valid_ids[id_val] = st.checkbox(
                f"有效資料 (ID = {id_val})",
                value=st.session_state.valid_ids[id_val],
                key=f"valid_{id_val}"
            )

            # 顯示警訊（位於 checkbox 下方）
            if id_val in st.session_state.alerts_dict and st.session_state.alerts_dict[id_val]:
                st.markdown(f"**🔎資料檢驗 (ID = {id_val})**")
                for msg in st.session_state.alerts_dict[id_val]:
                    st.warning(msg)
            else:
                st.success(f"ID = {id_val} 所有檢核項目皆通過！")

            # 如果 checkbox 被勾選（valid），則顯示 DataFrame
            if st.session_state.valid_ids[id_val]:
                st.subheader(f"📜 ID = {id_val}")
                st.dataframe(merged)

        # 計算最終有效 ID（僅包含 checkbox 勾選的 ID）
        final_valid_ids = {id_val for id_val in st.session_state.valid_ids if st.session_state.valid_ids[id_val]}

        # 顯示最終有效樣本數
        st.markdown(
            f"\u2705 最終有效樣本：<span style='font-weight:bold; color:red;'><strong>{len(final_valid_ids)}</strong> 筆</span>",
            unsafe_allow_html=True
        )

        # 顯示被排除的樣本 ID（僅包含 checkbox 未勾選的 ID）
        excluded_ids = {id_val for id_val in st.session_state.valid_ids if not st.session_state.valid_ids[id_val]}
        if excluded_ids:
            st.markdown("### \u274C 被排除的樣本 ID")
            for eid in sorted(excluded_ids):
                st.markdown(f"- {eid}")

        # 準備有效資料（僅包含 checkbox 勾選的 ID）
        df_all_valid = st.session_state.df_all[st.session_state.df_all['ID'].isin(final_valid_ids)].copy()
        df_minute_valid = st.session_state.df_minute[st.session_state.df_minute['ID'].isin(final_valid_ids)].copy()
        final_result = pd.concat(
            [merged for id_val, merged in st.session_state.merged_all_list if id_val in final_valid_ids],
            ignore_index=True
        )

        if not df_all_valid.empty and not df_minute_valid.empty:
            first_id = df_all_valid["ID"].iloc[0]  # e.g., "177-0422-15.fit"
            person_id = first_id.split("-")[0]     # e.g., "177"

            # 檢查是否有多個不同的 PERSONID
            all_person_ids = df_all_valid["ID"].apply(lambda x: x.split("-")[0]).unique()
            if len(all_person_ids) > 1:
                id_list_str = ", ".join(all_person_ids)
                st.warning(
                    f"⚠️ 資料中發現多個 PERSONID：{id_list_str}。將使用第一個 PERSONID ({person_id}) 作為檔名。"
                )

        # 顯示下載合併結果（需 person_id）
        if not final_result.empty:
            valid_count = len(final_valid_ids)
            st.markdown(f"📊 已選 <strong>{valid_count}</strong> 筆有效資料", unsafe_allow_html=True)

            csv = final_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 下載所有合併資料",
                data=csv,
                file_name=f"{person_id}_merged_emotion.csv",
                mime="text/csv",
                key="download_all_merged"
            )

        # 顯示下載原始資料與每分鐘資料
        if not df_all_valid.empty and not df_minute_valid.empty:
            # 匯出所有有效原始數據
            csv_all_valid = df_all_valid.to_csv(index=False)
            st.download_button(
                label="📥 下載所有有效原始數據",
                data=csv_all_valid,
                file_name=f"{person_id}_df_all_valid.csv",
                mime="text/csv",
                key="download_all_valid"
            )

            # 匯出所有有效每分鐘數據
            csv_minute_valid = df_minute_valid.to_csv(index=False)
            st.download_button(
                label="📥 下載所有有效每分鐘數據",
                data=csv_minute_valid,
                file_name=f"{person_id}_df_minute_valid.csv",
                mime="text/csv",
                key="download_minute_valid"
            )
        else:
            st.warning("❗ 無有效資料可供下載。")

    if st.button("返回 FIT 檔案處理"):
        st.session_state.page = 'fit_upload'