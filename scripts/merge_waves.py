import os
from datetime import datetime, timedelta
import pandas as pd

def parse_dgd_line(line):
    parts = line.split()
    if len(parts) < 30:
        return None
    vals = [float(x) for x in parts]
    year, month, day = int(vals[0]), int(vals[1]), int(vals[2])
    A_planetary = vals[21]
    K_planetary = [float(x) for x in vals[22:30]]
    return {'year': year, 'month': month, 'day': day, 'A_planetary': A_planetary, 'K_planetary': K_planetary}

def build_3hour_dataframe(dgd_daily):
    rows = []
    hours = [0, 3, 6, 9, 12, 15, 18, 21]
    for d in dgd_daily:
        base_date = datetime(d['year'], d['month'], d['day'])
        for i, h in enumerate(hours):
            ts = base_date + timedelta(hours=h)
            rows.append({'timestamp': ts, 'Kp': d['K_planetary'][i]})
    df_3h = pd.DataFrame(rows)
    df_3h.sort_values('timestamp', inplace=True)
    return df_3h

def load_all_dgd_data(project_root):
    data_dir = os.path.join(project_root, "data")
    all_daily = []
    for year in range(2015, 2025):
        file_path = os.path.join(data_dir, f"{year}_DGD.txt")
        if not os.path.exists(file_path):
            continue
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith(':'):
                    continue
                parsed = parse_dgd_line(line)
                if parsed:
                    all_daily.append(parsed)
    if not all_daily:
        return pd.DataFrame(columns=['timestamp', 'Kp'])
    return build_3hour_dataframe(all_daily)

def parse_image_timestamp(filename):
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    try:
        return datetime.strptime(name, "%Y-%m-%dT%H-%M-%SZ")
    except:
        return None

def collect_images(images_dir):
    wave_folders = ["AIA_171", "AIA_193", "AIA_304", "HMI_Magnetogram", "SOHO_LASCO_C2"]
    rows = []
    for wave in wave_folders:
        wave_dir = os.path.join(images_dir, wave)
        if not os.path.isdir(wave_dir):
            continue
        for fname in os.listdir(wave_dir):
            if fname.lower().endswith(".jp2"):
                ts = parse_image_timestamp(fname)
                if ts and ts.year < 2025:
                    rows.append({'wave': wave, 'image_path': os.path.join(wave_dir, fname), 'timestamp': ts})
    if not rows:
        return pd.DataFrame(columns=['wave', 'image_path', 'timestamp'])
    df_images = pd.DataFrame(rows)
    df_images.sort_values('timestamp', inplace=True)
    return df_images

def merge_images_and_dgd(df_images, df_dgd):
    df_images_sorted = df_images.sort_values('timestamp')
    df_dgd_sorted = df_dgd.sort_values('timestamp')
    return pd.merge_asof(df_images_sorted, df_dgd_sorted, on='timestamp', direction='nearest')

def main():
    project_root = os.path.join(os.path.dirname(__file__), "..")
    df_dgd = load_all_dgd_data(project_root)
    print(df_dgd.shape, df_dgd.head())
    df_images = collect_images("/mnt/remote_fits")
    print(df_images.shape, df_images.head())
    df_merged = merge_images_and_dgd(df_images, df_dgd)
    print(df_merged.shape, df_merged.head(20))
    df_merged.to_csv(os.path.join(project_root, "merged_images_kp.csv"), index=False)

if __name__ == "__main__":
    main()
