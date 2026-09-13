import pandas as pd
import os
import re
import orjson
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

def sort_jsons(mainPath, folder,resolutions):
    print("sorting jsons....")
    usePath = mainPath + folder
    #resolutions = ['[452, 144]', '[226, 72]', '[113, 36]', '[57, 18]', '[29, 9]']
    _hold = []
    _452 = []
    _226 = []
    _113 = []
    _57 = []
    _29 = []
    max_len = 4

    for file in os.listdir(usePath):
        if file.endswith(".json"):
            res = '[' + ', '.join(re.findall(r'\d+', file[5:14])) + ']'
            if res in resolutions:
                if res == '[452, 144]'and len(_452)<=max_len:
                    _452.append(file)
                elif res == '[226, 72]'and len(_226)<=max_len:
                    _226.append(file)
                elif res == '[113, 36]'and len(_113)<=max_len:
                    _113.append(file)
                elif res == '[57, 18]'and len(_57)<=max_len:
                    _57.append(file)
                elif res == '[29, 9]'and len(_29)<=max_len:
                    _29.append(file)
                else:
                    _hold.append(file)
    print(folder[:5])
    print("We've got piles! lets see how big...")
    print('452', len(_452))
    print('226', len(_226))
    print('113', len(_113))
    print('57', len(_57))
    print('29', len(_29))
    print('Hold', len(_hold))
    return _452, _226, _113, _57, _29





def _load_json_file(path: Path) -> dict:
    """Read + parse a single JSON file with orjson (fast path)."""
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def _json_to_dataframe(data: dict) -> pd.DataFrame:
    """
    Build a DataFrame from a dict of possibly-uneven-length lists,
    using pd.Series index-alignment so pandas fills the gaps with
    NaN for us instead of a manual max_len + pad-with-None loop.
    """
    list_cols = {k: pd.Series(v) for k, v in data.items() if isinstance(v, list)}
    df = pd.DataFrame(list_cols)
    for k, v in data.items():
        if not isinstance(v, list):
            df[k] = v  # scalar columns broadcast across all rows
    return df


def readin_json(usePath, sorted_list, required_col="TESTAccPeakDist"):
    """
    Walk through sorted_list, load+parse each JSON, build a DataFrame,
    show head/tail, and let the user approve (Y), quit (Q), or skip
    (anything else) to the next file.

    NOTE: this returns at most ONE file's DataFrame (whichever you
    approve). If you want every file merged into one combined
    DataFrame instead, use readin_json_merge_all() below.
    """
    usePath = Path(usePath)
    print("read in jsons....")

    for i, fname in enumerate(sorted_list):
        print(f"[{i}] {fname}")
        data = _load_json_file(usePath / fname)
        df = _json_to_dataframe(data)
        df_clean = df.dropna(subset=[required_col])

        print(df_clean.head())
        print(df_clean.tail())

        choice = input("Y to approve, Q to quit, any other key for next file: ").strip().lower()
        if choice == "y":
            return df_clean
        elif choice == "q":
            print("Quitting...")
            return None
        # anything else -> falls through to next file (fixes the
        # original's "i never increments" infinite-loop bug)

    print("No more files to review.")
    return None


def _load_and_build(args):
    """
    Worker function for the process pool: load one file and turn it
    straight into a cleaned DataFrame. Defined at module level (not
    nested) because ProcessPoolExecutor needs to pickle it.
    """
    usePath, fname, required_col = args
    data = _load_json_file(usePath / fname)
    df = _json_to_dataframe(data)
    if required_col is not None and required_col in df.columns:
        df = df.dropna(subset=[required_col])
    df["source_file"] = fname  # keep provenance — handy once files are merged
    return df


def readin_json_merge_all(usePath, sorted_list, required_col="TESTAccPeakDist",
                           parallel=True, max_workers=2):
    """
    Load every file in sorted_list, build a DataFrame for each, and
    concatenate them all into a single combined DataFrame — no
    interactive approval, every file is included.

    parallel=True (default) parses files across multiple processes,
    since JSON parsing is CPU-bound and holds the GIL — this is the
    main speedup for this version, not just I/O overlap. Set
    parallel=False for easier debugging or on machines where spinning
    up a process pool isn't worth it (e.g. very few, very small files).

    If files don't all share identical columns, pd.concat fills the
    gaps with NaN and keeps the union of columns — no need to
    pre-align schemas yourself.
    """
    usePath = Path(usePath)
    print(f"read in {len(sorted_list)} jsons and merging....")

    frames = []
    if parallel and len(sorted_list) > 1:
        print("Using ProcessPoolExecutor for parallel parsing...")
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            args = [(usePath, fname, required_col) for fname in sorted_list]
            for i, df in enumerate(pool.map(_load_and_build, args)):
                print(f"[{i}/{len(sorted_list)}] {sorted_list[i]} -> {df.shape}")
                frames.append(df)
    else:
        print("Parsing files serially...")
        for i, fname in enumerate(sorted_list):
            df = _load_and_build((usePath, fname, required_col))
            print(f"[{i}/{len(sorted_list)}] {fname} -> {df.shape}")
            frames.append(df)

    if not frames:
        print("No files found.")
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True, sort=False)
    print(f"merged shape: {merged.shape}")
    return merged