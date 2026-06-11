from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import pandas as pd

from common import load_toml, ensure_dir, path_from_cfg


def parse_datetime(dt_str: str) -> datetime:
    dt_str = str(dt_str).strip()
    formats = ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"]
    for fmt in formats:
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            pass
    raise ValueError(f"Could not parse datetime: {dt_str}")


def load_transmitter_gps(gps_file: Path) -> list[dict]:
    tx_points = []
    with gps_file.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_type = str(row.get("type", "")).strip()
            if row_type != "T":
                continue
            try:
                tx_points.append(
                    {
                        "time": parse_datetime(row["date time"]),
                        "latitude": float(row["latitude"]),
                        "longitude": float(row["longitude"]),
                        "altitude_m": float(row["altitude(m)"]) if row.get("altitude(m)") else None,
                    }
                )
            except Exception as exc:
                print(f"Skipping bad GPS row: {exc}")
    tx_points.sort(key=lambda x: x["time"])
    return tx_points


def find_first_point_at_or_after(target_time: datetime, tx_points: list[dict]) -> dict | None:
    for point in tx_points:
        if point["time"] >= target_time:
            return point
    return None


def process_sweeps(sweep_file: Path, gps_file: Path, output_file: Path, tx_depth_m: float) -> None:
    tx_points = load_transmitter_gps(gps_file)
    if not tx_points:
        raise RuntimeError("No transmitter GPS points with type == 'T' were found.")

    input_rows = pd.read_csv(sweep_file)
    output_rows = []

    for _, row in input_rows.iterrows():
        peak1_time = parse_datetime(row["utc_peak1"])
        peak2_time = parse_datetime(row["utc_peak2"])
        p1 = find_first_point_at_or_after(peak1_time, tx_points)
        p2 = find_first_point_at_or_after(peak2_time, tx_points)

        out = row.to_dict()
        out["tx_depth_m"] = float(tx_depth_m)

        for prefix, point in [("peak1", p1), ("peak2", p2)]:
            if point is None:
                out[f"tx_time_{prefix}"] = ""
                out[f"tx_lat_{prefix}"] = ""
                out[f"tx_lon_{prefix}"] = ""
                out[f"tx_altitude_m_{prefix}"] = ""
            else:
                out[f"tx_time_{prefix}"] = point["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                out[f"tx_lat_{prefix}"] = point["latitude"]
                out[f"tx_lon_{prefix}"] = point["longitude"]
                out[f"tx_altitude_m_{prefix}"] = "" if point["altitude_m"] is None else point["altitude_m"]

        output_rows.append(out)

    ensure_dir(output_file.parent)
    pd.DataFrame(output_rows).to_csv(output_file, index=False)
    print(f"Saved: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach transmitter positions to sweep timestamps.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_toml(args.config)
    output_dir = ensure_dir(path_from_cfg(cfg, "transmitter_output_dir"))
    out_csv = output_dir / "transmission_times_with_tx_positions.csv"

    process_sweeps(
        sweep_file=Path(cfg["paths"]["sweep_times_csv"]),
        gps_file=Path(cfg["paths"]["tx_gps_csv"]),
        output_file=out_csv,
        tx_depth_m=float(cfg["transmitter"]["tx_depth_m"]),
    )


if __name__ == "__main__":
    main()
