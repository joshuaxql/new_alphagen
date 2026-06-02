"""
获取数据
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from common import PATH, ADJUST_NONE, ADJUST_PREV, ADJUST_POST


class Data:
    def __init__(self, path: Path = PATH):
        self.path = path
        self._basic_cache = None

    def basic(self):
        if self._basic_cache is None:
            basic = pd.read_csv(self.path / "basic.csv")
            basic["list_date"] = pd.to_datetime(
                basic["list_date"], format="%Y%m%d", errors="coerce"
            ).dt.date
            self._basic_cache = basic
        return self._basic_cache.copy()

    def seasoned_mask(
        self,
        trade_dates: list,
        stock_codes: list,
        min_list_days: int = 365,
    ) -> np.ndarray:
        """
        返回 shape=(n_days, n_stocks) 的布尔矩阵。
        True 表示该股票在该交易日上市已满 min_list_days，可参与交易。
        """
        n_days = len(trade_dates)
        n_stocks = len(stock_codes)
        if n_days == 0 or n_stocks == 0:
            return np.zeros((n_days, n_stocks), dtype=bool)
        if min_list_days <= 0:
            return np.ones((n_days, n_stocks), dtype=bool)

        basic = self.basic()[["ts_code", "list_date"]].copy()
        list_date_map = dict(zip(basic["ts_code"], basic["list_date"]))
        eligible_dates = []
        for code in stock_codes:
            list_date = list_date_map.get(code)
            if pd.isna(list_date):
                eligible_dates.append(np.datetime64("NaT"))
            else:
                eligible_dates.append(
                    np.datetime64(
                        pd.Timestamp(list_date) + pd.Timedelta(days=min_list_days)
                    )
                )

        trade_dates_np = np.array(pd.to_datetime(trade_dates), dtype="datetime64[D]")
        eligible_dates_np = np.array(eligible_dates, dtype="datetime64[D]")
        valid_listing = ~np.isnat(eligible_dates_np)
        mask = trade_dates_np[:, None] >= eligible_dates_np[None, :]
        if not np.all(valid_listing):
            mask[:, ~valid_listing] = False
        return mask

    def trade_cal(self):
        cal = pd.read_csv(self.path / "trade_cal.csv")
        cal = cal[cal["is_open"] == 1]
        cal["cal_date"] = pd.to_datetime(cal["cal_date"], format="%Y%m%d").dt.date
        cal["pretrade_date"] = pd.to_datetime(cal["pretrade_date"]).dt.date
        cal = cal.sort_values("cal_date", kind="stable")
        return cal.reset_index(drop=True)

    def daily(
        self,
        start_date: str = "20250101",
        end_date: str = "20260101",
        bj: bool = False,
        st: bool = False,
        adjust: int = ADJUST_NONE,
        min_list_days: int | None = 365,
    ):
        daily_path = self.path / "daily"
        files = [
            entry.path
            for entry in os.scandir(daily_path)
            if entry.is_file()
            and entry.name.endswith(".csv")
            and start_date <= entry.name[:8] <= end_date
        ]
        if not files:
            return pd.DataFrame(
                columns=[
                    "ts_code",
                    "trade_date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "pre_close",
                    "change",
                    "pct_chg",
                    "vol",
                    "amount",
                    "vwap",
                ]
            )

        files.sort()
        stock_st_path = self.path / "stock_st"
        frames = []
        for file in files:
            frame = pd.read_csv(file)
            if not st:
                st_file = stock_st_path / Path(file).name
                if st_file.is_file():
                    st_codes = pd.read_csv(st_file, usecols=["ts_code"], dtype=str)[
                        "ts_code"
                    ]
                    if not st_codes.empty:
                        frame = frame[~frame["ts_code"].isin(st_codes)]
            frames.append(frame)

        daily = pd.concat(frames, ignore_index=True)
        if not bj:
            daily = daily[~daily["ts_code"].str.endswith("BJ")]
        if daily.empty:
            daily["trade_date"] = pd.to_datetime(
                daily["trade_date"], format="%Y%m%d"
            ).dt.date
            return daily

        daily["_trade_date_dt"] = pd.to_datetime(
            daily["trade_date"], format="%Y%m%d", errors="coerce"
        )
        if min_list_days is not None and min_list_days > 0:
            basic = pd.read_csv(
                self.path / "basic.csv",
                usecols=["ts_code", "list_date"],
                dtype={"ts_code": str, "list_date": str},
            )
            basic["list_date"] = pd.to_datetime(
                basic["list_date"], format="%Y%m%d", errors="coerce"
            )
            daily = daily.merge(basic, on="ts_code", how="left")
            eligible_date = daily["list_date"] + pd.to_timedelta(
                min_list_days, unit="D"
            )
            daily = daily[
                daily["list_date"].notna() & (daily["_trade_date_dt"] >= eligible_date)
            ].copy()
            if daily.empty:
                return pd.DataFrame(
                    columns=[
                        "ts_code",
                        "trade_date",
                        "open",
                        "high",
                        "low",
                        "close",
                        "pre_close",
                        "change",
                        "pct_chg",
                        "vol",
                        "amount",
                        "vwap",
                    ]
                )

        if adjust != ADJUST_NONE:
            if adjust not in {ADJUST_PREV, ADJUST_POST}:
                raise ValueError(f"unsupported adjust mode: {adjust}")

            adjusted = daily.sort_values(
                ["ts_code", "trade_date"], kind="stable"
            ).copy()
            ratio = adjusted["close"].div(adjusted["pre_close"])
            cum_ratio = ratio.groupby(adjusted["ts_code"], sort=False).cumprod()
            first_ratio = cum_ratio.groupby(adjusted["ts_code"], sort=False).transform(
                "first"
            )
            first_close = adjusted.groupby("ts_code", sort=False)["close"].transform(
                "first"
            )
            post_close = first_close.mul(cum_ratio.div(first_ratio))
            scale = post_close.div(adjusted["close"])

            if adjust == ADJUST_PREV:
                last_close = adjusted.groupby("ts_code", sort=False)["close"].transform(
                    "last"
                )
                last_post_close = post_close.groupby(
                    adjusted["ts_code"], sort=False
                ).transform("last")
                scale = scale.mul(last_close.div(last_post_close))

            price_cols = ["open", "high", "low", "close", "pre_close", "vwap"]
            adjusted.loc[:, price_cols] = adjusted[price_cols].mul(scale, axis=0)
            if "change" in adjusted.columns:
                adjusted.loc[:, "change"] = adjusted["close"] - adjusted["pre_close"]
            daily.loc[adjusted.index, price_cols] = adjusted[price_cols]
            if "change" in adjusted.columns:
                daily.loc[adjusted.index, "change"] = adjusted["change"]

        daily["trade_date"] = daily["_trade_date_dt"].dt.date
        daily = daily.drop(columns=["_trade_date_dt"], errors="ignore")
        daily = daily.drop(columns=["list_date"], errors="ignore")
        return daily

    def market(self, name: str = "000001.SH"):
        market = pd.read_csv(self.path / "market" / (name + ".csv"))
        market["trade_date"] = pd.to_datetime(
            market["trade_date"], format="%Y%m%d"
        ).dt.date
        market = market.sort_values("trade_date", kind="stable")
        return market.reset_index(drop=True)
