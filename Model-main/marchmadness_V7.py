# Sample build out the data files
# Only looking at Male Data

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.decomposition import PCA
from torch import *

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    learning_curve,
    validation_curve,
    GridSearchCV,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    recall_score,
    confusion_matrix,
    precision_score,
    f1_score,
    accuracy_score,
    classification_report,
)


def change_loc(x):
    if x == "H":
        return "A"
    elif x == "A":
        return "H"
    else:
        return "N"


def rpi_stat(df):
    # Multiplied by the arbitrary Average of other teams
    df["A_RPI"] = (
        (df["A_WinRatio"] * 0.25)
        + (df["B_WinRatio"] * 0.5)
        + (df["B_WinRatio"].mean() * 0.25)
    )
    df["B_RPI"] = (
        (df["B_WinRatio"] * 0.25)
        + (df["A_WinRatio"] * 0.5)
        + (df["A_WinRatio"].mean() * 0.25)
    )
    return df


def rpi_stat_comp(df):
    # Multiplied by the arbitrary Average of other teams
    df["A_RPI"] = (
        (df["A_WinRatio"] * 0.25)
        + (df["B_WinRatio"] * 0.5)
        + (((df["A_WinRatio"] + df["B_WinRatio"]) / 2) * 0.25)
    )
    df["B_RPI"] = (
        (df["B_WinRatio"] * 0.25)
        + (df["A_WinRatio"] * 0.5)
        + (((df["A_WinRatio"] + df["B_WinRatio"]) / 2) * 0.25)
    )
    return df


def s_agg():
    # Create train and test data here

    # Season Details
    s_df = pd.read_csv("data/MRegularSeasonDetailedResults.csv")

    t_df = pd.read_csv("data/MNCAATourneyDetailedResults.csv")

    s_df = pd.concat([s_df, t_df], ignore_index=True)

    # print(s_df)
    s_df["WFGM2"] = s_df["WFGM"] - s_df["WFGM3"]
    s_df["LFGM2"] = s_df["LFGM"] - s_df["LFGM3"]
    s_df["WFGR"] = s_df["WFGM"] / s_df["WFGA"]
    s_df["LFGR"] = s_df["LFGM"] / s_df["LFGA"]
    s_df["WFGperc"] = s_df["WFGM"] + s_df["WFGA"]
    s_df["LFGperc"] = s_df["LFGM"] + s_df["LFGA"]
    # Rebounding Rate
    s_df["WORCHANCE"] = s_df["WOR"] + s_df["LDR"]
    s_df["LORCHANCE"] = s_df["LOR"] + s_df["WDR"]
    s_df["WORR"] = s_df["WOR"] / s_df["WORCHANCE"]
    s_df["LORR"] = s_df["LOR"] / s_df["LORCHANCE"]
    s_df["WPFDIFF"] = s_df["WPF"] - s_df["LPF"]
    s_df["LPFDIFF"] = s_df["LPF"] - s_df["WPF"]
    s_df["WFGADIFF"] = s_df["WFGA"] + s_df["WFTA"] / 2 - s_df["LFGA"] - s_df["LFTA"] / 2
    s_df["LFGADIFF"] = s_df["LFGA"] - s_df["LFTA"] / 2 - s_df["WFGA"] - s_df["WFTA"] / 2
    s_df["WFGA2"] = s_df["WFGA"] - s_df["WFGA3"]
    s_df["LFGA2"] = s_df["LFGA"] - s_df["LFGA3"]
    # Field Goal %
    s_df["WFG2PCT"] = s_df["WFGM2"] - s_df["WFGA2"]
    s_df["LFG2PCT"] = s_df["LFGM2"] - s_df["LFGA2"]
    # Possession
    s_df["WPOSS"] = s_df["WFGA"] + s_df["WTO"] + (0.44 * s_df["WFTA"]) - s_df["WOR"]
    s_df["LPOSS"] = s_df["LFGA"] + s_df["LTO"] + (0.44 * s_df["LFTA"]) - s_df["LOR"]
    # Points per possession
    s_df["WPPP"] = s_df["WScore"] / s_df["WPOSS"]
    s_df["LPPP"] = s_df["LScore"] / s_df["LPOSS"]
    # Offensive Efficiency
    s_df["WOER"] = 100 * s_df["WPPP"]
    s_df["LOER"] = 100 * s_df["LPPP"]
    s_df["WDER"] = 100 * s_df["LPPP"]
    s_df["LDER"] = 100 * s_df["WPPP"]
    # Net Rating
    s_df["WNET_RAT"] = s_df["WOER"] - s_df["WDER"]
    s_df["LNET_RAT"] = s_df["LOER"] - s_df["LDER"]
    # Defensive Rebound Rate
    s_df["WDRR"] = s_df["WDR"] / (s_df["WDR"] + s_df["LOR"])
    s_df["LDRR"] = s_df["LDR"] / (s_df["LDR"] + s_df["WOR"])
    # Assist %
    s_df["WAstRatio"] = (100 * s_df["WAst"]) / (
        s_df["WAst"] + s_df["WFGA"] + (0.44 * s_df["WFTA"]) + s_df["WTO"]
    )
    s_df["LAstRatio"] = (100 * s_df["LAst"]) / (
        s_df["LAst"] + s_df["LFGA"] + (0.44 * s_df["LFTA"]) + s_df["LTO"]
    )
    # Pace
    s_df["WPace"] = s_df["WPOSS"] / (40 + s_df["NumOT"] * 5)
    s_df["LPace"] = s_df["LPOSS"] / (40 + s_df["NumOT"] * 5)
    # FTA Rate
    s_df["WFTARate"] = s_df["WFTA"] / s_df["WFGA"]
    s_df["LFTARate"] = s_df["LFTA"] / s_df["LFGA"]
    # 3 point Rate
    s_df["W3PAR"] = s_df["WFGA3"] / s_df["WFGA"]
    s_df["L3PAR"] = s_df["LFGA3"] / s_df["LFGA"]
    # FG % efficiency
    s_df["WeFG"] = (s_df["WFGM"] + (0.5 * s_df["WFGM3"])) / s_df["WFGA"]
    s_df["LeFG"] = (s_df["LFGM"] + (0.5 * s_df["LFGM3"])) / s_df["LFGA"]
    # Rebound Rate
    s_df["WTRR"] = (s_df["WOR"] + s_df["WDR"]) / (
        s_df["WOR"] + s_df["WDR"] + s_df["LOR"] + s_df["LDR"]
    )
    s_df["LTRR"] = (s_df["LOR"] + s_df["LDR"]) / (
        s_df["LOR"] + s_df["LDR"] + s_df["WOR"] + s_df["WDR"]
    )
    # Assist %
    s_df["WAstper"] = s_df["WAst"] / s_df["WFGM"]
    s_df["LAstper"] = s_df["LAst"] / s_df["LFGM"]
    # Stl %
    s_df["WStlper"] = s_df["WStl"] / s_df["LPOSS"]
    s_df["LStlper"] = s_df["LStl"] / s_df["WPOSS"]
    # Blk %
    s_df["WBlkper"] = s_df["WBlk"] / s_df["LFGA"]
    s_df["LBlkper"] = s_df["LBlk"] / s_df["WFGA"]
    # PPSA
    s_df["WPPSA"] = s_df["WScore"] / s_df["WFGA"]
    s_df["LPPSA"] = s_df["LScore"] / s_df["LFGA"]
    # 3 FG Percent
    s_df["W3Fper"] = s_df["WFGM3"] / s_df["WFGA3"]
    s_df["L3Fper"] = s_df["LFGM3"] / s_df["LFGA3"]
    # Turnover Ratio
    s_df["WToRatio"] = s_df["WTO"] / (s_df["WFGA"] + 0.44 * s_df["WFTA"] + s_df["WTO"])
    s_df["LToRatio"] = s_df["LTO"] / (s_df["LFGA"] + 0.44 * s_df["LFTA"] + s_df["LTO"])
    s_df.fillna(0, inplace=True)

    # Just copy and past the print of the tables columns when I update. Only keep WTeamID
    s_winner = s_df[
        [
            "Season",
            "DayNum",
            "WTeamID",
            "WScore",
            "LScore",
            "WLoc",
            "NumOT",
            "WFGM",
            "WFGA",
            "WFGM3",
            "WFGA3",
            "WFTM",
            "WFTA",
            "WOR",
            "WDR",
            "WAst",
            "WTO",
            "WStl",
            "WBlk",
            "WPF",
            "LFGM",
            "LFGA",
            "LFGM3",
            "LFGA3",
            "LFTM",
            "LFTA",
            "LOR",
            "LDR",
            "LAst",
            "LTO",
            "LStl",
            "LBlk",
            "LPF",
            "WFGM2",
            "LFGM2",
            "WFGR",
            "LFGR",
            "WORCHANCE",
            "LORCHANCE",
            "WORR",
            "LORR",
            "WPFDIFF",
            "LPFDIFF",
            "WFGADIFF",
            "LFGADIFF",
            "WFGA2",
            "LFGA2",
            "WFG2PCT",
            "LFG2PCT",
            "WPOSS",
            "LPOSS",
            "WPPP",
            "LPPP",
            "WOER",
            "LOER",
            "WDER",
            "LDER",
            "WNET_RAT",
            "LNET_RAT",
            "WDRR",
            "LDRR",
            "WAstRatio",
            "LAstRatio",
            "WPace",
            "LPace",
            "WFTARate",
            "LFTARate",
            "W3PAR",
            "L3PAR",
            "WeFG",
            "LeFG",
            "WTRR",
            "LTRR",
            "WAstper",
            "LAstper",
            "WStlper",
            "LStlper",
            "WBlkper",
            "LBlkper",
            "WPPSA",
            "LPPSA",
            "WFGperc",
            "LFGperc",
            "W3Fper",
            "L3Fper",
            "WToRatio",
            "LToRatio",
        ]
    ]
    # Copy the above, but remove W and change L to Opp
    s_winner.columns = [
        "Season",
        "DayNum",
        "TeamID",
        "Score",
        "OppScore",
        "Loc",
        "NumOT",
        "FGM",
        "FGA",
        "FGM3",
        "FGA3",
        "FTM",
        "FTA",
        "OR",
        "DR",
        "Ast",
        "TO",
        "Stl",
        "Blk",
        "PF",
        "OppFGM",
        "OppFGA",
        "OppFGM3",
        "OppFGA3",
        "OppFTM",
        "OppFTA",
        "OppOR",
        "OppDR",
        "OppAst",
        "OppTO",
        "OppStl",
        "OppBlk",
        "OppPF",
        "FGM2",
        "OppFGM2",
        "FGR",
        "OppFGR",
        "ORCHANCE",
        "OppORCHANCE",
        "ORR",
        "OppORR",
        "PFDIFF",
        "OppPFDIFF",
        "FGADIFF",
        "OppFGADIFF",
        "FGA2",
        "OppFGA2",
        "FG2PCT",
        "OppFG2PCT",
        "POSS",
        "OppPOSS",
        "PPP",
        "OppPPP",
        "OER",
        "OppOER",
        "DER",
        "OppDER",
        "NET_RAT",
        "OppNET_RAT",
        "DRR",
        "OppDRR",
        "AstRatio",
        "OppAstRatio",
        "Pace",
        "OppPace",
        "FTARate",
        "OppFTARate",
        "3PAR",
        "Opp3PAR",
        "eFG",
        "OppeFG",
        "TRR",
        "OppTRR",
        "Astper",
        "OppAstper",
        "Stlper",
        "OppStlper",
        "Blkper",
        "OppBlkper",
        "PPSA",
        "OppPPSA",
        "FGperc",
        "OppFGperc",
        "3Fper",
        "Opp3Fper",
        "ToRatio",
        "OppToRatio",
    ]
    s_winner["Result"] = 1

    s_loser = s_df[
        [
            "Season",
            "DayNum",
            "LTeamID",
            "LScore",
            "WScore",
            "WLoc",
            "NumOT",
            "LFGM",
            "LFGA",
            "LFGM3",
            "LFGA3",
            "LFTM",
            "LFTA",
            "LOR",
            "LDR",
            "LAst",
            "LTO",
            "LStl",
            "LBlk",
            "LPF",
            "WFGM",
            "WFGA",
            "WFGM3",
            "WFGA3",
            "WFTM",
            "WFTA",
            "WOR",
            "WDR",
            "WAst",
            "WTO",
            "WStl",
            "WBlk",
            "WPF",
            "LFGM2",
            "WFGM2",
            "LFGR",
            "WFGR",
            "LORCHANCE",
            "WORCHANCE",
            "LORR",
            "WORR",
            "LPFDIFF",
            "WPFDIFF",
            "LFGADIFF",
            "WFGADIFF",
            "LFGA2",
            "WFGA2",
            "LFG2PCT",
            "WFG2PCT",
            "LPOSS",
            "WPOSS",
            "LPPP",
            "WPPP",
            "LOER",
            "WOER",
            "LDER",
            "WDER",
            "LNET_RAT",
            "WNET_RAT",
            "LDRR",
            "WDRR",
            "LAstRatio",
            "WAstRatio",
            "LPace",
            "WPace",
            "LFTARate",
            "WFTARate",
            "L3PAR",
            "W3PAR",
            "LeFG",
            "WeFG",
            "LTRR",
            "WTRR",
            "LAstper",
            "WAstper",
            "LStlper",
            "WStlper",
            "LBlkper",
            "WBlkper",
            "LPPSA",
            "WPPSA",
            "LFGperc",
            "WFGperc",
            "L3Fper",
            "W3Fper",
            "LToRatio",
            "WToRatio",
        ]
    ]
    s_loser.columns = [
        "Season",
        "DayNum",
        "TeamID",
        "Score",
        "OppScore",
        "Loc",
        "NumOT",
        "FGM",
        "FGA",
        "FGM3",
        "FGA3",
        "FTM",
        "FTA",
        "OR",
        "DR",
        "Ast",
        "TO",
        "Stl",
        "Blk",
        "PF",
        "OppFGM",
        "OppFGA",
        "OppFGM3",
        "OppFGA3",
        "OppFTM",
        "OppFTA",
        "OppOR",
        "OppDR",
        "OppAst",
        "OppTO",
        "OppStl",
        "OppBlk",
        "OppPF",
        "FGM2",
        "OppFGM2",
        "FGR",
        "OppFGR",
        "ORCHANCE",
        "OppORCHANCE",
        "ORR",
        "OppORR",
        "PFDIFF",
        "OppPFDIFF",
        "FGADIFF",
        "OppFGADIFF",
        "FGA2",
        "OppFGA2",
        "FG2PCT",
        "OppFG2PCT",
        "POSS",
        "OppPOSS",
        "PPP",
        "OppPPP",
        "OER",
        "OppOER",
        "DER",
        "OppDER",
        "NET_RAT",
        "OppNET_RAT",
        "DRR",
        "OppDRR",
        "AstRatio",
        "OppAstRatio",
        "Pace",
        "OppPace",
        "FTARate",
        "OppFTARate",
        "3PAR",
        "Opp3PAR",
        "eFG",
        "OppeFG",
        "TRR",
        "OppTRR",
        "Astper",
        "OppAstper",
        "Stlper",
        "OppStlper",
        "Blkper",
        "OppBlkper",
        "PPSA",
        "OppPPSA",
        "FGperc",
        "OppFGperc",
        "3Fper",
        "Opp3Fper",
        "ToRatio",
        "OppToRatio",
    ]
    s_loser["Result"] = 0

    s_loser["Loc"] = s_loser["Loc"].apply(lambda x: change_loc(x))

    s_hist = pd.concat([s_winner, s_loser])
    s_hist.drop(["Loc"], axis=1, inplace=True)
    # print("The columns", s_hist.columns)

    s_hist_agg = (
        s_hist.groupby(["Season", "TeamID"])
        .agg(
            {
                "DayNum": "count",
                "Score": "sum",
                "OppScore": "sum",
                "NumOT": "sum",
                "FGM": "mean",
                "FGA": "mean",
                "FGM3": "mean",
                "FGA3": "mean",
                "FTM": "mean",
                "FTA": "mean",
                "OR": "mean",
                "DR": "mean",
                "Ast": "mean",
                "TO": "mean",
                "Stl": "mean",
                "Blk": "mean",
                "PF": "mean",
                "OppFGM": "mean",
                "OppFGA": "mean",
                "OppFGM3": "mean",
                "OppFGA3": "mean",
                "OppFTM": "mean",
                "OppFTA": "mean",
                "OppOR": "mean",
                "OppDR": "mean",
                "OppAst": "mean",
                "OppTO": "mean",
                "OppStl": "mean",
                "OppBlk": "mean",
                "OppPF": "mean",
                "FGM2": "mean",
                "OppFGM2": "mean",
                "FGR": "mean",
                "OppFGR": "mean",
                "ORCHANCE": "mean",
                "OppORCHANCE": "mean",
                "ORR": "mean",
                "OppORR": "mean",
                "PFDIFF": "mean",
                "OppPFDIFF": "mean",
                "FGADIFF": "mean",
                "OppFGADIFF": "mean",
                "FGA2": "mean",
                "OppFGA2": "mean",
                "FG2PCT": "mean",
                "OppFG2PCT": "mean",
                "POSS": "mean",
                "OppPOSS": "mean",
                "PPP": "mean",
                "OppPPP": "mean",
                "OER": "mean",
                "OppOER": "mean",
                "DER": "mean",
                "OppDER": "mean",
                "NET_RAT": "mean",
                "OppNET_RAT": "mean",
                "DRR": "mean",
                "OppDRR": "mean",
                "AstRatio": "mean",
                "OppAstRatio": "mean",
                "Pace": "mean",
                "OppPace": "mean",
                "FTARate": "mean",
                "OppFTARate": "mean",
                "3PAR": "mean",
                "Opp3PAR": "mean",
                "eFG": "mean",
                "OppeFG": "mean",
                "TRR": "mean",
                "OppTRR": "mean",
                "Astper": "mean",
                "OppAstper": "mean",
                "Stlper": "mean",
                "OppStlper": "mean",
                "Blkper": "mean",
                "OppBlkper": "mean",
                "PPSA": "mean",
                "OppPPSA": "mean",
                "FGperc": "mean",
                "OppFGperc": "mean",
                "3Fper": "mean",
                "Opp3Fper": "mean",
                "ToRatio": "mean",
                "OppToRatio": "mean",
                "Result": "sum",
            }
        )
        .reset_index()
        .rename(columns={"DayNum": "Matches_Played", "Result": "Matches_Won"})
    )

    # Win Ratio
    s_hist_agg["WinRatio"] = s_hist_agg["Matches_Won"] / s_hist_agg["Matches_Played"]
    # Loss Ratio
    s_hist_agg["LossRatio"] = (
        s_hist_agg["Matches_Played"] - s_hist_agg["Matches_Won"]
    ) / s_hist_agg["Matches_Played"]
    # AvgScore
    s_hist_agg["AvgScore"] = s_hist_agg["Score"] / s_hist_agg["Matches_Played"]
    s_hist_agg["AvgOppScore"] = s_hist_agg["OppScore"] / s_hist_agg["Matches_Played"]
    # Score Diff
    s_hist_agg["ScoreDiff"] = s_hist_agg["Score"] - s_hist_agg["OppScore"]
    # Turnover Difference
    s_hist_agg["TODIFF"] = s_hist_agg["TO"] - s_hist_agg["OppTO"]
    # STLDIFF
    s_hist_agg["STLDIFF"] = s_hist_agg["Stl"] - s_hist_agg["OppStl"]
    # BLKDIFF
    s_hist_agg["BLKDIFF"] = s_hist_agg["Blk"] - s_hist_agg["OppBlk"]

    # print("Aggregate Table Season", s_hist_agg)

    return s_hist_agg


def test_train_df(s_hist_agg):
    # Compact Results
    t_c_df = pd.read_csv("data/MNCAATourneyCompactResults.csv")

    train_df = (
        t_c_df[["Season", "WTeamID", "LTeamID"]]
        .reset_index(drop=True)
        .merge(
            s_hist_agg.add_prefix("A_"),
            left_on=["Season", "WTeamID"],
            right_on=["A_Season", "A_TeamID"],
        )
        .drop(["A_Season", "A_TeamID"], axis=1)
    )
    train_df = train_df.merge(
        s_hist_agg.add_prefix("B_"),
        left_on=["Season", "LTeamID"],
        right_on=["B_Season", "B_TeamID"],
    ).drop(["B_Season", "B_TeamID"], axis=1)
    train_df = train_df.rename(columns={"WTeamID": "ATeamID", "LTeamID": "BTeamID"})
    train_df["AWin"] = 1
    # print("Training Data Frame Columns", train_df.columns)
    # print(train_df)

    loss = train_df[
        [
            "Season",
            "BTeamID",
            "ATeamID",
            "B_Matches_Played",
            "B_Score",
            "B_OppScore",
            "B_NumOT",
            "B_FGM",
            "B_FGA",
            "B_FGM3",
            "B_FGA3",
            "B_FTM",
            "B_FTA",
            "B_OR",
            "B_DR",
            "B_Ast",
            "B_TO",
            "B_Stl",
            "B_Blk",
            "B_PF",
            "B_OppFGM",
            "B_OppFGA",
            "B_OppFGM3",
            "B_OppFGA3",
            "B_OppFTM",
            "B_OppFTA",
            "B_OppOR",
            "B_OppDR",
            "B_OppAst",
            "B_OppTO",
            "B_OppStl",
            "B_OppBlk",
            "B_OppPF",
            "B_POSS",
            "B_OppPOSS",
            "B_PPP",
            "B_OppPPP",
            "B_OER",
            "B_OppOER",
            "B_DER",
            "B_OppDER",
            "B_NET_RAT",
            "B_OppNET_RAT",
            "B_ORR",
            "B_OppORR",
            "B_DRR",
            "B_OppDRR",
            "B_AstRatio",
            "B_OppAstRatio",
            "B_FGM2",
            "B_OppFGM2",
            "B_FGR",
            "B_OppFGR",
            "B_ORCHANCE",
            "B_OppORCHANCE",
            "B_PFDIFF",
            "B_OppPFDIFF",
            "B_FGADIFF",
            "B_OppFGADIFF",
            "B_FGA2",
            "B_OppFGA2",
            "B_FG2PCT",
            "B_OppFG2PCT",
            "B_ScoreDiff",
            "B_TODIFF",
            "B_STLDIFF",
            "B_BLKDIFF",
            "B_Pace",
            "B_OppPace",
            "B_FTARate",
            "B_OppFTARate",
            "B_3PAR",
            "B_Opp3PAR",
            "B_eFG",
            "B_OppeFG",
            "B_TRR",
            "B_OppTRR",
            "B_Astper",
            "B_OppAstper",
            "B_Stlper",
            "B_OppStlper",
            "B_Blkper",
            "B_OppBlkper",
            "B_PPSA",
            "B_OppPPSA",
            "B_FGperc",
            "B_OppFGperc",
            "B_3Fper",
            "B_Opp3Fper",
            "B_ToRatio",
            "B_OppToRatio",
            "B_Matches_Won",
            "B_WinRatio",
            "B_AvgScore",
            "B_AvgOppScore",
            "B_LossRatio",
            "A_Matches_Played",
            "A_Score",
            "A_OppScore",
            "A_NumOT",
            "A_FGM",
            "A_FGA",
            "A_FGM3",
            "A_FGA3",
            "A_FTM",
            "A_FTA",
            "A_OR",
            "A_DR",
            "A_Ast",
            "A_TO",
            "A_Stl",
            "A_Blk",
            "A_PF",
            "A_OppFGM",
            "A_OppFGA",
            "A_OppFGM3",
            "A_OppFGA3",
            "A_OppFTM",
            "A_OppFTA",
            "A_OppOR",
            "A_OppDR",
            "A_OppAst",
            "A_OppTO",
            "A_OppStl",
            "A_OppBlk",
            "A_OppPF",
            "A_POSS",
            "A_OppPOSS",
            "A_PPP",
            "A_OppPPP",
            "A_OER",
            "A_OppOER",
            "A_DER",
            "A_OppDER",
            "A_NET_RAT",
            "A_OppNET_RAT",
            "A_ORR",
            "A_OppORR",
            "A_DRR",
            "A_OppDRR",
            "A_AstRatio",
            "A_OppAstRatio",
            "A_FGM2",
            "A_OppFGM2",
            "A_FGR",
            "A_OppFGR",
            "A_ORCHANCE",
            "A_OppORCHANCE",
            "A_PFDIFF",
            "A_OppPFDIFF",
            "A_FGADIFF",
            "A_OppFGADIFF",
            "A_FGA2",
            "A_OppFGA2",
            "A_FG2PCT",
            "A_OppFG2PCT",
            "A_ScoreDiff",
            "A_TODIFF",
            "A_STLDIFF",
            "A_BLKDIFF",
            "A_Pace",
            "A_OppPace",
            "A_FTARate",
            "A_OppFTARate",
            "A_3PAR",
            "A_Opp3PAR",
            "A_eFG",
            "A_OppeFG",
            "A_TRR",
            "A_OppTRR",
            "A_Astper",
            "A_OppAstper",
            "A_Stlper",
            "A_OppStlper",
            "A_Blkper",
            "A_OppBlkper",
            "A_PPSA",
            "A_OppPPSA",
            "A_FGperc",
            "A_OppFGperc",
            "A_3Fper",
            "A_Opp3Fper",
            "A_ToRatio",
            "A_OppToRatio",
            "A_Matches_Won",
            "A_WinRatio",
            "A_AvgScore",
            "A_AvgOppScore",
            "A_LossRatio",
        ]
    ]

    loss["AWin"] = 0
    loss.columns = train_df.columns
    train_df = pd.concat([train_df, loss])
    x = train_df["AWin"]

    # Creates the RPI DataFrame
    # rpi_data = train_df.drop(columns=['AWin'])
    # rpi_data = rpi_stat(rpi_data)
    # rpi_data.to_csv('data/rpi_data.csv', index=False)

    train_df = train_df[
        [
            "Season",
            "A_Score",
            "A_OppScore",
            "A_NumOT",
            "A_FGM",
            "A_FGA",
            "A_FGM3",
            "A_FGA3",
            "A_FTM",
            "A_FTA",
            "A_OR",
            "A_DR",
            "A_Ast",
            "A_TO",
            "A_Stl",
            "A_Blk",
            "A_PF",
            "A_OppNET_RAT",
            "B_Score",
            "B_OppScore",
            "B_NumOT",
            "B_FGM",
            "B_FGA",
            "B_FGM3",
            "B_FGA3",
            "B_FTM",
            "B_FTA",
            "B_OR",
            "B_DR",
            "B_Ast",
            "B_TO",
            "B_Stl",
            "B_Blk",
            "B_PF",
            "B_OppNET_RAT",
        ]
    ]

    train_df["AWin"] = x

    data_all = train_df.drop(columns=["Season"])
    data_rec = train_df[(train_df["Season"] >= 2020)].drop(columns=["Season"])

    # print("All data", data_all)
    # print("Recent Data", data_rec)

    X_all = data_all.iloc[:, :-1]
    Y_all = data_all.iloc[:, -1]
    X_rec = data_rec.iloc[:, :-1]
    Y_rec = data_rec.iloc[:, -1]

    X_train_rec, X_test_rec, y_train_rec, y_test_rec = train_test_split(
        X_rec, Y_rec, test_size=0.2, random_state=7
    )
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
        X_all, Y_all, test_size=0.2, random_state=7
    )

    # print("The X Train Data", X_train_rec)

    return (
        X_train_rec,
        X_test_rec,
        y_train_rec,
        y_test_rec,
        X_train_all,
        X_test_all,
        y_train_all,
        y_test_all,
    )


def cinderella_weight(teamA_stats, teamB_stats, seedA, seedB, teamA_id, teamB_id):

    kpom = pd.read_csv("data/kenpom_luck_teamID.csv")
    rpi = pd.read_csv("data/rpi_data.csv")
    a_rpi = rpi[["ATeamID", "A_RPI"]].groupby("ATeamID")["A_RPI"].mean().reset_index()
    b_rpi = rpi[["BTeamID", "B_RPI"]].groupby("BTeamID")["B_RPI"].mean().reset_index()
    # print("The rpi's", a_rpi, b_rpi)

    weight = 1.0

    if (seedB - seedA) >= 5:
        weight += 1.50
    if "B_AvgScore" in teamB_stats and "A_AvgScore" in teamA_stats:
        if np.int32(teamB_stats["B_AvgScore"]) > np.int32(teamA_stats["A_AvgScore"]):
            weight += 0.55
    if teamA_id in kpom["TeamID"].values and teamB_id in kpom["TeamID"].values:
        if (
            kpom.loc[kpom["TeamID"] == teamA_id, "luck"].values[0]
            > kpom.loc[kpom["TeamID"] == teamB_id, "luck"].values[0]
        ):
            weight += 0.25
    if teamA_id in a_rpi["ATeamID"].values and teamB_id in b_rpi["BTeamID"].values:
        if (
            b_rpi.loc[b_rpi["BTeamID"] == teamA_id, "B_RPI"].values[0]
            > a_rpi.loc[a_rpi["ATeamID"] == teamB_id, "A_RPI"].values[0]
        ):
            weight += -0.15

    # print("cinderella weight")
    return weight


def rf_model(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=256, max_depth=10, random_state=7)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf)
    rf_pr = precision_score(y_test, y_pred_rf)
    print("RF accuracy, f1 and precision", rf_accuracy, rf_f1, rf_pr)

    # Used Gridsearch to finish finding the model
    # gscv_rf = GridSearchCV(estimator=rf, param_grid={'n_estimators': [4, 16, 100, 256], 'max_depth': [2, 4, 8, 10, 16]})
    # gscv_rf.fit(X_train, y_train)
    # print("The best params", gscv_rf.best_params_)
    # print("The best scores", gscv_rf.best_score_)

    # rf_cm = confusion_matrix(y_test, y_pred_rf)
    # fig, ax = plt.subplots(1, 1, figsize=(12,8))
    # sns.heatmap(rf_cm, annot=True, ax=ax, fmt='g', cmap='Blues')
    # ax.set_title(' RF Confusion Matrix')
    # ax.set_xlabel('Predicted')
    # ax.set_ylabel('Actual')
    # plt.tight_layout()
    # plt.savefig('data/RF_Confusion_Matrix.png')
    # plt.show()
    # print("RF Class")
    # print(classification_report(y_test, y_pred_rf))
    # important = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(10)
    # print("These features from random Forest that are important: ", important)

    return rf
    # return reg


def log_model(X_train, X_test, y_train, y_test):
    lr = LogisticRegression(C=1, max_iter=200, solver="lbfgs")
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    lr_accuracy = accuracy_score(y_test, y_pred_lr)
    lr_f1 = f1_score(y_test, y_pred_lr)
    lr_pr = precision_score(y_test, y_pred_lr)
    print("LR accuracy, f1 and precision", lr_accuracy, lr_f1, lr_pr)

    # Used Gridsearch to finish finding the model
    # gscv_lr = GridSearchCV(estimator=lr, param_grid={'C': [.5, 1, 1.5, 3, 5], 'max_iter': [25, 100, 150, 200],
    #                                                  'solver': ['lbfgs', 'sag', 'liblinear', 'newton-cg']})
    # gscv_lr.fit(X_train, y_train)
    # print("The best params", gscv_lr.best_params_)
    # print("The best scores", gscv_lr.best_score_)

    # lr_cm = confusion_matrix(y_test, y_pred_lr)
    # fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    # sns.heatmap(lr_cm, annot=True, ax=ax, fmt='g', cmap='Blues')
    # ax.set_title('LR Confusion Matrix')
    # ax.set_xlabel('Predicted')
    # ax.set_ylabel('Actual')
    # plt.tight_layout()
    # plt.savefig('data/LR_Confusion_Matrix.png')
    # plt.show()
    # print("LR Class")
    # print(classification_report(y_test, y_pred_lr))
    # coefficients = pd.Series(lr.coef_[0], index=X_train.columns).sort_values(ascending=False).head(10)
    # print("These are the feature columns important to LR: ", coefficients)

    return lr


def who_won(teamA_id, seed1, teamB_id, seed2, s_hist_agg, lr, rf):
    # Code for the input of 2 team Ids
    team_stats_df = s_hist_agg.copy()
    # Adjusted the Team Stats
    team_stats_df["weight"] = team_stats_df["Season"].apply(
        lambda x: 1 / (2024 - x + 1)
    )
    team_stats_df = team_stats_df.groupby("TeamID", as_index=False).mean()
    team_stats_df = team_stats_df.set_index("TeamID")
    # print(team_stats_df)

    # Grab from f_df
    teamA_stats = team_stats_df.loc[[teamA_id]].add_prefix("A_")
    teamA_stats_adj = teamA_stats[
        [
            "A_Score",
            "A_OppScore",
            "A_NumOT",
            "A_FGM",
            "A_FGA",
            "A_FGM3",
            "A_FGA3",
            "A_FTM",
            "A_FTA",
            "A_OR",
            "A_DR",
            "A_Ast",
            "A_TO",
            "A_Stl",
            "A_Blk",
            "A_PF",
            "A_OppNET_RAT",
        ]
    ]

    teamB_stats = team_stats_df.loc[[teamB_id]].add_prefix("B_")
    teamB_stats_adj = teamB_stats[
        [
            "B_Score",
            "B_OppScore",
            "B_NumOT",
            "B_FGM",
            "B_FGA",
            "B_FGM3",
            "B_FGA3",
            "B_FTM",
            "B_FTA",
            "B_OR",
            "B_DR",
            "B_Ast",
            "B_TO",
            "B_Stl",
            "B_Blk",
            "B_PF",
            "B_OppNET_RAT",
        ]
    ]

    new_game_features = pd.concat(
        [
            teamA_stats_adj.reset_index(drop=True),
            teamB_stats_adj.reset_index(drop=True),
        ],
        axis=1,
    )
    # print("The new game features", new_game_features)

    lr_prob = lr.predict_proba(new_game_features)
    rf_prob = rf.predict_proba(new_game_features)
    # print(f"The Win probability for the LR {lr_prob}, the Random Forest {rf_prob}")

    # Cinderella Weighting Calculation
    teama_name, teamb_name = get_team_name(teamA_id, teamB_id)

    c_weight = cinderella_weight(
        teamA_stats, teamB_stats, seed1, seed2, teamA_id, teamB_id
    )
    # print("The cinderella weight: ", c_weight)

    win_prob = ((np.int32(lr_prob[0][1] * c_weight) + np.int32(rf_prob[0][1])) / 2) - (
        ((np.int32(lr_prob[0][0]) * c_weight) + np.int32(rf_prob[0][0])) / 2
    )

    y = 0
    # print("The Win Probability", win_prob)
    if win_prob > 0:
        winner = teama_name
        y = 1
    else:
        winner = teamb_name

    # Output Dictionary of the winnner
    winner_dict = {}
    winner_dict["TeamA"] = teama_name
    winner_dict["TeamA_Avg Points Per Game: "] = teamA_stats["A_AvgScore"].iloc[0]
    winner_dict["TeamA_Turnover Ratio: "] = teamA_stats["A_TODIFF"].iloc[0]
    winner_dict["TeamA_Efficiency Rating: "] = teamA_stats["A_NET_RAT"].iloc[0]
    winner_dict["TeamB"] = teamb_name
    winner_dict["TeamB_Avg Points Per Game: "] = teamB_stats["B_AvgScore"].iloc[0]
    winner_dict["TeamB_Turnover Ratio: "] = teamB_stats["B_TODIFF"].iloc[0]
    winner_dict["TeamB_Efficiency Rating: "] = teamB_stats["B_NET_RAT"].iloc[0]
    winner_dict["Winner"] = winner

    return lr_prob, rf_prob, winner_dict, y


def fit_models():
    season_data = s_agg()

    (
        X_train_rec,
        X_test_rec,
        y_train_rec,
        y_test_rec,
        X_train_all,
        X_test_all,
        y_train_all,
        y_test_all,
    ) = test_train_df(season_data)

    lr = log_model(X_train_all, X_test_all, y_train_all, y_test_all)
    rf = rf_model(X_train_rec, X_test_rec, y_train_rec, y_test_rec)

    return lr, rf, season_data


def get_seeds(AteamID, BteamID):
    seed = pd.read_csv("data/2024_tourney_seeds.csv")
    seed["Seed"] = seed["Seed"].str.extract("(\d+)").astype(np.int32)
    # print(seed)

    seed_a = seed.loc[seed["TeamID"] == AteamID, "Seed"]
    seed_b = seed.loc[seed["TeamID"] == BteamID, "Seed"]

    return seed_a.values[0], seed_b.values[0]


def get_team_name(AteamID, BteamID):
    teams = pd.read_csv("data/MTeams.csv")

    team_a = teams.loc[teams["TeamID"] == AteamID, "TeamName"]
    team_b = teams.loc[teams["TeamID"] == BteamID, "TeamName"]

    return team_a.values[0], team_b.values[0]


def welcome():
    AteamID = input("The First Team to compete: ")
    BteamID = input("The Second Team to compete: ")

    return AteamID, BteamID


def test_project():

    # AteamID, BteamID = welcome()
    # seed1, seed2 = get_seeds(AteamID, BteamID)

    seed1, seed2 = get_seeds(1163, 1391)
    # print("The seeds", seed1, seed2)

    lr, rf, season_data = fit_models()

    lr_prob, rf_prob, game_stats, y = who_won(
        teamA_id=1163,
        seed1=seed1,
        teamB_id=1391,
        seed2=seed2,
        s_hist_agg=season_data,
        lr=lr,
        rf=rf,
    )

    print("Probabilities of Team A vs Team B", lr_prob, rf_prob)
    print("The Game Stats!", game_stats)

    # Used to create the KenPom luck value for the Cinderella Weighting
    # df = pd.read_csv('data/kenpom_2024.csv')
    # df_n = pd.read_csv('data/MTeams.csv')
    # df = pd.merge(df_n, df, on='TeamName', how='inner')
    # df = df[df['Season'] == 2024]
    # df = df[['TeamID', 'luck']].reset_index()
    # print(df)
    # df.to_csv('data/kenpom_luck_teamID.csv')

    # Sample output for the final matrix for 2024. Only uncomment if wanting to compare actual vs predicted results
    # matchups = pd.read_csv('data/2024_tourney_matchups.csv')
    # target = matchups.iloc[:, -1]
    # games = matchups.iloc[:, :-1]
    # y_pred = []
    # for i, matchup in games.iterrows():
    #
    #     TeamA = matchup['TeamA']
    #     TeamB = matchup['TeamB']
    #
    #     seed1, seed2 = get_seeds(TeamA, TeamB)
    #     # print("The seeds", seed1, seed2)
    #
    #     lr, rf, season_data = fit_models()
    #
    #     lr_prob, rf_prob, game_stats, y = who_won(teamA_id=TeamA, seed1=seed1, teamB_id=TeamB, seed2=seed2,
    #                                                      s_hist_agg=season_data, lr=lr, rf=rf)
    #     y_pred.append(y)
    #
    #     print("Probabilities of Team A vs Team B", lr_prob, rf_prob)
    #     print("The Game Stats!", game_stats)
    #
    # y_pred = pd.DataFrame(y_pred)
    # print("The Final y prediction", y_pred)
    #
    # cm = confusion_matrix(target, y_pred)
    # fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    # sns.heatmap(cm, annot=True, ax=ax, fmt='g', cmap='Blues')
    # ax.set_title('March Madness 2024 Confusion Matrix')
    # ax.set_xlabel('Predicted')
    # ax.set_ylabel('Actual')
    # plt.tight_layout()
    # plt.savefig('data/2024_tourney_confusion_matrix.png')
    # plt.show()


if __name__ == "__main__":
    test_project()
